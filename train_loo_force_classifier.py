"""Predict contraction *force* from GFP volumes, framed as classification.

The numeric force column (e.g. ``peak_amplitude_week_5``) is a **per-replicate
(per-tissue) property**: every field-of-view / z-stack of the same physical
tissue shares one force value. We discretize force into ordinal bins (default
terciles: low / mid / high) and train an encoder to classify a GFP volume into
its force bin.

Bins are *cohort-relative percentile* bands (quantiles of the tissues in this
dataset), not absolute force levels — "high" means "top tercile of these
replicates", not a fixed Newton value.

Leakage control
---------------
Because force is a per-tissue label, the only honest cross-validation unit is the
*replicate* (tissue). A per-volume split would let the model memorize a tissue's
force from one FOV and read it back off another FOV of the same tissue. We
default to ``--cv_unit replicate`` (leave-one-tissue-out): all FOVs of a held-out
tissue are evaluated together, never seen in training. Within each fold, EVERY
data-driven quantity is fit on the training tissues only and never sees the
held-out tissue:
  * the bin edges (the discretization boundaries),
  * the per-bin representative force used for the expected-force correlation,
  * the inner train/val split used for early stopping.
The encoder is warm-started from a BF->GFP U-Net (``--init_from``); that U-Net
was trained without force labels, so it is unlabeled transfer learning, not a
force leak. (The preferred ``_holdPt``/``_holdEx`` encoders never saw the
perturbation tissues at all.)

Metrics
-------
  * accuracy           — argmax bin == true bin, per-replicate (headline, matches
                         the LOO unit) and per-volume.
  * per-class accuracy  + confusion matrix.
  * correlation         — between true continuous force and a continuous
                         prediction, the softmax-weighted "expected force"
                             E[force] = sum_k  p_k * mean_force(train bin k).
                         Spearman (rank) is the primary readout; on this skewed,
                         N=12 distribution Pearson is dominated by 2 outliers and
                         by bin-mean compression, so it is reported but
                         secondary (a log-force Pearson is also reported).
  * permutation test    on per-replicate accuracy (label shuffle).

Usage
-----
    python train_loo_force_classifier.py \
        -c configs/gfp_classifier.yaml \
        --input gfp --target_col peak_amplitude_week_5 \
        --n_bins 3 --cv_unit replicate \
        --init_from ckpts/unet_2d_imagenet_pearson_frac100/best.pth \
        --output results/force_from_gfp/force_2d.json
"""

import os
import re
import json
import random
import shutil
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.utils import set_seed, prepare_env
from src.data.regression_dataset import VolumeRegressionDataset
from src.data.grouping import stem_to_group
from src.models.gfp_classifier import build_gfp_classifier
from src.data import transforms as T
from extract_features import load_metadata


def build_transforms(cfg, train):
    dims = cfg["model"].get("dims", "2d")
    crop = cfg["data"]["crop_size"]
    if dims == "2d":
        if train:
            return T.Compose([
                T.RandomCrop2D(crop), T.RandomHFlip2D(), T.RandomVFlip2D(),
                T.RandomRot90_2D(), T.IntensityJitter2D(n_input_channels=1),
                T.ToTensor2D()])
        return T.Compose([T.CenterCrop2D(crop), T.ToTensor2D()])
    if train:
        return T.Compose([
            T.RandomHFlip3D(), T.RandomVFlip3D(), T.RandomZFlip3D(),
            T.RandomRot90_3D(), T.IntensityJitter3D(n_input_channels=1),
            T.ToTensor3D()])
    return T.Compose([T.ToTensor3D()])


def load_encoder_from_unet(model, ckpt_path, device):
    """Load encoder weights from a BF->GFP U-Net checkpoint into the classifier.

    Returns (n_matched, n_ckpt_keys). smp_3d's ResNet encoder overrides
    load_state_dict to do 2D->3D weight conversion and may return None even on
    success, so we compute the matched count by key intersection.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    encoder_state = {k[len("encoder."):]: v for k, v in state.items()
                     if k.startswith("encoder.")}
    if not encoder_state:
        raise RuntimeError(f"No encoder.* keys in {ckpt_path}")
    sample_conv = next((v for k, v in encoder_state.items()
                        if "conv" in k.lower() and v.ndim >= 4), None)
    if sample_conv is not None:
        ckpt_dims = "3d" if sample_conv.ndim == 5 else "2d"
        cls_sample = next((p for n, p in model.encoder.named_parameters()
                           if "conv" in n.lower() and p.ndim >= 4), None)
        cls_dims = (("3d" if cls_sample.ndim == 5 else "2d")
                    if cls_sample is not None else "?")
        if ckpt_dims != cls_dims:
            raise RuntimeError(
                f"Dim mismatch: ckpt {ckpt_path} is {ckpt_dims} but classifier "
                f"encoder is {cls_dims}. Use a checkpoint matching the config.")
    model_keys = set(model.encoder.state_dict().keys())
    n_matched = len(model_keys & set(encoder_state.keys()))
    model.encoder.load_state_dict(encoder_state, strict=False)
    return n_matched, len(encoder_state)


def parse_target(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def bin_label(i, n_bins):
    if n_bins == 2:
        return ["low", "high"][i]
    if n_bins == 3:
        return ["low", "mid", "high"][i]
    return f"q{i}"


def compute_bin_edges(values, n_bins, scheme):
    """Interior bin edges for `n_bins` classes over 1D `values`.

    quantile -> equal-count bins (terciles for n_bins=3).
    uniform  -> equal-width bins between min and max.
    Returns a length-(n_bins-1) sorted array of cut points.
    """
    values = np.asarray(values, dtype=np.float64)
    if scheme == "uniform":
        lo, hi = float(values.min()), float(values.max())
        return np.linspace(lo, hi, n_bins + 1)[1:-1]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(values, qs)


def assign_bin(value, edges):
    """Map a scalar to its bin index given interior edges (right-open bins)."""
    return int(np.searchsorted(edges, value, side="right"))


def edges_to_ranges(edges, n_bins):
    cut = list(edges)
    out = []
    for i in range(n_bins):
        lo = "-inf" if i == 0 else f"{cut[i-1]:.3f}"
        hi = "+inf" if i == n_bins - 1 else f"{cut[i]:.3f}"
        out.append(f"[{lo}, {hi})")
    return out


def _rank(a):
    """Average ranks (1..n), ties averaged — for Spearman without scipy."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(x, y):
    """Pearson r; NaN (not 0) when undefined (constant input or <2 points)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def _seed_worker(worker_id):
    """Make NumPy RNG in DataLoader workers deterministic (torch doesn't)."""
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def _eval_det(fn, eval_seed):
    """Run an eval pass with a fixed NumPy RNG so random eval patches (3D) are
    deterministic, then restore the training RNG stream untouched."""
    st = np.random.get_state()
    np.random.seed(eval_seed)
    try:
        return fn()
    finally:
        np.random.set_state(st)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default="data_mapping_drew.csv")
    p.add_argument("--target_col", default="peak_amplitude_week_5",
                   help="Numeric force column to discretize and classify")
    p.add_argument("--n_bins", type=int, default=3,
                   help="Number of force classes (default 3: low/mid/high)")
    p.add_argument("--bin_scheme", choices=["quantile", "uniform"],
                   default="quantile",
                   help="quantile=equal-count (balanced); uniform=equal-width")
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp",
                   help="Modality fed to the encoder (force-from-GFP => gfp)")
    p.add_argument("--data_dir", default=None,
                   help="Override cfg.data.data_dir (root with <input>/ + stats/)")
    p.add_argument("--init_from", default=None,
                   help="BF->GFP U-Net checkpoint to warm-start the encoder")
    p.add_argument("--output", required=True)
    p.add_argument("--cv_unit", choices=["volume", "replicate"],
                   default="replicate",
                   help="Leave-one-group-out unit. 'replicate' (default) is the "
                        "ONLY leak-free unit for a per-tissue force label.")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Label-shuffle permutation test on accuracy (0 to skip)")
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="Fraction of training groups held as inner val for early "
                        "stopping (outer held-out group never used). 0 disables.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_ckpt_dir", default=None,
                   help="If set, save each fold's best weights + bin metadata to "
                        "<dir>/<group>/best.pth")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    data_dir = args.data_dir or dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)

    metadata = load_metadata(args.metadata)

    # Force value per stem (volumes lacking the numeric column are dropped).
    forces = {}
    for stem, row in metadata.items():
        v = parse_target(row.get(args.target_col))
        if v is not None:
            forces[stem] = v

    all_stems = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob(os.path.join(mod_dir, "*.npy")))
    stems = [s for s in all_stems if s in forces]
    n_total = len(all_stems)
    n_dropped = n_total - len(stems)
    accelerator.print(
        f"using {len(stems)}/{n_total} volumes with numeric "
        f"'{args.target_col}'; dropped {n_dropped} lacking it (e.g. exercise FOVs)")
    if len(stems) < 2:
        raise SystemExit(
            f"Need >=2 vols with numeric '{args.target_col}', got {len(stems)} "
            f"(looked in {mod_dir}/)")

    # Group by CV unit. Force lives in the perturbation dataset, so the grouping
    # task hint is 'perturbation' (replicate id = "{Perturbation}_tissue={T}").
    groups = {}
    for s in stems:
        g = stem_to_group(s, metadata, args.cv_unit, task="perturbation")
        if g is None:
            accelerator.print(f"  skip {s}: no group id for cv_unit={args.cv_unit}")
            continue
        groups.setdefault(g, []).append(s)
    if len(groups) < 2:
        raise SystemExit(
            f"cv_unit={args.cv_unit} produced {len(groups)} group(s); need >= 2.")

    # Each group must be force-homogeneous (true for "{cond}_tissue={T}" since a
    # tissue has one force). Validate so a bad metadata row fails loud.
    for g, members in groups.items():
        fvals = {round(forces[s], 6) for s in members}
        if len(fvals) > 1:
            raise SystemExit(
                f"Group {g} mixes force values {fvals}; replicate grouping "
                "requires force-homogeneous groups.")

    rep_force = {g: float(forces[members[0]]) for g, members in groups.items()}
    rep_values = np.array(sorted(rep_force.values()), dtype=np.float64)
    n_bins = args.n_bins
    if len(rep_force) < n_bins:
        raise SystemExit(
            f"Only {len(rep_force)} replicates but n_bins={n_bins}; lower --n_bins.")

    classes = [bin_label(i, n_bins) for i in range(n_bins)]
    group_keys = list(groups.keys())

    # GLOBAL edges are DISPLAY-ONLY (a balanced reference for the printout / class
    # range labels). The SCORED labels and training targets are computed per fold
    # from TRAINING replicates only (see fold loop) so the held-out tissue never
    # influences its own class boundary.
    display_edges = compute_bin_edges(rep_values, n_bins, args.bin_scheme)
    display_ranges = edges_to_ranges(display_edges, n_bins)
    display_group_bin = {g: assign_bin(rep_force[g], display_edges) for g in groups}

    accelerator.print(
        f"target={args.target_col} input={args.input} cv_unit={args.cv_unit} "
        f"n_volumes={len(stems)} n_replicates={len(groups)} n_bins={n_bins} "
        f"scheme={args.bin_scheme}  (cohort-relative percentile bins)")
    accelerator.print(f"  display edges={[round(float(e),3) for e in display_edges]} "
                      f"(per-fold train-only edges used for scoring)")
    disp_counts = {i: 0 for i in range(n_bins)}
    for g in groups:
        disp_counts[display_group_bin[g]] += 1
    accelerator.print("  replicates/bin (display): "
                      + ", ".join(f"{classes[i]}{display_ranges[i]}={disp_counts[i]}"
                                  for i in range(n_bins)))

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))
    dims = cfg["model"].get("dims", "2d")
    warm_started = bool(args.init_from)
    eval_seed = seed + 9973  # fixed → deterministic eval patches

    def make_ds(stem_list, train, targets):
        return VolumeRegressionDataset(
            [os.path.join(mod_dir, f"{s}.npy") for s in stem_list],
            stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip, mode=dims,
            patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=(dcfg.get("patches_per_volume", 32)
                                if train else 8),
            crop_size=dcfg.get("crop_size", 256), modality=args.input)

    epochs = tcfg.get("epochs", 100)
    patience = tcfg.get("patience", 15)
    min_delta = tcfg.get("min_delta", 1e-3)
    lr = tcfg["lr"]

    results = []  # one entry per held-out replicate
    for fold_idx, held_g in enumerate(group_keys):
        held_stems = groups[held_g]
        train_groups = [g for g in group_keys if g != held_g]
        train_stems = [s for g in train_groups for s in groups[g]]

        # ── Per-fold, TRAIN-ONLY discretization (no held-out leak) ──
        fold_edges = compute_bin_edges(
            [rep_force[g] for g in train_groups], n_bins, args.bin_scheme)
        true_bin = assign_bin(rep_force[held_g], fold_edges)
        # integer-bin targets (as float) for every stem, from train-only edges
        fold_targets = {s: float(assign_bin(forces[s], fold_edges)) for s in forces}
        fold_group_bin = {g: assign_bin(rep_force[g], fold_edges) for g in group_keys}

        # Per-bin representative force from TRAIN replicates only (continuous
        # calibration for expected-force; leak-free).
        class_rep = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            gv = [rep_force[g] for g in train_groups if fold_group_bin[g] == b]
            class_rep[b] = (float(np.mean(gv)) if gv
                            else float(np.mean([rep_force[g] for g in train_groups])))

        # Inner split for early stopping — by GROUP, deterministic from fold idx.
        rng_split = np.random.default_rng([seed, fold_idx])
        tg = np.array(train_groups, dtype=object)
        rng_split.shuffle(tg)
        if args.inner_val_frac and args.inner_val_frac > 0 and len(tg) >= 2:
            n_iv = max(1, int(round(len(tg) * args.inner_val_frac)))
            n_iv = min(n_iv, len(tg) - 1)
            inner_val_groups = list(tg[:n_iv])
            inner_train_groups = list(tg[n_iv:])
        else:
            inner_val_groups = []
            inner_train_groups = list(tg)
        inner_train = [s for g in inner_train_groups for s in groups[g]]
        inner_val = [s for g in inner_val_groups for s in groups[g]]

        accelerator.print(
            f"\n── LOO {args.cv_unit} [{fold_idx+1}/{len(group_keys)}]: {held_g} "
            f"(true={classes[true_bin]}, force={rep_force[held_g]:.3f}, "
            f"edges={[round(float(e),3) for e in fold_edges]}, "
            f"n_held_vols={len(held_stems)}, n_inner_val_grp={len(inner_val_groups)}) ──")

        fold_ckpt_path = None
        if args.save_ckpt_dir:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(held_g))
            fold_dir = os.path.join(args.save_ckpt_dir, safe)
            os.makedirs(fold_dir, exist_ok=True)
            fold_ckpt_path = os.path.join(fold_dir, "best.pth")
            if accelerator.is_main_process:
                cfg_dst = os.path.join(fold_dir, "config.yaml")
                if not os.path.exists(cfg_dst):
                    try:
                        shutil.copy(args.config, cfg_dst)
                    except Exception as e:
                        accelerator.print(f"  warn: copy config failed ({e})")

        # Force routes through the exercise head (n_exercise = n_bins).
        model = build_gfp_classifier(cfg, n_bins, 2)
        if args.init_from:
            n_match, n_keys = load_encoder_from_unet(model, args.init_from, "cpu")
            tag = os.path.basename(os.path.dirname(args.init_from))
            accelerator.print(
                f"  warm-started encoder: matched {n_match}/{n_keys} tensors "
                f"from {tag}")
            if n_match == 0:
                raise SystemExit(
                    f"--init_from {args.init_from} matched 0 encoder tensors — "
                    "architecture mismatch? Refusing to train an un-warm-started "
                    "model while claiming a warm start.")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=tcfg.get("weight_decay", 0.01))
        criterion = nn.CrossEntropyLoss()

        gen = torch.Generator()
        gen.manual_seed(seed * 1000 + fold_idx)
        train_loader = torch.utils.data.DataLoader(
            make_ds(inner_train, True, fold_targets),
            batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
            pin_memory=True, num_workers=tcfg.get("num_workers", 4),
            worker_init_fn=_seed_worker, generator=gen)
        # Eval loaders: plain DataLoaders, num_workers=0, NOT routed through
        # Accelerate (keeps per-volume aggregation exact + RNG controllable).
        predict_loader = torch.utils.data.DataLoader(
            make_ds(held_stems, False, fold_targets),
            batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
        inner_val_loader = (torch.utils.data.DataLoader(
            make_ds(inner_val, False, fold_targets),
            batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
            if inner_val else None)

        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader)

        def eval_loss_on(loader):
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for img, tgt, _ in loader:
                    img = img.to(device)
                    lex, _lpt = model(img)
                    target = tgt.long().to(lex.device)
                    tot += criterion(lex, target).item() * img.shape[0]
                    n += img.shape[0]
            return tot / max(n, 1)

        def predict_per_vol():
            """Mean softmax per held-out VOLUME (file_idx). Returns
            (probs[n_held, n_bins], counts[n_held]). No held labels used."""
            model.eval()
            sums = np.zeros((len(held_stems), n_bins))
            counts = np.zeros(len(held_stems), dtype=int)
            with torch.no_grad():
                for img, _tgt, fidx in predict_loader:
                    img = img.to(device)
                    lex, _lpt = model(img)
                    sm = lex.softmax(dim=1).detach().cpu().numpy()
                    f = fidx.detach().cpu().numpy().reshape(-1)
                    for row, i in zip(sm, f):
                        sums[int(i)] += row
                        counts[int(i)] += 1
            probs = np.zeros_like(sums)
            valid = counts > 0
            probs[valid] = sums[valid] / counts[valid, None]
            return probs, counts

        best_sig = float("inf")
        best_probs = None
        best_counts = None
        best_epoch = 0
        no_improve = 0
        for ep in range(epochs):
            model.train()
            losses = []
            for img, tgt, _ in train_loader:
                lex, _lpt = model(img)
                target = tgt.long().to(lex.device)
                loss = criterion(lex, target)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                losses.append(loss.item())
            tr = float(np.mean(losses)) if losses else float("inf")

            if inner_val_loader is not None:
                sig = _eval_det(lambda: eval_loss_on(inner_val_loader), eval_seed)
                sig_name = "inner_val_ce"
            else:
                sig = tr; sig_name = "train_ce"

            probs, counts = _eval_det(predict_per_vol, eval_seed)

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} train_ce={tr:.4f} {sig_name}={sig:.4f}")
            if sig < best_sig - min_delta:
                best_sig, best_probs, best_counts, best_epoch = (
                    sig, probs, counts, ep + 1)
                no_improve = 0
                if fold_ckpt_path and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    tmp = fold_ckpt_path + ".tmp"
                    torch.save({
                        "epoch": best_epoch, "val_loss": float(sig),
                        "model_state_dict": unwrapped.state_dict(),
                        "head": "exercise",
                        "target_col": args.target_col,
                        "n_bins": n_bins,
                        "bin_edges": [float(e) for e in fold_edges],
                        "classes": classes, "class_rep_force": class_rep.tolist(),
                        "cv_unit": args.cv_unit, "fold": str(held_g),
                        "input": args.input,
                    }, tmp)
                    os.replace(tmp, fold_ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    accelerator.print(
                        f"  early stop ep{ep+1} ({sig_name}={best_sig:.4f} "
                        f"@ ep{best_epoch})")
                    break

        if best_probs is None:
            best_probs, best_counts = probs, counts

        # Keep only held volumes that produced >=1 patch/slice (a short z-stack
        # under z_range yields 0 rows; averaging an all-zero vector would bias
        # the expected-force correlation and per-volume accuracy).
        valid = best_counts > 0
        if not valid.any():
            raise SystemExit(
                f"Held replicate {held_g}: no volume produced any patches "
                f"(z_range={z_range} larger than every stack?).")
        for s, c in zip(held_stems, best_counts):
            if c == 0:
                accelerator.print(
                    f"  warn: held vol {s} produced 0 slices (short stack?) — "
                    "excluded from this replicate's prediction")
        held_valid = [s for s, v in zip(held_stems, valid) if v]
        vol_probs = best_probs[valid]                 # (n_valid, n_bins)

        rep_prob = vol_probs.mean(axis=0)
        rep_pred_bin = int(rep_prob.argmax())
        rep_expected_force = float(np.dot(rep_prob, class_rep))

        vol_records = []
        for s, pr in zip(held_valid, vol_probs):
            vol_records.append({
                "stem": s, "pred_bin": int(pr.argmax()),
                "expected_force": float(np.dot(pr, class_rep)),
                "probs": [float(x) for x in pr],
            })

        correct = int(rep_pred_bin == true_bin)
        accelerator.print(
            f"  pred={classes[rep_pred_bin]} (p={rep_prob[rep_pred_bin]:.3f}) "
            f"E[force]={rep_expected_force:.3f} true_force={rep_force[held_g]:.3f} "
            f"correct={correct}")
        results.append({
            "group": held_g, "stems": held_stems,
            "true_bin": true_bin, "true_class": classes[true_bin],
            "true_force": float(rep_force[held_g]),
            "pred_bin": rep_pred_bin, "pred_class": classes[rep_pred_bin],
            "expected_force": rep_expected_force,
            "correct": correct, "best_epoch": best_epoch,
            "fold_edges": [float(e) for e in fold_edges],
            "rep_probs": [float(x) for x in rep_prob],
            "class_rep_force": class_rep.tolist(),
            "per_volume": vol_records,
        })

    # ------------------------------ metrics ------------------------------
    true_bins = np.array([r["true_bin"] for r in results])
    pred_bins = np.array([r["pred_bin"] for r in results])
    true_force = np.array([r["true_force"] for r in results], dtype=np.float64)
    exp_force = np.array([r["expected_force"] for r in results], dtype=np.float64)

    rep_acc = float(np.mean(true_bins == pred_bins))

    vol_true, vol_pred = [], []
    for r in results:
        for v in r["per_volume"]:
            vol_true.append(r["true_bin"]); vol_pred.append(v["pred_bin"])
    vol_acc = (float(np.mean(np.array(vol_true) == np.array(vol_pred)))
               if vol_true else float("nan"))

    confusion = np.zeros((n_bins, n_bins), dtype=int)
    for t, pp in zip(true_bins, pred_bins):
        confusion[t, pp] += 1
    per_class = {}
    for b in range(n_bins):
        tot = int((true_bins == b).sum())
        cor = int(((true_bins == b) & (pred_bins == b)).sum())
        per_class[classes[b]] = {
            "total": tot, "correct": cor,
            "accuracy": (cor / tot) if tot else float("nan"),
            # display-only global range; actual scoring uses per-fold edges
            "force_range_display": display_ranges[b],
        }

    log_force = np.log10(np.clip(true_force, 1e-9, None))
    corr = {
        "spearman_expected_vs_force": spearman(true_force, exp_force),   # primary
        "pearson_expected_vs_force": pearson(true_force, exp_force),
        "pearson_logforce_vs_expected": pearson(log_force, exp_force),
        "spearman_predbin_vs_force": spearman(true_force, pred_bins.astype(float)),
        "pearson_predbin_vs_force": pearson(true_force, pred_bins.astype(float)),
        "_note": ("Spearman is primary; on this skewed N=12 distribution Pearson "
                  "is dominated by 2 outliers and bin-mean compression."),
    }

    chance = 1.0 / n_bins
    accelerator.print(
        f"\nReplicate-LOO accuracy: {rep_acc:.3f} "
        f"({int((true_bins==pred_bins).sum())}/{len(results)}) "
        f"chance={chance:.3f} | per-volume acc={vol_acc:.3f}")
    accelerator.print(
        f"Force correlation (E[force] vs true): "
        f"spearman={corr['spearman_expected_vs_force']:.3f} (primary) | "
        f"pearson={corr['pearson_expected_vs_force']:.3f}")

    perm_info = None
    if args.n_permutations and len(results) > 2:
        rng = np.random.default_rng(0)
        perm_acc = np.empty(args.n_permutations)
        for i in range(args.n_permutations):
            perm_acc[i] = float(np.mean(rng.permutation(true_bins) == pred_bins))
        n_ge = int(np.sum(perm_acc >= rep_acc))
        perm_info = {
            "n_permutations": int(args.n_permutations),
            "p_value_accuracy": (n_ge + 1) / (args.n_permutations + 1),
            "perm_mean": float(perm_acc.mean()), "perm_std": float(perm_acc.std()),
        }
        accelerator.print(
            f"Permutation test (accuracy): p={perm_info['p_value_accuracy']:.4f} "
            f"(perm mean={perm_acc.mean():.3f})")

    summary = {
        "task": "force_classification",
        "target_col": args.target_col,
        "input": args.input, "dims": dims,
        "init_from": args.init_from, "warm_started": warm_started,
        "seed": int(seed), "cv_unit": args.cv_unit,
        "n_bins": n_bins, "bin_scheme": args.bin_scheme,
        "edges_are_per_fold": True,
        "display_bin_edges": [float(e) for e in display_edges],
        "classes": classes, "bin_ranges": display_ranges,
        "chance": chance,
        "n_volumes": len(stems), "n_volumes_total": n_total,
        "n_volumes_dropped": n_dropped, "n_replicates": len(results),
        "replicate_accuracy": rep_acc, "volume_accuracy": vol_acc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_axes": {"rows": "true", "cols": "pred", "order": classes},
        "correlation": corr,
        "permutation_test": perm_info,
        "config_flags": {"inner_val_frac": float(args.inner_val_frac)},
        "per_replicate": results,
    }
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        accelerator.print(f"Saved {args.output}")
        make_plot(summary, os.path.splitext(args.output)[0] + ".png")
        accelerator.print(f"Saved {os.path.splitext(args.output)[0] + '.png'}")


def make_plot(summary, path):
    classes = summary["classes"]
    n_bins = summary["n_bins"]
    conf = np.array(summary["confusion_matrix"], dtype=int)
    per_class = summary["per_class"]
    results = summary["per_replicate"]
    chance = summary["chance"]
    rep_acc = summary["replicate_accuracy"]
    corr = summary["correlation"]
    dims = summary["dims"]
    perm = summary.get("permutation_test") or {}

    true_force = np.array([r["true_force"] for r in results], dtype=float)
    exp_force = np.array([r["expected_force"] for r in results], dtype=float)
    true_bin = np.array([r["true_bin"] for r in results], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (A) confusion matrix
    ax = axes[0]
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n_bins)); ax.set_yticks(range(n_bins))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    vmax = conf.max() if conf.max() > 0 else 1
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > vmax / 2 else "black",
                    fontsize=11)
    p_txt = (f"  perm p={perm['p_value_accuracy']:.3f}"
             if perm.get("p_value_accuracy") is not None else "")
    ax.set_title(f"Confusion (replicate-LOO)\nacc={rep_acc:.2f} "
                 f"chance={chance:.2f}{p_txt}")

    # (B) per-class accuracy bar
    ax = axes[1]
    ys = [per_class[c]["accuracy"] for c in classes]
    ns = [per_class[c]["total"] for c in classes]
    xs = ["overall"] + classes
    yvals = [rep_acc] + ys
    nvals = [len(results)] + ns
    colors = ["#4363d8"] + ["#3cb44b"] * n_bins
    ax.bar(xs, [0 if (y is None or np.isnan(y)) else y for y in yvals],
           color=colors)
    for i, (y, n) in enumerate(zip(yvals, nvals)):
        yy = 0 if (y is None or np.isnan(y)) else y
        ax.text(i, yy + 0.02, f"{yy:.2f}\n(n={n})", ha="center", fontsize=9)
    ax.axhline(chance, color="gray", linestyle=":", label=f"chance={chance:.2f}")
    ax.set_ylim(0, 1.18); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by force class"); ax.legend(loc="upper right", fontsize=8)

    # (C) true vs predicted (expected) force
    ax = axes[2]
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
    for b in range(n_bins):
        m = true_bin == b
        if m.any():
            ax.scatter(true_force[m], exp_force[m], s=60, alpha=0.8,
                       color=palette[b % len(palette)], label=classes[b],
                       edgecolor="k", linewidth=0.4)
    allv = np.concatenate([true_force, exp_force])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", ls=":")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"True {summary['target_col']}")
    ax.set_ylabel("Predicted E[force]")
    sp = corr.get("spearman_expected_vs_force")
    pe = corr.get("pearson_expected_vs_force")
    ax.set_title(f"True vs predicted force\n"
                 f"spearman={sp:.2f} (primary)  pearson={pe:.2f}")
    ax.legend(fontsize=8, title="true class"); ax.grid(True, alpha=0.3)

    enc = "warm-started BF->GFP encoder" if summary.get("warm_started") \
        else "ImageNet encoder (NO BF->GFP warm-start)"
    fig.suptitle(f"Force-from-GFP ({dims.upper()}) | {summary['target_col']} | "
                 f"{summary['n_replicates']} replicates, {n_bins} bins "
                 f"({summary['bin_scheme']}) | {enc}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
