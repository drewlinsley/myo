"""Leave-one-out regression on a numeric metadata column (e.g. Wk5 amplitude).

Mirrors train_loo_classifier.py but with a scalar regression head, MSE loss,
and held-out metrics = Pearson correlation + MSE (computed across all held-out
predictions vs ground-truth scalars).

Usage:
    python train_loo_regression.py \
        -c configs/gfp_classifier_3d.yaml \
        --metadata data_mapping_drew.csv \
        --target_col peak_amplitude_week_5 \
        --input bf \
        --init_from ckpts/unet_3d_imagenet_pearson_frac100_holdPt/best.pth \
        --output results/loo_reg/holdPt_frac100_wk5_bf.json
"""

import os
import re
import json
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
from src.models.gfp_regressor import build_gfp_regressor
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
                f"Dim mismatch: ckpt {ckpt_path} is {ckpt_dims} but model "
                f"encoder is {cls_dims}.")
    model.encoder.load_state_dict(encoder_state, strict=False)
    return len(encoder_state)


def parse_target(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default="data_mapping_drew.csv")
    p.add_argument("--target_col", default="peak_amplitude_week_5",
                   help="Numeric metadata column to regress")
    p.add_argument("--input", choices=["bf", "gfp"], default="bf")
    p.add_argument("--init_from", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cv_unit", choices=["volume", "replicate"],
                   default="volume")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Permutation test on Pearson (0 to skip)")
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="Fraction of train_stems set aside as inner val for "
                        "early stopping (so the outer held-out group is NEVER "
                        "used to select epochs). 0 disables and runs full epochs.")
    p.add_argument("--peek_val_for_earlystop", action="store_true",
                   help="(Legacy / debug) early-stop on held-out loss. Leaks "
                        "the test target — kept only for back-compat.")
    p.add_argument("--save_ckpt_dir", default=None,
                   help="If set, save each fold's best-epoch weights to "
                        "<save_ckpt_dir>/<group>/best.pth (with t_mean/t_std "
                        "for de-normalization). Use with predict_regression.py "
                        "--ckpts <glob> for ensemble inference on new data.")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)

    metadata = load_metadata(args.metadata)
    targets = {}
    for stem, row in metadata.items():
        v = parse_target(row.get(args.target_col))
        if v is not None:
            targets[stem] = v

    all_stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob(os.path.join(mod_dir, "*.npy"))])
    stems = [s for s in all_stems if s in targets]
    if len(stems) < 2:
        raise SystemExit(
            f"Need >=2 vols with target '{args.target_col}', got {len(stems)}")

    # Determine task hint for grouping (uses Perturbation column for the
    # tissue-uniqueness check when present; perturbation default is fine).
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
    accelerator.print(
        f"target={args.target_col} input={args.input} cv_unit={args.cv_unit} "
        f"n_volumes={len(stems)} n_groups={len(groups)}")

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))
    dims = cfg["model"].get("dims", "2d")

    def make_ds(stem_list, train):
        paths = [os.path.join(mod_dir, f"{s}.npy") for s in stem_list]
        return VolumeRegressionDataset(
            paths, stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip,
            mode=dims, patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=(dcfg.get("patches_per_volume", 32)
                                if train else 8),
            crop_size=dcfg.get("crop_size", 256), modality=args.input)

    epochs = tcfg.get("epochs", 50)
    patience = tcfg.get("patience", 10)
    min_delta = tcfg.get("min_delta", 1e-3)
    lr = tcfg["lr"]

    # Whole-dataset stats (for the predict-the-mean baseline in the summary).
    target_arr = np.array([targets[s] for s in stems], dtype=np.float64)
    overall_mean = float(target_arr.mean())
    overall_std = float(target_arr.std() or 1.0)
    accelerator.print(
        f"target stats (all stems): mean={overall_mean:.4f} "
        f"std={overall_std:.4f} (predict-mean RMSE baseline = {overall_std:.4f})")

    per_vol = []  # one entry per held-out vol across all groups
    for held_g, held_stems in groups.items():
        train_stems = [s for g, ss in groups.items() if g != held_g for s in ss]
        # Per-fold normalization — strictly train-only (no held-out leak).
        train_targets = np.array(
            [targets[s] for s in train_stems], dtype=np.float64)
        t_mean = float(train_targets.mean())
        t_std = float(train_targets.std() or 1.0)
        norm_targets = {s: (targets[s] - t_mean) / t_std for s in targets}

        # Inner train/val split for early stopping (no peek at outer held-out).
        rng_split = np.random.default_rng(seed + hash(held_g) % (2 ** 31))
        train_stems_arr = np.array(train_stems)
        rng_split.shuffle(train_stems_arr)
        if args.peek_val_for_earlystop or args.inner_val_frac <= 0:
            inner_train = list(train_stems_arr)
            inner_val = []
        else:
            n_val = max(1, int(round(len(train_stems_arr) * args.inner_val_frac)))
            inner_val = list(train_stems_arr[:n_val])
            inner_train = list(train_stems_arr[n_val:])

        accelerator.print(
            f"\n── LOO group: {held_g} (n_held={len(held_stems)}, "
            f"n_inner_train={len(inner_train)}, n_inner_val={len(inner_val)}, "
            f"t_mean={t_mean:.3f}, t_std={t_std:.3f}) ──")

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

        model = build_gfp_regressor(cfg)
        if args.init_from:
            n_load = load_encoder_from_unet(model, args.init_from, "cpu")
            accelerator.print(f"  loaded {n_load} encoder tensors")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=tcfg.get("weight_decay", 0.01))
        criterion = nn.MSELoss()

        def make_loader(stem_list, train):
            ds = VolumeRegressionDataset(
                [os.path.join(mod_dir, f"{s}.npy") for s in stem_list],
                stats_dir=stats_dir, targets=norm_targets,
                transform=build_transforms(cfg, train),
                z_range=z_range, apply_timm=apply_timm,
                percentile_clip=percentile_clip, mode=dims,
                patch_depth=dcfg.get("patch_depth", 32),
                patches_per_volume=(dcfg.get("patches_per_volume", 32)
                                    if train else 8),
                crop_size=dcfg.get("crop_size", 256), modality=args.input)
            return torch.utils.data.DataLoader(
                ds, batch_size=tcfg["batch_size"],
                shuffle=train, drop_last=train, pin_memory=train,
                num_workers=tcfg.get("num_workers", 4))

        train_loader = make_loader(inner_train, True)
        inner_val_loader = (make_loader(inner_val, False)
                            if inner_val else None)
        predict_loader = make_loader(held_stems, False)

        to_prep = [model, optimizer, train_loader, predict_loader]
        if inner_val_loader is not None:
            to_prep.append(inner_val_loader)
        prepared = accelerator.prepare(*to_prep)
        model, optimizer, train_loader, predict_loader = prepared[:4]
        if inner_val_loader is not None:
            inner_val_loader = prepared[4]

        def eval_loss_on(loader):
            model.eval()
            tot_loss, tot_n = 0.0, 0
            with torch.no_grad():
                for img, tgt, _ in loader:
                    out = model(img)
                    loss = criterion(out, tgt.float().to(out.device))
                    tot_loss += loss.item() * img.shape[0]
                    tot_n += img.shape[0]
            return tot_loss / max(tot_n, 1)

        def predict_on_held():
            """Predictions for held_stems — used for output only, NEVER
            for epoch selection. No held-out target involved."""
            model.eval()
            sums = {i: 0.0 for i in range(len(held_stems))}
            counts = {i: 0 for i in range(len(held_stems))}
            with torch.no_grad():
                for img, _tgt, fidx in predict_loader:
                    out = model(img)
                    o = out.detach().cpu().numpy().reshape(-1)
                    f = fidx.detach().cpu().numpy().reshape(-1)
                    for v, i in zip(o, f):
                        sums[int(i)] += float(v)
                        counts[int(i)] += 1
            return np.array(
                [sums[i] / max(counts[i], 1) for i in range(len(held_stems))])

        # Epoch selection signal:
        #   - if --peek_val_for_earlystop: held-out loss (LEAK, legacy)
        #   - elif inner_val: inner_val MSE (clean)
        #   - else: train MSE (no peek, but weak signal)
        best_sig = float("inf")
        best_preds = None
        best_epoch = 0
        no_improve = 0
        for ep in range(epochs):
            model.train()
            losses = []
            for img, tgt, _ in train_loader:
                out = model(img)
                loss = criterion(out, tgt.float().to(out.device))
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                losses.append(loss.item())
            tr = float(np.mean(losses)) if losses else float("inf")

            if args.peek_val_for_earlystop:
                sig = eval_loss_on(predict_loader)  # LEAK
                sig_name = "held_val_mse"
            elif inner_val_loader is not None:
                sig = eval_loss_on(inner_val_loader)
                sig_name = "inner_val_mse"
            else:
                sig = tr
                sig_name = "train_mse"

            # Predict on held-out at current epoch (no target leak)
            preds_norm = predict_on_held()

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} train_mse={tr:.4f} "
                    f"{sig_name}={sig:.4f}")
            if sig < best_sig - min_delta:
                best_sig = sig
                best_preds = preds_norm
                best_epoch = ep + 1
                no_improve = 0
                if fold_ckpt_path and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    tmp = fold_ckpt_path + ".tmp"
                    torch.save({
                        "epoch": best_epoch,
                        "val_loss": float(sig),
                        "model_state_dict": unwrapped.state_dict(),
                        "t_mean": float(t_mean),
                        "t_std": float(t_std),
                        "target_col": args.target_col,
                        "cv_unit": args.cv_unit,
                        "fold": str(held_g),
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

        # De-normalize predictions back to raw target scale (per-fold stats).
        preds_raw = best_preds * t_std + t_mean
        for s, pred in zip(held_stems, preds_raw):
            per_vol.append({
                "stem": s, "group": held_g,
                "true": float(targets[s]),
                "pred": float(pred),
                "best_epoch": best_epoch,
                "best_inner_val_loss": float(best_sig),
                "fold_train_mean": t_mean,
                "fold_train_std": t_std,
            })

    # Overall metrics (in raw target units — predictions were de-normalized)
    trues = np.array([r["true"] for r in per_vol], dtype=np.float64)
    preds = np.array([r["pred"] for r in per_vol], dtype=np.float64)
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - trues)))
    if trues.std() == 0 or preds.std() == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(trues, preds)[0, 1])

    # Baseline: predict-the-mean (per-fold). Lower bound a real model must beat.
    baseline_preds = np.array(
        [r["fold_train_mean"] for r in per_vol], dtype=np.float64)
    baseline_mse = float(np.mean((baseline_preds - trues) ** 2))
    baseline_rmse = float(np.sqrt(baseline_mse))
    baseline_mae = float(np.mean(np.abs(baseline_preds - trues)))

    accelerator.print(
        f"\nOverall: pearson={pearson:.3f} rmse={rmse:.4f} "
        f"mse={mse:.4f} mae={mae:.4f} (n={len(per_vol)})")
    accelerator.print(
        f"Baseline (predict fold-train mean): rmse={baseline_rmse:.4f} "
        f"mse={baseline_mse:.4f} mae={baseline_mae:.4f}")
    if rmse > baseline_rmse:
        accelerator.print(
            "  WARNING: model RMSE is WORSE than the predict-mean baseline. "
            "Encoder features may be uninformative for this target.")

    perm_info = None
    if args.n_permutations and len(per_vol) > 2:
        rng = np.random.default_rng(0)
        perm_pearsons = np.empty(args.n_permutations, dtype=float)
        for i in range(args.n_permutations):
            shuf = rng.permutation(trues)
            if shuf.std() == 0 or preds.std() == 0:
                perm_pearsons[i] = 0.0
            else:
                perm_pearsons[i] = float(np.corrcoef(shuf, preds)[0, 1])
        n_ge = int(np.sum(perm_pearsons >= pearson))
        p_value = (n_ge + 1) / (args.n_permutations + 1)
        perm_info = {
            "n_permutations": int(args.n_permutations),
            "p_value_pearson": float(p_value),
            "perm_mean": float(perm_pearsons.mean()),
            "perm_std": float(perm_pearsons.std()),
        }
        accelerator.print(
            f"Permutation test (Pearson): p={p_value:.4f}")

    summary = {
        "task": "regression",
        "target_col": args.target_col,
        "input": args.input,
        "init_from": args.init_from,
        "seed": int(seed),
        "cv_unit": args.cv_unit,
        "n_groups": len(groups),
        "n_volumes": len(per_vol),
        "metrics": {
            "pearson": pearson, "mse": mse, "rmse": rmse, "mae": mae,
            "target_mean": overall_mean, "target_std": overall_std,
        },
        "baseline_predict_mean": {
            "rmse": baseline_rmse, "mse": baseline_mse, "mae": baseline_mae,
        },
        "config_flags": {
            "inner_val_frac": float(args.inner_val_frac),
            "peek_val_for_earlystop": bool(args.peek_val_for_earlystop),
        },
        "permutation_test": perm_info,
        "per_volume": per_vol,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"Saved {args.output}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(trues, preds, alpha=0.7, color="#4363d8")
    lo, hi = float(min(trues.min(), preds.min())), float(max(trues.max(), preds.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", linestyle=":")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"True {args.target_col}")
    ax.set_ylabel("Predicted")
    ax.set_title(f"LOO regression | pearson={pearson:.3f} "
                 f"rmse={rmse:.4f} (n={len(per_vol)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    accelerator.print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
