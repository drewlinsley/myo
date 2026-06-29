"""Predict contraction *force* from GFP volumes on the NEW (phalloidin/MHC 051826)
drop, framed as classification, with a single replicate-level train/val/test split
(NOT leave-one-out).

Use this when you want to train *and* test on one freshly collected dataset:
the model never sees the old perturbation tissues. Force (``peak_amplitude_week3``
by default) is a per-replicate (per-tissue) label, so the split is by *replicate*
(``--group_cols plate,Tissue``): every FOV of a tissue lands wholly in train, val,
or test — never split — so a tissue's force can't leak from one FOV to another.

What stays leak-free
--------------------
  * The split is by replicate (above).
  * The scoring bin edges and the per-bin representative force (used for the
    expected-force correlation) are fit on the TRAIN replicates only.
  * Early stopping uses the VAL split (or train loss if no val); the TEST split is
    only ever *measured*, never used for model selection.
The encoder is warm-started from a BF->GFP U-Net (``--init_from``); that U-Net saw
no force labels, so it is unlabeled transfer, not a force leak.

Not enough replicates for three way? Pass ``--val_frac 0`` (or let it auto-fall
back) for a plain train/test split.

Dry run first
-------------
``--plan_only`` builds the metadata match, the replicate groups, and the split,
prints/saves the plan, and exits WITHOUT training — run it on the VM to confirm
match coverage and that each split spans the force range before spending GPU.

Usage
-----
    python train_split_force_classifier.py \
        -c configs/gfp_classifier_3d.yaml \
        --data_dir data_phalloidin_mhc_051826_staged \
        --metadata "phalloidin_mhc_mapping_051426_SS edit.xlsx" \
        --target_col peak_amplitude_week3 --group_cols plate,Tissue \
        --n_bins 3 --init_from ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
        --output results/force_from_gfp_new/force_3d.json
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.utils import set_seed, prepare_env
from src.data.regression_dataset import VolumeRegressionDataset
from src.models.gfp_classifier import build_gfp_classifier
from src.data.force_metadata import build_force_groups
from train_loo_force_classifier import (
    compute_bin_edges, assign_bin, bin_label, edges_to_ranges,
    pearson, spearman, _seed_worker, _eval_det, load_encoder_from_unet,
    build_transforms)


def split_replicates(group_keys, rep_force, n_bins, test_frac, val_frac, seed):
    """Stratified replicate split. Strata are GLOBAL force terciles (so each of
    train/val/test spans the force range); strata are split-only — scoring edges
    are recomputed on TRAIN later, so this stratification leaks nothing into the
    labels. Every stratum with >=2 members yields >=1 test replicate; train is
    never starved (keeps >=1 per stratum). val may end up empty on small data."""
    rng = np.random.default_rng([int(seed), 20240617])
    forces = [rep_force[g] for g in group_keys]
    strat_edges = compute_bin_edges(forces, n_bins, "quantile")
    strata = {}
    for g in group_keys:
        strata.setdefault(assign_bin(rep_force[g], strat_edges), []).append(g)
    train, val, test = [], [], []
    for b in sorted(strata):
        gs = sorted(strata[b])
        rng.shuffle(gs)
        n = len(gs)
        n_test = int(np.floor(test_frac * n + 0.5))   # round-half-up (not banker's)
        n_val = int(np.floor(val_frac * n + 0.5))
        if test_frac > 0 and n >= 2 and n_test == 0:
            n_test = 1                                 # ensure test spans every stratum
        while n_test + n_val >= n and (n_test + n_val) > 0:
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        test += gs[:n_test]
        val += gs[n_test:n_test + n_val]
        train += gs[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test), [float(e) for e in strat_edges]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", required=True,
                   help="Mapping spreadsheet (.xlsx/.csv) with file + force cols")
    p.add_argument("--target_col", default="peak_amplitude_week3",
                   help="Force column to discretize and classify")
    p.add_argument("--file_col", default="file",
                   help="Column in the spreadsheet holding the volume filename")
    p.add_argument("--group_cols", default="plate,Tissue",
                   help="Comma-separated columns identifying a replicate (tissue)")
    p.add_argument("--n_bins", type=int, default=3,
                   help="Number of force classes (default 3: low/mid/high)")
    p.add_argument("--bin_scheme", choices=["quantile", "uniform"],
                   default="quantile")
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--data_dir", default=None,
                   help="Staged dataset root (has <input>/ + stats/)")
    p.add_argument("--init_from", default=None,
                   help="BF->GFP U-Net checkpoint to warm-start the encoder")
    p.add_argument("--output", required=True)
    p.add_argument("--test_frac", type=float, default=0.25,
                   help="Fraction of replicates held out for TEST")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Fraction of replicates for VAL (early stopping). "
                        "0 -> plain train/test.")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Label-shuffle permutation test on test accuracy (0 skip)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_ckpt", default=None,
                   help="If set, save best (val-selected) weights + bin metadata")
    p.add_argument("--plan_only", action="store_true",
                   help="Build metadata match + groups + split, print/save the "
                        "plan, and exit WITHOUT training.")
    p.add_argument("--allow_partial_match", action="store_true",
                   help="Proceed even if some force-labeled metadata rows fail to "
                        "match a staged volume (default: refuse, to catch a bad "
                        "filename mapping before training).")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    data_dir = args.data_dir or dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)
    dims = cfg["model"].get("dims", "2d")
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    # ---- metadata match + replicate groups (no torch needed yet) ----
    data = build_force_groups(
        args.metadata, data_dir, args.target_col,
        file_col=args.file_col, group_cols=tuple(group_cols),
        modality=args.input)
    forces_per_stem = data["forces"]
    groups = data["groups"]
    rep_force = data["rep_force"]
    print("\n".join(data["report"]))

    if data["n_matched"] == 0:
        raise SystemExit(
            "No metadata force rows matched any staged volume — check --file_col "
            "and that the spreadsheet filenames correspond to the staged .npy "
            "names (see the unmatched examples above).")
    if data["unmatched_meta"] and not args.allow_partial_match:
        raise SystemExit(
            f"{len(data['unmatched_meta'])} force-labeled metadata row(s) matched "
            "NO staged volume (examples above). Fix the filename mapping, or pass "
            "--allow_partial_match to train on the matched subset anyway.")

    n_bins = args.n_bins
    group_keys = sorted(groups.keys())
    if len(group_keys) < n_bins:
        raise SystemExit(
            f"Only {len(group_keys)} replicate(s) but n_bins={n_bins}; "
            f"lower --n_bins or coarsen --group_cols.")

    # ---- replicate split ----
    train_g, val_g, test_g, strat_edges = split_replicates(
        group_keys, rep_force, n_bins, args.test_frac, args.val_frac, seed)

    # Small-data guard: don't starve train for a val split.
    min_train = max(n_bins, 3)
    if val_g and len(train_g) < min_train:
        print(f"  note: only {len(train_g)} train replicates after carving val; "
              f"folding val back into train (train/test only).")
        train_g = sorted(train_g + val_g)
        val_g = []
    if not test_g:
        raise SystemExit(
            "Split produced an empty TEST set — too few replicates. Lower "
            "--n_bins or raise --test_frac.")
    if len(train_g) < n_bins:
        raise SystemExit(
            f"Only {len(train_g)} train replicate(s) for {n_bins} bins — "
            "lower --n_bins or --test_frac/--val_frac.")

    # ---- TRAIN-only scoring discretization + calibration ----
    edges = compute_bin_edges([rep_force[g] for g in train_g], n_bins,
                              args.bin_scheme)
    ranges = edges_to_ranges(edges, n_bins)
    classes = [bin_label(i, n_bins) for i in range(n_bins)]
    group_bin = {g: assign_bin(rep_force[g], edges) for g in group_keys}
    targets = {s: float(assign_bin(forces_per_stem[s], edges))
               for s in forces_per_stem}
    class_rep = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        gv = [rep_force[g] for g in train_g if group_bin[g] == b]
        class_rep[b] = (float(np.mean(gv)) if gv
                        else float(np.mean([rep_force[g] for g in train_g])))

    def split_summary(name, gs):
        bincnt = {i: 0 for i in range(n_bins)}
        for g in gs:
            bincnt[group_bin[g]] += 1
        nvol = sum(len(groups[g]) for g in gs)
        comp = ", ".join(f"{classes[i]}={bincnt[i]}" for i in range(n_bins))
        return f"{name}: {len(gs)} reps / {nvol} vols  [{comp}]"

    print(f"\ntarget={args.target_col} input={args.input} dims={dims} "
          f"n_bins={n_bins} scheme={args.bin_scheme}")
    print(f"  train-only scoring edges={[round(float(e),3) for e in edges]}  "
          f"class_rep_force={[round(float(x),3) for x in class_rep]}")
    print("  " + split_summary("train", train_g))
    print("  " + split_summary("val  ", val_g))
    print("  " + split_summary("test ", test_g))
    test_true_bins = sorted({group_bin[g] for g in test_g})
    if len(test_true_bins) < 2:
        print("  WARNING: test replicates span <2 force classes — accuracy / "
              "correlation on this split will be weakly informative.")

    plan = {
        "task": "force_classification_split",
        "target_col": args.target_col, "input": args.input, "dims": dims,
        "n_bins": n_bins, "bin_scheme": args.bin_scheme,
        "group_cols": group_cols,
        "n_replicates": len(group_keys),
        "n_volumes_matched": data["n_matched"],
        "metadata_columns": data["columns"],
        "stratify_edges": strat_edges,
        "scoring_bin_edges": [float(e) for e in edges],
        "bin_ranges": ranges, "classes": classes,
        "class_rep_force": class_rep.tolist(),
        "split": {
            "train": [{"group": g, "force": rep_force[g],
                       "bin": int(group_bin[g]), "n_vols": len(groups[g])}
                      for g in train_g],
            "val": [{"group": g, "force": rep_force[g],
                     "bin": int(group_bin[g]), "n_vols": len(groups[g])}
                    for g in val_g],
            "test": [{"group": g, "force": rep_force[g],
                      "bin": int(group_bin[g]), "n_vols": len(groups[g])}
                     for g in test_g],
        },
        "match_report": data["report"],
        "n_unmatched_meta": len(data["unmatched_meta"]),
        "n_unmatched_staged": len(data["unmatched_staged"]),
    }

    if args.plan_only:
        out = os.path.splitext(args.output)[0] + ".plan.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\n[plan_only] wrote {out} — review match coverage + split, then "
              "rerun without --plan_only to train.")
        return

    # ---------------------------- training ----------------------------
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))
    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))
    warm_started = bool(args.init_from)
    eval_seed = seed + 9973

    train_stems = [s for g in train_g for s in groups[g]]
    val_stems = [s for g in val_g for s in groups[g]]
    test_stems = [s for g in test_g for s in groups[g]]

    def make_ds(stem_list, train):
        return VolumeRegressionDataset(
            [os.path.join(mod_dir, f"{s}.npy") for s in stem_list],
            stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip, mode=dims,
            patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=(dcfg.get("patches_per_volume", 32) if train else 8),
            crop_size=dcfg.get("crop_size", 256), modality=args.input)

    epochs = tcfg.get("epochs", 100)
    patience = tcfg.get("patience", 15)
    min_delta = tcfg.get("min_delta", 1e-3)
    lr = tcfg["lr"]

    model = build_gfp_classifier(cfg, n_bins, 2)   # force routes thru exercise head
    if args.init_from:
        n_match, n_keys = load_encoder_from_unet(model, args.init_from, "cpu")
        tag = os.path.basename(os.path.dirname(args.init_from))
        accelerator.print(
            f"warm-started encoder: matched {n_match}/{n_keys} tensors from {tag}")
        if n_match == 0:
            raise SystemExit(
                f"--init_from {args.init_from} matched 0 encoder tensors — "
                "architecture mismatch? Refusing a fake warm start.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tcfg.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(seed * 1000 + 1)
    train_loader = torch.utils.data.DataLoader(
        make_ds(train_stems, True),
        batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        pin_memory=True, num_workers=tcfg.get("num_workers", 4),
        worker_init_fn=_seed_worker, generator=gen)
    test_loader = torch.utils.data.DataLoader(
        make_ds(test_stems, False),
        batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
    val_loader = (torch.utils.data.DataLoader(
        make_ds(val_stems, False),
        batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)
        if val_stems else None)

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

    def predict_probs(loader, n_stems):
        """Mean softmax per volume (file_idx local to this loader's stem list)."""
        model.eval()
        sums = np.zeros((n_stems, n_bins))
        counts = np.zeros(n_stems, dtype=int)
        with torch.no_grad():
            for img, _tgt, fidx in loader:
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
    best_probs = best_counts = None
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

        if val_loader is not None:
            sig = _eval_det(lambda: eval_loss_on(val_loader), eval_seed)
            sig_name = "val_ce"
        else:
            sig, sig_name = tr, "train_ce"

        probs, counts = _eval_det(lambda: predict_probs(test_loader,
                                                        len(test_stems)), eval_seed)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            accelerator.print(
                f"  ep{ep+1}/{epochs} train_ce={tr:.4f} {sig_name}={sig:.4f}")
        if sig < best_sig - min_delta:
            best_sig, best_probs, best_counts, best_epoch = sig, probs, counts, ep + 1
            no_improve = 0
            if args.save_ckpt and accelerator.is_main_process:
                os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                tmp = args.save_ckpt + ".tmp"
                torch.save({
                    "epoch": best_epoch, "val_loss": float(sig),
                    "model_state_dict": unwrapped.state_dict(), "head": "exercise",
                    "target_col": args.target_col, "n_bins": n_bins,
                    "bin_edges": [float(e) for e in edges], "classes": classes,
                    "class_rep_force": class_rep.tolist(), "input": args.input,
                    "split": "train_val_test",
                }, tmp)
                os.replace(tmp, args.save_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                accelerator.print(
                    f"  early stop ep{ep+1} ({sig_name}={best_sig:.4f} "
                    f"@ ep{best_epoch})")
                break

    if best_probs is None:
        best_probs, best_counts = probs, counts

    valid = best_counts > 0
    for s, c in zip(test_stems, best_counts):
        if c == 0:
            accelerator.print(f"  warn: test vol {s} produced 0 slices "
                              "(short stack?) — excluded")
    if not valid.any():
        raise SystemExit("No test volume produced any patches "
                         f"(z_range={z_range} larger than every stack?).")

    # ---- aggregate per test volume, then per test replicate ----
    stem_to_probs = {s: best_probs[i] for i, s in enumerate(test_stems) if valid[i]}
    results = []
    for g in test_g:
        member_probs = [stem_to_probs[s] for s in groups[g] if s in stem_to_probs]
        if not member_probs:
            accelerator.print(f"  warn: test replicate {g} had 0 valid vols — skipped")
            continue
        vp = np.array(member_probs)
        rep_prob = vp.mean(axis=0)
        rep_pred_bin = int(rep_prob.argmax())
        true_bin = group_bin[g]
        rep_expected = float(np.dot(rep_prob, class_rep))
        vol_records = [{"stem": s, "pred_bin": int(p.argmax()),
                        "expected_force": float(np.dot(p, class_rep)),
                        "probs": [float(x) for x in p]}
                       for s, p in zip([s for s in groups[g] if s in stem_to_probs], vp)]
        results.append({
            "group": g, "true_bin": int(true_bin), "true_class": classes[true_bin],
            "true_force": float(rep_force[g]), "pred_bin": rep_pred_bin,
            "pred_class": classes[rep_pred_bin], "expected_force": rep_expected,
            "correct": int(rep_pred_bin == true_bin),
            "rep_probs": [float(x) for x in rep_prob],
            "per_volume": vol_records,
        })

    if not results:
        raise SystemExit("No test replicate produced a prediction.")

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
            "force_range_display": ranges[b]}

    log_force = np.log10(np.clip(true_force, 1e-9, None))
    corr = {
        "spearman_expected_vs_force": spearman(true_force, exp_force),
        "pearson_expected_vs_force": pearson(true_force, exp_force),
        "pearson_logforce_vs_expected": pearson(log_force, exp_force),
        "spearman_predbin_vs_force": spearman(true_force, pred_bins.astype(float)),
        "_note": ("Spearman is primary; Pearson is sensitive to outliers and "
                  "bin-mean compression on small/skewed test sets."),
    }

    chance = 1.0 / n_bins
    accelerator.print(
        f"\nTEST replicate accuracy: {rep_acc:.3f} "
        f"({int((true_bins==pred_bins).sum())}/{len(results)}) chance={chance:.3f} "
        f"| per-volume acc={vol_acc:.3f}")
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
            "perm_mean": float(perm_acc.mean()), "perm_std": float(perm_acc.std())}
        accelerator.print(
            f"Permutation test (accuracy): p={perm_info['p_value_accuracy']:.4f} "
            f"(perm mean={perm_acc.mean():.3f})")
    elif args.n_permutations:
        accelerator.print(
            f"(permutation test skipped: only {len(results)} test replicates)")

    summary = dict(plan)
    summary.update({
        "init_from": args.init_from, "warm_started": warm_started,
        "seed": int(seed),
        "n_test_replicates": len(results),
        "replicate_accuracy": rep_acc, "volume_accuracy": vol_acc,
        "chance": chance,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_axes": {"rows": "true", "cols": "pred", "order": classes},
        "correlation": corr, "permutation_test": perm_info,
        "best_epoch": best_epoch,
        "config_flags": {"test_frac": float(args.test_frac),
                         "val_frac": float(args.val_frac),
                         "has_val": bool(val_stems)},
        "per_test_replicate": results,
    })
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
    results = summary["per_test_replicate"]
    chance = summary["chance"]
    rep_acc = summary["replicate_accuracy"]
    corr = summary["correlation"]
    dims = summary["dims"]
    perm = summary.get("permutation_test") or {}

    true_force = np.array([r["true_force"] for r in results], dtype=float)
    exp_force = np.array([r["expected_force"] for r in results], dtype=float)
    true_bin = np.array([r["true_bin"] for r in results], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n_bins)); ax.set_yticks(range(n_bins))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    vmax = conf.max() if conf.max() > 0 else 1
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > vmax / 2 else "black", fontsize=11)
    p_txt = (f"  perm p={perm['p_value_accuracy']:.3f}"
             if perm.get("p_value_accuracy") is not None else "")
    ax.set_title(f"Confusion (held-out test)\nacc={rep_acc:.2f} "
                 f"chance={chance:.2f}{p_txt}")

    ax = axes[1]
    ys = [per_class[c]["accuracy"] for c in classes]
    ns = [per_class[c]["total"] for c in classes]
    xs = ["overall"] + classes
    yvals = [rep_acc] + ys
    nvals = [len(results)] + ns
    colors = ["#4363d8"] + ["#3cb44b"] * n_bins
    ax.bar(xs, [0 if (y is None or np.isnan(y)) else y for y in yvals], color=colors)
    for i, (y, n) in enumerate(zip(yvals, nvals)):
        yy = 0 if (y is None or np.isnan(y)) else y
        ax.text(i, yy + 0.02, f"{yy:.2f}\n(n={n})", ha="center", fontsize=9)
    ax.axhline(chance, color="gray", linestyle=":", label=f"chance={chance:.2f}")
    ax.set_ylim(0, 1.18); ax.set_ylabel("Accuracy")
    ax.set_title("Test accuracy by force class")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
    for b in range(n_bins):
        m = true_bin == b
        if m.any():
            ax.scatter(true_force[m], exp_force[m], s=60, alpha=0.8,
                       color=palette[b % len(palette)], label=classes[b],
                       edgecolor="k", linewidth=0.4)
    allv = np.concatenate([true_force, exp_force]) if len(true_force) else np.array([0, 1])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", ls=":")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"True {summary['target_col']}")
    ax.set_ylabel("Predicted E[force]")
    sp = corr.get("spearman_expected_vs_force")
    pe = corr.get("pearson_expected_vs_force")
    ax.set_title(f"True vs predicted force (test)\n"
                 f"spearman={sp:.2f} (primary)  pearson={pe:.2f}")
    ax.legend(fontsize=8, title="true class"); ax.grid(True, alpha=0.3)

    enc = "warm-started BF->GFP encoder" if summary.get("warm_started") \
        else "ImageNet encoder (NO BF->GFP warm-start)"
    n_tr = len(summary["split"]["train"]); n_te = len(summary["split"]["test"])
    fig.suptitle(f"Force-from-GFP ({dims.upper()}, new-data split) | "
                 f"{summary['target_col']} | train {n_tr} / test {n_te} reps, "
                 f"{n_bins} bins ({summary['bin_scheme']}) | {enc}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
