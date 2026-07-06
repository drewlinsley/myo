"""Predict CONTINUOUS contraction force from GFP (or BF) volumes — regression
variant of train_split_force_classifier.py.

Same leak-free machinery: replicate train/val/test split, warm-start / freeze
(linear probe), and the shared split-manifest (so it drops into the two-stage
BF->GFP pipeline unchanged). The only differences vs. the classifier:
  * target = the raw force, standardized with TRAIN-only mean/std (no bins);
  * head  = a single linear unit (n_out=1); loss = Huber (or MSE / L1);
  * metrics = MAE, RMSE, R^2, Spearman (primary), Pearson, and MAE vs a
    mean-force baseline; plot = true-vs-pred scatter + residuals.

Early stopping uses the VAL split by default (--val_frac 0.2) — on this small,
noisy data, selecting the best-on-held-out epoch matters (train-loss "early
stopping" just picks the most-overfit epoch).

Usage
-----
    python train_split_force_regressor.py -c configs/gfp_classifier_3d.yaml \
        --data_dir data_phalloidin_mhc_051826_staged \
        --metadata "phalloidin_mhc_mapping_051426_SS edit.xlsx" \
        --target_col peak_amplitude_week3 --group_cols plate,Tissue \
        --init_from ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
        --output results/force_reg_new/reg_3d.json
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
from src.utils import set_seed, prepare_env, make_train_val_split
from src.data.regression_dataset import VolumeRegressionDataset
from src.models.gfp_classifier import build_gfp_classifier
from src.data.force_metadata import build_force_groups
from train_loo_force_classifier import (
    pearson, spearman, _seed_worker, _eval_det, load_encoder_from_unet,
    build_transforms)
from train_split_force_classifier import split_replicates


def _emit_reg_manifests(base, args, seed, groups, rep_force, train_g, val_g,
                        test_g, forces_per_stem):
    """Task-agnostic split manifest (+ BF->GFP non-test split) — same schema the
    classifier emits, minus the bin fields, so the two pipelines interoperate."""
    manifest = {
        "schema": "force_split_v1", "seed": int(seed), "task": "regression",
        "target_col": args.target_col, "group_cols": list(
            c.strip() for c in args.group_cols.split(",") if c.strip()),
        "input": args.input,
        "groups": {g: {"stems": list(groups[g]), "force": float(rep_force[g])}
                   for g in groups},
        "train_groups": list(train_g), "val_groups": list(val_g),
        "test_groups": list(test_g),
        "stem_force": {s: float(forces_per_stem[s]) for s in forces_per_stem},
    }
    with open(base + ".split.json", "w") as f:
        json.dump(manifest, f, indent=2)
    nontest = [s for g in (list(train_g) + list(val_g)) for s in groups[g]]
    bf_tr, bf_va = make_train_val_split(nontest, val_fraction=0.15, seed=seed)
    test_stems = [s for g in test_g for s in groups[g]]
    with open(base + ".bfgfp_split.json", "w") as f:
        json.dump({"train": sorted(bf_tr), "val": sorted(bf_va),
                   "excluded_test_stems": sorted(test_stems)}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default=None)
    p.add_argument("--target_col", default="peak_amplitude_week3")
    p.add_argument("--file_col", default="file")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--init_from", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--loss", choices=["huber", "mse", "l1"], default="huber")
    p.add_argument("--test_frac", type=float, default=0.25)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Held-out val for early stopping (default 0.2; 0 -> stop "
                        "on train loss, which overfits on small data).")
    p.add_argument("--n_permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_ckpt", default=None)
    p.add_argument("--plan_only", action="store_true")
    p.add_argument("--allow_partial_match", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--split_json", default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    args = p.parse_args()

    cfg = validate_config(load_config(args.config))
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    data_dir = args.data_dir or dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)
    dims = cfg["model"].get("dims", "2d")
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    # ---- resolve split (consume manifest, or build + emit) ----
    if args.split_json:
        with open(args.split_json) as f:
            sp = json.load(f)
        groups = {g: list(v["stems"]) for g, v in sp["groups"].items()}
        rep_force = {g: float(v["force"]) for g, v in sp["groups"].items()}
        train_g, val_g, test_g = (sp["train_groups"], sp["val_groups"],
                                  sp["test_groups"])
        forces_per_stem = {s: float(v) for s, v in sp["stem_force"].items()}
        group_keys = sorted(groups)
        print(f"consuming split manifest {args.split_json} "
              f"(seed={sp.get('seed')}); modality '{args.input}'")
    else:
        if not args.metadata:
            raise SystemExit("--metadata is required unless --split_json is given.")
        data = build_force_groups(
            args.metadata, data_dir, args.target_col, file_col=args.file_col,
            group_cols=tuple(group_cols), modality=args.input)
        print("\n".join(data["report"]))
        if data["n_matched"] == 0:
            raise SystemExit("No metadata force rows matched any staged volume.")
        if data["unmatched_meta"] and not args.allow_partial_match:
            raise SystemExit(
                f"{len(data['unmatched_meta'])} force row(s) matched no staged "
                "volume; pass --allow_partial_match to use the matched subset.")
        forces_per_stem = data["forces"]
        groups = data["groups"]
        rep_force = data["rep_force"]
        group_keys = sorted(groups)
        if len(group_keys) < 3:
            raise SystemExit(f"Only {len(group_keys)} replicates; need >=3.")
        # stratify the split by force terciles for a spread (labels are continuous)
        train_g, val_g, test_g, _se = split_replicates(
            group_keys, rep_force, min(3, len(group_keys)),
            args.test_frac, args.val_frac, seed)
        min_train = 3
        if val_g and len(train_g) < min_train:
            train_g = sorted(train_g + val_g); val_g = []
        if not test_g:
            raise SystemExit("Empty TEST set — too few replicates.")
        _emit_reg_manifests(os.path.splitext(args.output)[0], args, seed, groups,
                            rep_force, train_g, val_g, test_g, forces_per_stem)

    # ---- TRAIN-only standardization (leak-free) ----
    train_forces = np.array([rep_force[g] for g in train_g], dtype=np.float64)
    mu = float(train_forces.mean())
    sd = float(train_forces.std())
    if sd < 1e-8:
        sd = 1.0
    targets = {s: (forces_per_stem[s] - mu) / sd for s in forces_per_stem}

    def nvol(gs):
        return sum(len(groups[g]) for g in gs)
    print(f"\ntarget={args.target_col} input={args.input} dims={dims} REGRESSION "
          f"loss={args.loss}")
    print(f"  train std: mu={mu:.3f} sd={sd:.3f}  | force range "
          f"[{min(rep_force.values()):.2f}, {max(rep_force.values()):.2f}]")
    print(f"  train: {len(train_g)} reps / {nvol(train_g)} vols | "
          f"val: {len(val_g)} reps / {nvol(val_g)} vols | "
          f"test: {len(test_g)} reps / {nvol(test_g)} vols")

    plan = {
        "task": "force_regression", "target_col": args.target_col,
        "input": args.input, "dims": dims, "seed": int(seed),
        "loss": args.loss, "standardize": {"mu": mu, "sd": sd},
        "group_cols": group_cols, "n_replicates": len(group_keys),
        "init_from": args.init_from, "warm_started": bool(args.init_from),
        "freeze_encoder": bool(args.freeze_encoder),
        "consumed_split_json": args.split_json,
        "split": {name: [{"group": g, "force": rep_force[g],
                          "n_vols": len(groups[g])} for g in gs]
                  for name, gs in [("train", train_g), ("val", val_g),
                                   ("test", test_g)]},
    }
    if args.plan_only:
        out = os.path.splitext(args.output)[0] + ".plan.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(plan, open(out, "w"), indent=2)
        print(f"\n[plan_only] wrote {out}")
        return

    # ---------------------------- training ----------------------------
    accelerator, device, _ = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))
    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))
    eval_seed = seed + 9973

    train_stems = [s for g in train_g for s in groups[g]]
    val_stems = [s for g in val_g for s in groups[g]]
    test_stems = [s for g in test_g for s in groups[g]]

    def make_ds(stem_list, train):
        return VolumeRegressionDataset(
            [os.path.join(mod_dir, f"{s}.npy") for s in stem_list],
            stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, train), z_range=z_range,
            apply_timm=apply_timm, percentile_clip=percentile_clip, mode=dims,
            patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=(dcfg.get("patches_per_volume", 32) if train else 8),
            crop_size=dcfg.get("crop_size", 256), modality=args.input)

    epochs = tcfg.get("epochs", 100)
    patience = tcfg.get("patience", 15)
    min_delta = tcfg.get("min_delta", 1e-4)
    lr = tcfg["lr"]

    model = build_gfp_classifier(cfg, 1, 2)   # n_out=1 -> single regression unit
    if args.init_from:
        n_match, n_keys = load_encoder_from_unet(model, args.init_from, "cpu")
        tag = os.path.basename(os.path.dirname(args.init_from))
        accelerator.print(f"warm-started encoder: {n_match}/{n_keys} from {tag}")
        if n_match == 0:
            raise SystemExit(f"--init_from {args.init_from} matched 0 tensors.")
    if args.freeze_encoder:
        for pm in model.encoder.parameters():
            pm.requires_grad = False
        trainable = [pm for pm in model.parameters() if pm.requires_grad]
        accelerator.print(f"freeze_encoder: linear probe — "
                          f"{sum(p.numel() for p in trainable):,} head params")
    else:
        trainable = model.parameters()

    wd = args.weight_decay if args.weight_decay is not None else tcfg.get(
        "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    criterion = {"huber": nn.SmoothL1Loss(), "mse": nn.MSELoss(),
                 "l1": nn.L1Loss()}[args.loss]
    accelerator.print(f"optimizer: AdamW lr={lr} weight_decay(L2)={wd} loss={args.loss}")

    gen = torch.Generator(); gen.manual_seed(seed * 1000 + 7)
    train_loader = torch.utils.data.DataLoader(
        make_ds(train_stems, True), batch_size=tcfg["batch_size"], shuffle=True,
        drop_last=True, pin_memory=True, num_workers=tcfg.get("num_workers", 4),
        worker_init_fn=_seed_worker, generator=gen)
    test_loader = torch.utils.data.DataLoader(
        make_ds(test_stems, False), batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=0)
    val_loader = (torch.utils.data.DataLoader(
        make_ds(val_stems, False), batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=0) if val_stems else None)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader)

    def eval_loss_on(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for img, tgt, _ in loader:
                img = img.to(device)
                pred = model(img)[0].squeeze(1)
                t = tgt.float().to(pred.device)
                tot += criterion(pred, t).item() * img.shape[0]
                n += img.shape[0]
        return tot / max(n, 1)

    def predict_per_vol(loader, n_stems):
        """Mean standardized prediction per volume (file_idx local to loader)."""
        model.eval()
        sums = np.zeros(n_stems); counts = np.zeros(n_stems, dtype=int)
        with torch.no_grad():
            for img, _t, fidx in loader:
                pred = model(img.to(device))[0].squeeze(1).detach().cpu().numpy()
                f = fidx.detach().cpu().numpy().reshape(-1)
                for v, i in zip(np.atleast_1d(pred), f):
                    sums[int(i)] += float(v); counts[int(i)] += 1
        preds = np.full(n_stems, np.nan)
        valid = counts > 0
        preds[valid] = sums[valid] / counts[valid]
        return preds, counts

    best_sig = float("inf")
    best_preds = best_counts = None
    best_epoch = 0
    no_improve = 0
    for ep in range(epochs):
        model.train()
        if args.freeze_encoder:
            accelerator.unwrap_model(model).encoder.eval()
        losses = []
        for img, tgt, _ in train_loader:
            pred = model(img)[0].squeeze(1)
            loss = criterion(pred, tgt.float().to(pred.device))
            optimizer.zero_grad(); accelerator.backward(loss); optimizer.step()
            losses.append(loss.item())
        tr = float(np.mean(losses)) if losses else float("inf")
        if val_loader is not None:
            sig = _eval_det(lambda: eval_loss_on(val_loader), eval_seed)
            sig_name = "val"
        else:
            sig, sig_name = tr, "train"
        preds, counts = _eval_det(
            lambda: predict_per_vol(test_loader, len(test_stems)), eval_seed)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            accelerator.print(f"  ep{ep+1}/{epochs} train={tr:.4f} {sig_name}={sig:.4f}")
        if sig < best_sig - min_delta:
            best_sig, best_preds, best_counts, best_epoch = sig, preds, counts, ep + 1
            no_improve = 0
            if args.save_ckpt and accelerator.is_main_process:
                os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                tmp = args.save_ckpt + ".tmp"
                torch.save({"epoch": best_epoch, "val_loss": float(sig),
                            "model_state_dict": unwrapped.state_dict(),
                            "task": "regression", "head": "exercise", "n_out": 1,
                            "target_col": args.target_col, "input": args.input,
                            "standardize": {"mu": mu, "sd": sd}}, tmp)
                os.replace(tmp, args.save_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                accelerator.print(f"  early stop ep{ep+1} ({sig_name}={best_sig:.4f} "
                                  f"@ ep{best_epoch})")
                break
    if best_preds is None:
        best_preds, best_counts = preds, counts

    # ---- per-replicate predictions (mean of its test volumes; un-standardize) ----
    stem_pred = {s: best_preds[i] for i, s in enumerate(test_stems)
                 if best_counts[i] > 0}
    results = []
    for g in test_g:
        pv = [stem_pred[s] for s in groups[g] if s in stem_pred]
        if not pv:
            accelerator.print(f"  warn: test replicate {g} had 0 valid vols")
            continue
        pred_force = float(np.mean(pv)) * sd + mu
        results.append({"group": g, "true_force": float(rep_force[g]),
                        "pred_force": pred_force, "n_vols": len(pv),
                        "stems": [s for s in groups[g] if s in stem_pred]})
    if not results:
        raise SystemExit("No test replicate produced a prediction.")

    true = np.array([r["true_force"] for r in results], dtype=np.float64)
    pred = np.array([r["pred_force"] for r in results], dtype=np.float64)
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    sp = spearman(true, pred)
    pe = pearson(true, pred)
    baseline_mae = float(np.mean(np.abs(true - mu)))  # predict the train mean

    perm = None
    if args.n_permutations and len(results) > 2 and not np.isnan(sp):
        rng = np.random.default_rng(0)
        obs = sp
        cnt = 0
        for _ in range(args.n_permutations):
            cnt += spearman(rng.permutation(true), pred) >= obs
        perm = {"n_permutations": int(args.n_permutations),
                "p_value_spearman": float((cnt + 1) / (args.n_permutations + 1))}

    accelerator.print(
        f"\nTEST (n={len(results)}): MAE={mae:.3f} (baseline={baseline_mae:.3f}) "
        f"RMSE={rmse:.3f} R2={r2:.3f} spearman={sp:.3f} pearson={pe:.3f}"
        + (f" perm_p={perm['p_value_spearman']:.4f}" if perm else ""))

    summary = dict(plan)
    summary.update({
        "n_test_replicates": len(results),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2,
                    "spearman": sp, "pearson": pe,
                    "baseline_mae_predict_mean": baseline_mae,
                    "beats_baseline": bool(mae < baseline_mae)},
        "permutation_test": perm, "best_epoch": best_epoch,
        "config_flags": {"test_frac": float(args.test_frac),
                         "val_frac": float(args.val_frac),
                         "has_val": bool(val_stems), "weight_decay_l2": float(wd)},
        "per_test_replicate": results,
    })
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(summary, open(args.output, "w"), indent=2)
        accelerator.print(f"Saved {args.output}")
        make_plot(summary, os.path.splitext(args.output)[0] + ".png")
        accelerator.print(f"Saved {os.path.splitext(args.output)[0]}.png")


def make_plot(summary, path):
    res = summary["per_test_replicate"]
    m = summary["metrics"]
    true = np.array([r["true_force"] for r in res], dtype=float)
    pred = np.array([r["pred_force"] for r in res], dtype=float)
    dims = summary["dims"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))

    ax[0].scatter(true, pred, s=60, alpha=0.85, color="#4363d8",
                  edgecolor="k", linewidth=0.4)
    allv = np.concatenate([true, pred]) if len(true) else np.array([0, 1])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", ls=":")
    ax[0].set_xlim(lo - pad, hi + pad); ax[0].set_ylim(lo - pad, hi + pad)
    ax[0].set_xlabel(f"True {summary['target_col']}")
    ax[0].set_ylabel("Predicted force")
    ax[0].set_title(f"True vs predicted (test)\nMAE={m['mae']:.2f} "
                    f"(base {m['baseline_mae_predict_mean']:.2f})  R²={m['r2']:.2f}  "
                    f"spearman={m['spearman']:.2f}")
    ax[0].grid(True, alpha=0.3)

    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].scatter(true, pred - true, s=60, alpha=0.85, color="#e6194b",
                  edgecolor="k", linewidth=0.4)
    ax[1].set_xlabel(f"True {summary['target_col']}")
    ax[1].set_ylabel("Residual (pred − true)")
    ax[1].set_title("Residuals"); ax[1].grid(True, alpha=0.3)

    enc = "warm-started BF->GFP encoder" if summary.get("warm_started") \
        else "ImageNet encoder"
    probe = " | frozen linear probe" if summary.get("freeze_encoder") else ""
    fig.suptitle(f"Force REGRESSION ({dims.upper()}) | {summary['target_col']} | "
                 f"{summary['n_test_replicates']} test reps | {enc}{probe}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
