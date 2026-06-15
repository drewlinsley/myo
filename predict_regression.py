"""Run trained regression fold-ckpts on a folder of BF .npy volumes; dump
per-volume force-amplitude predictions to JSON.

Each fold ckpt stores its own (t_mean, t_std) from when it was trained — we
de-normalize that fold's patch-averaged prediction with its own stats, then
average across folds. Inter-fold std is reported as a rough uncertainty.

Requires fold ckpts produced by:
    python train_loo_regression.py ... --save_ckpt_dir <dir>

Usage:
    python predict_regression.py \\
        --ckpts 'ckpts/loo_regression_3d/*/best.pth' \\
        --data_dir data_phalloidin_mhc_051826_staged \\
        --output results/regress_new_dataset/force_3d_loo_ens.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from src.config import load_config, resolve_ckpt_config
from src.utils import prepare_env
from src.models.gfp_regressor import build_gfp_regressor
from src.data.normalization import normalize


def grid_starts(total, step, overlap):
    stride = max(1, step - overlap)
    s = list(range(0, max(1, total - step + 1), stride))
    if not s:
        s = [0]
    elif s[-1] + step < total:
        s.append(total - step)
    return s


def pad_to_min(vol, min_z, min_h, min_w):
    Z, H, W = vol.shape
    pz = max(0, min_z - Z); ph = max(0, min_h - H); pw = max(0, min_w - W)
    if pz or ph or pw:
        vol = np.pad(vol, ((0, pz), (0, ph), (0, pw)), mode="reflect")
    return vol


def resolve_ckpts(ckpt, ckpts):
    if ckpts:
        resolved = []
        for pat in ckpts:
            hits = sorted(glob.glob(pat))
            resolved.extend(hits if hits else [pat])
        resolved = [p for p in resolved if os.path.isfile(p)]
        if not resolved:
            raise SystemExit(f"--ckpts matched no files: {ckpts}")
        return resolved
    if ckpt:
        return [ckpt]
    raise SystemExit("Need --ckpt or --ckpts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="Single regression .pth")
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="Multiple regression fold ckpts (or glob); ensembled.")
    p.add_argument("--data_dir", required=True, help="Root with bf/ + stats/")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None,
                   help="Override config (default: <first ckpt dir>/config.yaml)")
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--patch_overlap", type=int, nargs=3, default=None,
                   help="(z, y, x) overlap for 3D tiling (default: half-patch)")
    args = p.parse_args()

    ckpt_paths = resolve_ckpts(args.ckpt, args.ckpts)
    cfg_path = resolve_ckpt_config(os.path.dirname(ckpt_paths[0]), args.config)
    cfg = load_config(cfg_path)
    dims = cfg["model"].get("dims", "2d")
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_gfp_regressor(cfg_copy)
    model = accelerator.prepare(model)
    model.eval()
    accelerator.print(
        f"dims={dims} ensembling {len(ckpt_paths)} regression fold ckpt(s)")

    bf_dir = os.path.join(args.data_dir, "bf")
    stats_dir = os.path.join(args.data_dir, "stats")
    if not os.path.isdir(bf_dir):
        raise SystemExit(f"Missing {bf_dir}/")
    if not os.path.isdir(stats_dir):
        raise SystemExit(f"Missing {stats_dir}/ — run compute_stats.py first")

    stems = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(bf_dir, "*.npy")))
    if args.stems:
        keep = set(args.stems)
        stems = [s for s in stems if s in keep]
    stems = [s for s in stems
             if os.path.exists(os.path.join(stats_dir, f"{s}.json"))]
    if not stems:
        raise SystemExit(f"No BF vols with stats under {args.data_dir}")
    accelerator.print(f"Will regress on {len(stems)} volume(s)")

    z_range = cfg["data"].get("z_range", None)
    patch_depth = cfg["data"].get("patch_depth", 32)
    crop = cfg["data"].get("crop_size", 256)
    if args.patch_overlap is None:
        ovl = (patch_depth // 2, crop // 2, crop // 2)
    else:
        ovl = tuple(args.patch_overlap)

    unwrap = accelerator.unwrap_model(model)

    def load_weights(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st.get("model_state_dict", st.get("state_dict", st))
        missing, unexpected = unwrap.load_state_dict(sd, strict=False)
        if missing:
            accelerator.print(f"  warn: {len(missing)} missing keys "
                              f"loading {os.path.basename(ckpt_path)}")
        return st

    # Per-stem accumulators: one de-normalized scalar per ckpt
    per_stem_preds = {s: [] for s in stems}  # list of (fold_tag, prediction_raw)
    target_col = None

    with torch.no_grad():
        for ck_path in ckpt_paths:
            st = load_weights(ck_path)
            t_mean = float(st.get("t_mean", 0.0))
            t_std = float(st.get("t_std", 1.0))
            target_col = target_col or st.get("target_col")
            tag = os.path.basename(os.path.dirname(ck_path))
            accelerator.print(
                f"\n== ckpt {tag} ep={st.get('epoch', '?')} "
                f"t_mean={t_mean:.3f} t_std={t_std:.3f} ==")
            for stem in tqdm(stems, desc=f"ckpt {tag}"):
                with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                    stats = json.load(f)
                bf_raw = np.load(os.path.join(bf_dir, f"{stem}.npy"))
                if z_range is not None:
                    z_lo = max(0, z_range[0])
                    z_hi = min(bf_raw.shape[0], z_range[1])
                    bf_raw = bf_raw[z_lo:z_hi]
                bf = normalize(bf_raw, stats["bf"]["p_low"],
                               stats["bf"]["p_high"], apply_timm=apply_timm)

                pred_sum = 0.0
                n_views = 0
                batch = []

                def feed(batch):
                    nonlocal pred_sum, n_views
                    if not batch:
                        return
                    arr = np.stack(batch).astype(np.float32)
                    if dims == "3d":
                        arr = arr.transpose(0, 2, 3, 1)[:, None]
                    else:
                        arr = arr[:, None]
                    x = torch.from_numpy(arr).to(device)
                    out = model(x).detach().cpu().numpy()  # (B,) in z-scored space
                    pred_sum += float(out.sum())
                    n_views += int(out.shape[0])

                if dims == "3d":
                    vol = pad_to_min(bf, patch_depth, crop, crop)
                    Z, H, W = vol.shape
                    for z in grid_starts(Z, patch_depth, ovl[0]):
                        for y in grid_starts(H, crop, ovl[1]):
                            for x in grid_starts(W, crop, ovl[2]):
                                batch.append(vol[z:z+patch_depth,
                                                 y:y+crop, x:x+crop])
                                if len(batch) >= args.batch_size:
                                    feed(batch); batch = []
                    feed(batch)
                else:
                    for z in range(bf.shape[0]):
                        batch.append(bf[z])
                        if len(batch) >= args.batch_size:
                            feed(batch); batch = []
                    feed(batch)

                if n_views:
                    z_pred = pred_sum / n_views
                    raw_pred = z_pred * t_std + t_mean
                else:
                    raw_pred = float("nan")
                per_stem_preds[stem].append((tag, float(raw_pred)))

    rows = []
    for stem in stems:
        preds = [p for _, p in per_stem_preds[stem] if not np.isnan(p)]
        if not preds:
            mean_pred = float("nan"); std_pred = float("nan")
        else:
            mean_pred = float(np.mean(preds))
            std_pred = float(np.std(preds)) if len(preds) > 1 else 0.0
        row = {
            "stem": stem,
            "pred_mean": mean_pred,
            "pred_std": std_pred,
            "n_ckpts": len(preds),
            "per_ckpt": [{"fold": tag, "pred": p}
                         for tag, p in per_stem_preds[stem]],
        }
        rows.append(row)
        accelerator.print(
            f"  {stem}: pred={mean_pred:.3f} ± {std_pred:.3f} "
            f"(n_ckpts={len(preds)})")

    summary = {
        "ckpts": ckpt_paths,
        "n_ckpts": len(ckpt_paths),
        "data_dir": args.data_dir,
        "dims": dims,
        "target_col": target_col,
        "n_volumes": len(rows),
        "per_volume": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
