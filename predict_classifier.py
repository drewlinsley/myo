"""Run a trained two-head classifier on a folder of BF .npy volumes; dump
per-volume predictions to JSON.

Requires a checkpoint saved by train_gfp_classifier.py — best.pth plus a
config.yaml in the same directory. (train_loo_classifier.py does NOT save
weights; it dumps predictions inline. Use train_gfp_classifier.py once to
materialize a usable inference checkpoint.)

Reads {data_dir}/bf/<stem>.npy and {data_dir}/stats/<stem>.json — run
compute_stats.py first.

Usage:
    python predict_classifier.py \\
        --ckpt ckpts/gfp_classifier_3d_frac100/best.pth \\
        --data_dir data_phalloidin_mhc_051826_staged \\
        --task perturbation \\
        --output results/classify_new_dataset/perturbation_3d.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config, resolve_ckpt_config
from src.utils import prepare_env, load_checkpoint
from src.models.gfp_classifier import build_gfp_classifier
from src.data.normalization import normalize


def grid_starts(total, step, overlap):
    """Inclusive list of start indices covering [0, total) in `step`-wide tiles."""
    stride = max(1, step - overlap)
    s = list(range(0, max(1, total - step + 1), stride))
    if not s:
        s = [0]
    elif s[-1] + step < total:
        s.append(total - step)
    return s


def pad_to_min(vol, min_z, min_h, min_w):
    Z, H, W = vol.shape
    pz = max(0, min_z - Z)
    ph = max(0, min_h - H)
    pw = max(0, min_w - W)
    if pz or ph or pw:
        vol = np.pad(vol, ((0, pz), (0, ph), (0, pw)), mode="reflect")
    return vol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Classifier best.pth path")
    p.add_argument("--data_dir", required=True, help="Root with bf/ + stats/")
    p.add_argument("--task", choices=["exercise", "perturbation"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None,
                   help="Override config (default: <ckpt_dir>/config.yaml)")
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_classes", type=int, default=2,
                   help="Number of output classes the ckpt was trained with.")
    p.add_argument("--patch_overlap", type=int, nargs=3, default=None,
                   help="(z, y, x) overlap for 3D tiling (default: half-patch)")
    args = p.parse_args()

    ckpt_dir = os.path.dirname(args.ckpt)
    cfg_path = resolve_ckpt_config(ckpt_dir, args.config)
    cfg = load_config(cfg_path)
    dims = cfg["model"].get("dims", "2d")
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    n_ex = args.n_classes if args.task == "exercise" else 2
    n_pt = args.n_classes if args.task == "perturbation" else 2

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_gfp_classifier(cfg_copy, n_ex, n_pt)
    ckpt = load_checkpoint(args.ckpt, model)
    accelerator.print(
        f"Loaded {args.ckpt} dims={dims} task={args.task} "
        f"epoch={ckpt.get('epoch', '?')} val_loss={ckpt.get('val_loss', '?')}")
    model = accelerator.prepare(model)
    model.eval()

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
    accelerator.print(f"Will classify {len(stems)} volume(s)")

    z_range = cfg["data"].get("z_range", None)
    patch_depth = cfg["data"].get("patch_depth", 32)
    crop = cfg["data"].get("crop_size", 256)
    if args.patch_overlap is None:
        ovl = (patch_depth // 2, crop // 2, crop // 2)
    else:
        ovl = tuple(args.patch_overlap)

    rows = []
    with torch.no_grad():
        for stem in tqdm(stems, desc="Classify"):
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                stats = json.load(f)
            bf_raw = np.load(os.path.join(bf_dir, f"{stem}.npy"))
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            probs_sum = None
            n_seen = 0

            def feed(batch):
                """batch: list of (D, H, W) for 3d, or (H, W) for 2d."""
                nonlocal probs_sum, n_seen
                if not batch:
                    return
                arr = np.stack(batch).astype(np.float32)
                if dims == "3d":
                    # (B, D, H, W) -> (B, 1, H, W, D)
                    arr = arr.transpose(0, 2, 3, 1)[:, None]
                else:
                    # (B, H, W) -> (B, 1, H, W)
                    arr = arr[:, None]
                x = torch.from_numpy(arr).to(device)
                lex, lpt = model(x)
                logits = lex if args.task == "exercise" else lpt
                sm = F.softmax(logits, dim=-1).sum(dim=0).cpu().numpy()
                probs_sum = sm if probs_sum is None else probs_sum + sm
                n_seen += x.shape[0]

            batch = []
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

            probs = (probs_sum / max(n_seen, 1)).tolist()
            pred_idx = int(np.argmax(probs))
            rows.append({
                "stem": stem,
                "task": args.task,
                "probs": [float(p) for p in probs],
                "pred_idx": pred_idx,
                "n_views": n_seen,
            })
            accelerator.print(
                f"  {stem}: pred={pred_idx} "
                f"probs={[round(float(p), 3) for p in probs]} "
                f"(n_views={n_seen})")

    summary = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "task": args.task,
        "dims": dims,
        "n_volumes": len(rows),
        "per_volume": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
