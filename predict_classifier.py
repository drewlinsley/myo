"""Run a trained two-head classifier on a folder of BF .npy volumes; dump
per-volume predictions to JSON. Supports ensembling multiple fold ckpts.

Two ways to get inference-ready ckpts:
  1. `train_loo_classifier.py --save_ckpt_dir <dir>` saves per-fold weights at
     `<dir>/<group>/best.pth`. Use `--ckpts '<dir>/*/best.pth'` to ensemble.
  2. `train_gfp_classifier.py` saves a single `best.pth`. Use `--ckpt <path>`.

Reads {data_dir}/bf/<stem>.npy and {data_dir}/stats/<stem>.json — run
compute_stats.py first.

Usage:
    # single ckpt
    python predict_classifier.py \\
        --ckpt ckpts/gfp_classifier_3d_frac100/best.pth \\
        --data_dir data_phalloidin_mhc_051826_staged \\
        --task perturbation \\
        --output results/classify_new_dataset/perturbation_3d.json

    # LOO ensemble (recommended)
    python predict_classifier.py \\
        --ckpts ckpts/loo_classifier_3d_pt/*/best.pth \\
        --data_dir data_phalloidin_mhc_051826_staged \\
        --task perturbation \\
        --output results/classify_new_dataset/perturbation_loo_ens.json
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
from src.data.zband import resolve_z_range


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


def resolve_ckpts(ckpt, ckpts):
    if ckpts:
        # nargs+ from CLI may already be expanded by shell; also allow glob patterns.
        resolved = []
        for pat in ckpts:
            hits = sorted(glob.glob(pat))
            resolved.extend(hits if hits else [pat])
        # Filter to actually-existing files (a bare path that didn't glob will be checked here)
        resolved = [p for p in resolved if os.path.isfile(p)]
        if not resolved:
            raise SystemExit(f"--ckpts matched no files: {ckpts}")
        return resolved
    if ckpt:
        return [ckpt]
    raise SystemExit("Need --ckpt <path> or --ckpts <paths|glob>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="Single classifier .pth")
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="Multiple classifier .pths (or glob pattern) — ensembled "
                        "by averaging softmax across folds.")
    p.add_argument("--data_dir", required=True, help="Root with bf/ + stats/")
    p.add_argument("--task", choices=["exercise", "perturbation"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None,
                   help="Override config (default: <first ckpt dir>/config.yaml)")
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_classes", type=int, default=2,
                   help="Number of output classes the ckpts were trained with.")
    p.add_argument("--patch_overlap", type=int, nargs=3, default=None,
                   help="(z, y, x) overlap for 3D tiling (default: half-patch)")
    args = p.parse_args()

    ckpt_paths = resolve_ckpts(args.ckpt, args.ckpts)
    cfg_path = resolve_ckpt_config(os.path.dirname(ckpt_paths[0]), args.config)
    cfg = load_config(cfg_path)
    dims = cfg["model"].get("dims", "2d")
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    # Peek at first ckpt for raw_classes / n_classes (LOO ckpts store it; full-data
    # ckpts may not — fall back to --n_classes in that case).
    peek = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    raw_classes = peek.get("raw_classes")
    n_cls = len(raw_classes) if raw_classes else args.n_classes
    n_ex = n_cls if args.task == "exercise" else 2
    n_pt = n_cls if args.task == "perturbation" else 2

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_gfp_classifier(cfg_copy, n_ex, n_pt)
    model = accelerator.prepare(model)
    model.eval()
    accelerator.print(
        f"dims={dims} task={args.task} n_classes={n_cls} "
        f"ensembling {len(ckpt_paths)} ckpt(s)")
    if raw_classes:
        accelerator.print(f"raw_classes={raw_classes}")

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

    unwrap = accelerator.unwrap_model(model)

    def load_weights(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st.get("model_state_dict", st.get("state_dict", st))
        missing, unexpected = unwrap.load_state_dict(sd, strict=False)
        if missing:
            accelerator.print(f"  warn: {len(missing)} missing keys "
                              f"loading {os.path.basename(ckpt_path)}")
        return st

    # Per-stem accumulators across all ckpts × all patches/slices
    probs_sum = {s: np.zeros(n_cls, dtype=np.float64) for s in stems}
    n_seen = {s: 0 for s in stems}
    per_ckpt_pred = {s: [] for s in stems}  # one entry per ckpt for diagnostics

    with torch.no_grad():
        for ck_path in ckpt_paths:
            st = load_weights(ck_path)
            tag = os.path.basename(os.path.dirname(ck_path))
            accelerator.print(f"\n== ckpt {tag} (ep {st.get('epoch', '?')}) ==")
            for stem in tqdm(stems, desc=f"ckpt {tag}"):
                with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                    stats = json.load(f)
                bf_raw = np.load(os.path.join(bf_dir, f"{stem}.npy"))
                if z_range is not None:
                    z_lo, z_hi = resolve_z_range(z_range, stats,
                                                 bf_raw.shape[0])
                    bf_raw = bf_raw[z_lo:z_hi]
                bf = normalize(bf_raw, stats["bf"]["p_low"],
                               stats["bf"]["p_high"], apply_timm=apply_timm)

                local_sum = np.zeros(n_cls, dtype=np.float64)
                local_n = 0
                batch = []

                def feed(batch):
                    nonlocal local_sum, local_n
                    if not batch:
                        return
                    arr = np.stack(batch).astype(np.float32)
                    if dims == "3d":
                        arr = arr.transpose(0, 2, 3, 1)[:, None]
                    else:
                        arr = arr[:, None]
                    x = torch.from_numpy(arr).to(device)
                    lex, lpt = model(x)
                    logits = lex if args.task == "exercise" else lpt
                    sm = F.softmax(logits, dim=-1).sum(dim=0).cpu().numpy()
                    local_sum += sm
                    local_n += x.shape[0]

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

                probs_sum[stem] += local_sum
                n_seen[stem] += local_n
                # Per-ckpt prediction for diagnostics
                if local_n:
                    per_ckpt_pred[stem].append({
                        "ckpt": tag,
                        "pred_idx": int(np.argmax(local_sum / local_n)),
                    })

    rows = []
    for stem in stems:
        probs = (probs_sum[stem] / max(n_seen[stem], 1))
        pred_idx = int(np.argmax(probs))
        row = {
            "stem": stem,
            "task": args.task,
            "probs": [float(p) for p in probs],
            "pred_idx": pred_idx,
            "n_views": int(n_seen[stem]),
            "n_ckpts": len(ckpt_paths),
        }
        if raw_classes:
            row["pred_class"] = raw_classes[pred_idx]
            row["class_probs"] = {raw_classes[i]: float(probs[i])
                                  for i in range(n_cls)}
        if len(ckpt_paths) > 1:
            row["per_ckpt_pred_idx"] = [d["pred_idx"]
                                        for d in per_ckpt_pred[stem]]
        rows.append(row)
        accelerator.print(
            f"  {stem}: pred={pred_idx} "
            f"probs={[round(float(p), 3) for p in probs]} "
            f"(n_views={n_seen[stem]})")

    summary = {
        "ckpts": ckpt_paths,
        "n_ckpts": len(ckpt_paths),
        "data_dir": args.data_dir,
        "task": args.task,
        "dims": dims,
        "n_classes": n_cls,
        "raw_classes": raw_classes,
        "n_volumes": len(rows),
        "per_volume": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
