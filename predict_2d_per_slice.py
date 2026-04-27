"""Per-slice 2D BF→GFP predictions, saved as individual .npy files.

For each volume in `data/bf/*.npy`, runs the 2D model on every Z-slice and
writes one .npy per slice:
    {output_dir}/{stem}/z{idx:04d}.npy   shape (H, W), float32, [0, 1]

Usage:
    python predict_2d_per_slice.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson/best.pth \
        --output_dir predictions/per_slice
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch

from src.config import load_config
from src.utils import prepare_env, load_checkpoint
from src.models import build_model
from src.data.normalization import normalize, denormalize
from predict import predict_2d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="predictions/per_slice")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Optional subset of volume stems to process")
    p.add_argument("--denormalize", action="store_true",
                   help="Save in raw GFP intensity scale (default [0,1])")
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg["model"]["dims"] != "2d":
        raise SystemExit(f"This script is 2D-only; config is {cfg['model']['dims']}")
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)

    ckpt = load_checkpoint(args.checkpoint, model)
    accelerator.print(f"Loaded {args.checkpoint} (epoch={ckpt.get('epoch', '?')}, "
                      f"val_loss={ckpt.get('val_loss', '?')})")

    model = accelerator.prepare(model)
    model.eval()

    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    stats_dir = os.path.join(data_dir, "stats")
    z_range = cfg["data"].get("z_range", None)

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    if args.stems:
        keep = set(args.stems)
        bf_files = [f for f in bf_files
                    if os.path.splitext(os.path.basename(f))[0] in keep]
    accelerator.print(f"Processing {len(bf_files)} volume(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for bf_path in tqdm(bf_files, desc="Volumes"):
            stem = os.path.splitext(os.path.basename(bf_path))[0]
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                stats = json.load(f)

            bf_raw = np.load(bf_path)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            pred = predict_2d(model, bf, device)  # (Z, H, W) in [0, 1]
            if args.denormalize:
                pred = denormalize(pred, stats["gfp"]["p_low"],
                                   stats["gfp"]["p_high"])

            vol_dir = os.path.join(args.output_dir, stem)
            os.makedirs(vol_dir, exist_ok=True)
            for z in range(pred.shape[0]):
                np.save(os.path.join(vol_dir, f"z{z:04d}.npy"),
                        pred[z].astype(np.float32))
            accelerator.print(
                f"  {stem}: wrote {pred.shape[0]} slices "
                f"-> {vol_dir}/z0000.npy ... z{pred.shape[0]-1:04d}.npy")

    accelerator.print("Done.")


if __name__ == "__main__":
    main()
