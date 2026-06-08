"""Per-slice BF→GFP predictions (2D or 3D model), saved as individual .npy.

For each volume in `data/bf/*.npy`, runs the model and writes one .npy per
Z-slice:
    {output_dir}/{stem}/z{idx:04d}.npy   shape (H, W), float32

The script dispatches on `model.dims` in the config — 2D runs the per-slice
network, 3D runs sliding-window patch inference.

Usage:
    python predict_per_slice.py \
        -c configs/unet_3d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
        --output_dir predictions/frac100
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch

from src.config import load_config, resolve_ckpt_config
from src.utils import prepare_env, load_checkpoint
from src.models import build_model
from src.data.normalization import normalize, denormalize
from src.data.foreground_mask import compute_bf_foreground_mask
from predict import predict_2d, predict_3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=None,
                   help="Optional; defaults to <ckpt_dir>/config.yaml with "
                        "fallback to configs/<experiment>.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="predictions/per_slice")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Optional subset of volume stems to process")
    p.add_argument("--denormalize", action="store_true",
                   help="Save in raw GFP intensity scale (default [0,1])")
    p.add_argument("--mask_background", action="store_true",
                   help="Zero out predicted GFP where BF is background "
                        "(removes edge / OOD artifacts).")
    p.add_argument("--mask_method", default="minimum",
                   choices=["minimum", "otsu", "li", "triangle"],
                   help="Threshold method for --mask_background.")
    p.add_argument("--mask_dilate", type=int, default=3,
                   help="2D dilation iterations (per Z-slice) before masking.")
    p.add_argument("--mask_min_frac", type=float, default=0.01,
                   help="Drop connected components smaller than this fraction "
                        "of foreground per Z-slice. 0 disables cleanup.")
    p.add_argument("--data_dir", default=None,
                   help="Override dataset root (e.g. new dataset path).")
    args = p.parse_args()

    config_path = resolve_ckpt_config(os.path.dirname(args.checkpoint),
                                      args.config)
    cfg = load_config(config_path)
    dims = cfg["model"]["dims"]
    if dims not in ("2d", "3d"):
        raise SystemExit(f"Unknown model.dims={dims}")
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)

    ckpt = load_checkpoint(args.checkpoint, model)
    accelerator.print(f"Loaded {args.checkpoint} dims={dims} "
                      f"(epoch={ckpt.get('epoch', '?')}, "
                      f"val_loss={ckpt.get('val_loss', '?')})")

    model = accelerator.prepare(model)
    model.eval()

    data_dir = args.data_dir or cfg["data"]["data_dir"]
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

            if dims == "2d":
                pred = predict_2d(model, bf, device)
            else:
                patch_depth = cfg["data"].get("patch_depth", 32)
                overlap = cfg.get("inference", {}).get("overlap", [16, 128, 128])
                inf_batch = cfg.get("inference", {}).get("batch_size", 4)
                spatial_tile = cfg.get("inference", {}).get(
                    "spatial_tile", cfg["data"].get("crop_size", 256))
                pred = predict_3d(model, bf, device, patch_depth, overlap,
                                  inf_batch, spatial_tile=spatial_tile)

            if args.mask_background:
                fg = compute_bf_foreground_mask(
                    bf_raw, method=args.mask_method,
                    dilate=args.mask_dilate,
                    min_component_frac=args.mask_min_frac)
                pred = pred * fg.astype(pred.dtype)

            if args.denormalize:
                pred = denormalize(pred, stats["gfp"]["p_low"],
                                   stats["gfp"]["p_high"])

            vol_dir = os.path.join(args.output_dir, stem)
            os.makedirs(vol_dir, exist_ok=True)
            for z in range(pred.shape[0]):
                np.save(os.path.join(vol_dir, f"z{z:04d}.npy"),
                        pred[z].astype(np.float32))
            accelerator.print(
                f"  {stem}: wrote {pred.shape[0]} slices -> {vol_dir}/")

    accelerator.print("Done.")


if __name__ == "__main__":
    main()
