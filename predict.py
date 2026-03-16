"""Unified inference for 2D (per-slice) and 3D (sliding window) prediction.

Usage:
    python predict.py -c configs/unet_2d_imagenet.yaml --checkpoint ckpts/unet_2d_imagenet/best.pth --output_dir predictions/unet_2d_imagenet
"""

import os
import json
import argparse
import numpy as np
import torch

from glob import glob
from src.config import load_config
from src.utils import prepare_env, load_checkpoint
from src.models import build_model
from src.data.normalization import normalize, denormalize


def predict_2d(model, bf_vol, device):
    """Predict GFP from BF volume one Z-slice at a time.

    Args:
        model: 2D U-Net model
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device

    Returns:
        (Z, H, W) predicted GFP volume
    """
    Z, H, W = bf_vol.shape
    predictions = np.zeros((Z, H, W), dtype=np.float32)

    # Pad spatial dims to be divisible by 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32

    for z in range(Z):
        slc = bf_vol[z]  # (H, W)
        if pad_h > 0 or pad_w > 0:
            slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")

        inp = torch.from_numpy(slc[np.newaxis, np.newaxis]).float().to(device)  # (1, 1, H', W')
        pred = model(inp)
        pred = pred[0, 0].cpu().numpy()[:H, :W]
        predictions[z] = pred

    return predictions


def predict_3d(model, bf_vol, device, patch_depth=32, overlap=(16, 128, 128),
               batch_size=4, spatial_tile=256):
    """Predict GFP from BF volume using 3D sliding window with Gaussian blending.

    Tiles in Z, H, and W to stay within GPU memory.

    Args:
        model: 3D U-Net model
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device
        patch_depth: depth of each 3D patch
        overlap: (z, h, w) overlap between patches
        batch_size: number of patches to process at once
        spatial_tile: spatial patch size (H and W)

    Returns:
        (Z, H, W) predicted GFP volume
    """
    Z, H, W = bf_vol.shape

    # Pad to be divisible by 32
    pad_z = (32 - Z % 32) % 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32

    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        bf_padded = np.pad(bf_vol, ((0, pad_z), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        bf_padded = bf_vol

    Zp, Hp, Wp = bf_padded.shape

    # Build Gaussian weight map for a single tile
    def gaussian_weight_3d(depth, height, width):
        gz = np.exp(-0.5 * ((np.linspace(-1, 1, depth)) ** 2))
        gh = np.exp(-0.5 * ((np.linspace(-1, 1, height)) ** 2))
        gw = np.exp(-0.5 * ((np.linspace(-1, 1, width)) ** 2))
        weight = gz[:, None, None] * gh[None, :, None] * gw[None, None, :]
        return weight.astype(np.float32)

    # Generate sliding window coordinates for all 3 axes
    def make_coords(total, tile, ovlp):
        stride = max(1, tile - ovlp)
        starts = list(range(0, max(1, total - tile + 1), stride))
        if starts and starts[-1] + tile < total:
            starts.append(total - tile)
        return sorted(set(starts))

    z_coords = make_coords(Zp, patch_depth, overlap[0])
    h_coords = make_coords(Hp, spatial_tile, overlap[1])
    w_coords = make_coords(Wp, spatial_tile, overlap[2])

    # Build all (z, h, w) tile coordinates
    all_coords = [(zs, hs, ws)
                  for zs in z_coords for hs in h_coords for ws in w_coords]

    predictions = np.zeros((Zp, Hp, Wp), dtype=np.float32)
    weights = np.zeros((Zp, Hp, Wp), dtype=np.float32)

    # Process in batches
    for batch_start in range(0, len(all_coords), batch_size):
        batch = all_coords[batch_start:batch_start + batch_size]
        patches = []
        for zs, hs, ws in batch:
            patch = bf_padded[zs:zs + patch_depth, hs:hs + spatial_tile, ws:ws + spatial_tile]
            # (D, H, W) -> (1, H, W, D) for model's BCHWD format
            patches.append(patch.transpose(1, 2, 0)[np.newaxis])

        inp = torch.from_numpy(np.stack(patches)).float().to(device)
        preds = model(inp)

        for i, (zs, hs, ws) in enumerate(batch):
            pred = preds[i, 0].cpu().numpy()
            if pred.ndim == 3:
                # pred is (H, W, D') — transpose to (D', H, W)
                pred = pred.transpose(2, 0, 1)
                pd, ph, pw = pred.shape
                w = gaussian_weight_3d(pd, ph, pw)
                predictions[zs:zs + pd, hs:hs + ph, ws:ws + pw] += pred * w
                weights[zs:zs + pd, hs:hs + ph, ws:ws + pw] += w
            elif pred.ndim == 2:
                predictions[zs, hs:hs + pred.shape[0], ws:ws + pred.shape[1]] += pred
                weights[zs, hs:hs + pred.shape[0], ws:ws + pred.shape[1]] += 1.0

    # Normalize by weights
    mask = weights > 0
    predictions[mask] /= weights[mask]

    # Unpad
    return predictions[:Z, :H, :W]


def main(config_path, checkpoint, output_dir):
    cfg = load_config(config_path)
    dims = cfg["model"]["dims"]
    apply_timm = cfg["model"].get("encoder_weights") is not None

    # Environment
    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    # Build model (no pretrained weights — loading from checkpoint)
    cfg_copy = cfg.copy()
    cfg_copy["model"] = cfg["model"].copy()
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)

    # Load checkpoint
    ckpt = load_checkpoint(checkpoint, model)
    accelerator.print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
                      f"(val_loss={ckpt.get('val_loss', '?')})")

    model = accelerator.prepare(model)
    model.eval()

    # Find input files
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    stats_dir = os.path.join(data_dir, "stats")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    accelerator.print(f"Found {len(bf_files)} volumes to process")

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for bf_path in tqdm(bf_files, desc="Predicting"):
            stem = os.path.splitext(os.path.basename(bf_path))[0]

            # Load stats
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            # Load and normalize
            bf_raw = np.load(bf_path)
            z_range = cfg["data"].get("z_range", None)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            # Predict
            if dims == "2d":
                pred = predict_2d(model, bf, device)
            else:
                patch_depth = cfg["data"].get("patch_depth", 32)
                overlap = cfg.get("inference", {}).get("overlap", [16, 128, 128])
                inf_batch = cfg.get("inference", {}).get("batch_size", 4)
                spatial_tile = cfg.get("inference", {}).get("spatial_tile", cfg["data"].get("crop_size", 256))
                pred = predict_3d(model, bf, device, patch_depth, overlap, inf_batch,
                                  spatial_tile=spatial_tile)

            # Save prediction (in [0,1] normalized space)
            out_path = os.path.join(output_dir, f"{stem}.npy")
            np.save(out_path, pred)
            accelerator.print(f"  {stem}: {pred.shape}, range=[{pred.min():.3f}, {pred.max():.3f}]")

    accelerator.print("Prediction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict GFP from brightfield")
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="predictions/",
                        help="Directory for output predictions")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)
