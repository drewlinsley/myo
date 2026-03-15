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
               batch_size=4):
    """Predict GFP from BF volume using 3D sliding window with Gaussian blending.

    Args:
        model: 3D U-Net model
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device
        patch_depth: depth of each 3D patch
        overlap: (z, h, w) overlap between patches
        batch_size: number of patches to process at once

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

    # Build Gaussian weight map for blending
    def gaussian_weight_3d(depth, height, width):
        gz = np.exp(-0.5 * ((np.linspace(-1, 1, depth)) ** 2))
        gh = np.exp(-0.5 * ((np.linspace(-1, 1, height)) ** 2))
        gw = np.exp(-0.5 * ((np.linspace(-1, 1, width)) ** 2))
        weight = gz[:, None, None] * gh[None, :, None] * gw[None, None, :]
        return weight.astype(np.float32)

    weight_map = gaussian_weight_3d(patch_depth, Hp, Wp)

    # Generate sliding window coordinates
    stride_z = max(1, patch_depth - overlap[0])
    stride_h = max(1, Hp - overlap[1]) if Hp <= overlap[1] + 32 else Hp  # full spatial
    stride_w = max(1, Wp - overlap[2]) if Wp <= overlap[2] + 32 else Wp

    # For spatial, use full volume (no spatial tiling for now since volumes are small)
    coords = []
    for z_start in range(0, max(1, Zp - patch_depth + 1), stride_z):
        z_end = min(z_start + patch_depth, Zp)
        z_start = z_end - patch_depth  # snap back if needed
        coords.append(z_start)

    # Ensure we include the last window
    if coords and coords[-1] + patch_depth < Zp:
        coords.append(Zp - patch_depth)
    coords = sorted(set(coords))

    predictions = np.zeros((Zp, Hp, Wp), dtype=np.float32)
    weights = np.zeros((Zp, Hp, Wp), dtype=np.float32)

    # Process in batches
    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start:batch_start + batch_size]
        patches = []
        for zs in batch_coords:
            patch = bf_padded[zs:zs + patch_depth]  # (D, H, W)
            # (D, H, W) -> (1, H, W, D) for model's BCHWD format
            patches.append(patch.transpose(1, 2, 0)[np.newaxis])  # (1, H, W, D)

        inp = torch.from_numpy(np.stack(patches)).float().to(device)  # (B, 1, H, W, D)
        preds = model(inp)  # (B, 1, H, W, D')

        for i, zs in enumerate(batch_coords):
            pred = preds[i, 0].cpu().numpy()  # (H, W, D') or similar
            # Handle potential temporal pooling
            if pred.ndim == 3:
                # pred is (H, W, D') — transpose to (D', H, W)
                pred = pred.transpose(2, 0, 1)
                pd = pred.shape[0]
                w = gaussian_weight_3d(pd, Hp, Wp)
                predictions[zs:zs + pd] += pred * w
                weights[zs:zs + pd] += w
            elif pred.ndim == 2:
                predictions[zs] += pred
                weights[zs] += 1.0

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
                pred = predict_3d(model, bf, device, patch_depth, overlap, inf_batch)

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
