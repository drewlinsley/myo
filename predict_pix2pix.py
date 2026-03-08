"""Inference script for pix2pix-turbo (Experiment 5).

Per-volume: load BF -> normalize to [0,1] -> for each Z-slice: tile 512x512
with overlap -> replicate to 3ch -> forward -> average channels -> [-1,1]->[0,1]
-> Gaussian blend -> stack -> save .npy

Usage:
    python predict_pix2pix.py -c configs/pix2pix_turbo.yaml \
        --checkpoint ckpts/pix2pix_turbo/best.pkl \
        --output_dir predictions/pix2pix_turbo
"""

import os
import json
import argparse
import numpy as np
import torch

from glob import glob
from src.config import load_config
from src.utils import prepare_env
from src.models.pix2pix_turbo import Pix2Pix_Turbo
from src.data.normalization import normalize


def make_gaussian_weight_2d(size):
    """Create a 2D Gaussian weight map for tile blending."""
    g = np.exp(-0.5 * (np.linspace(-1, 1, size) ** 2))
    weight = g[:, None] * g[None, :]
    return weight.astype(np.float32)


def predict_slice_tiled(model, bf_slice, prompt_tokens, device, tile_size=512,
                        tile_overlap=128):
    """Predict a single 2D slice using tiled inference with Gaussian blending.

    Args:
        model: Pix2Pix_Turbo model (in eval mode)
        bf_slice: (H, W) normalized float32 array in [0, 1]
        prompt_tokens: (1, seq_len) tokenized prompt tensor
        device: torch device
        tile_size: tile size in pixels
        tile_overlap: overlap between tiles

    Returns:
        (H, W) predicted GFP slice in [0, 1]
    """
    H, W = bf_slice.shape
    stride = tile_size - tile_overlap

    # Pad to ensure full coverage
    pad_h = (stride - (H - tile_size) % stride) % stride if H > tile_size else tile_size - H
    pad_w = (stride - (W - tile_size) % stride) % stride if W > tile_size else tile_size - W
    if pad_h > 0 or pad_w > 0:
        bf_padded = np.pad(bf_slice, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        bf_padded = bf_slice
    Hp, Wp = bf_padded.shape

    # If image fits in one tile, skip tiling
    if Hp <= tile_size and Wp <= tile_size:
        # Pad to tile_size exactly
        bf_tile = np.pad(bf_padded,
                         ((0, tile_size - Hp), (0, tile_size - Wp)),
                         mode="reflect")
        # (3, tile_size, tile_size) in [-1, 1]
        inp = torch.from_numpy(
            np.stack([bf_tile] * 3, axis=0)
        ).float().unsqueeze(0).to(device) * 2.0 - 1.0

        pred = model(inp, prompt_tokens=prompt_tokens.to(device), deterministic=True)
        pred_np = pred[0].cpu().numpy()  # (3, H, W) in [-1, 1]
        pred_gray = pred_np.mean(axis=0)  # average RGB channels
        pred_01 = (pred_gray + 1.0) / 2.0  # -> [0, 1]
        return np.clip(pred_01[:H, :W], 0, 1)

    # Gaussian weight for blending
    weight = make_gaussian_weight_2d(tile_size)
    output = np.zeros((Hp, Wp), dtype=np.float32)
    weight_sum = np.zeros((Hp, Wp), dtype=np.float32)

    # Generate tile coordinates
    y_coords = list(range(0, Hp - tile_size + 1, stride))
    if y_coords[-1] + tile_size < Hp:
        y_coords.append(Hp - tile_size)
    x_coords = list(range(0, Wp - tile_size + 1, stride))
    if x_coords[-1] + tile_size < Wp:
        x_coords.append(Wp - tile_size)
    y_coords = sorted(set(y_coords))
    x_coords = sorted(set(x_coords))

    for y in y_coords:
        for x in x_coords:
            tile = bf_padded[y:y + tile_size, x:x + tile_size]
            # (3, tile_size, tile_size) in [-1, 1]
            inp = torch.from_numpy(
                np.stack([tile] * 3, axis=0)
            ).float().unsqueeze(0).to(device) * 2.0 - 1.0

            pred = model(inp, prompt_tokens=prompt_tokens.to(device), deterministic=True)
            pred_np = pred[0].cpu().numpy()  # (3, H, W) in [-1, 1]
            pred_gray = pred_np.mean(axis=0)  # average RGB channels
            pred_01 = (pred_gray + 1.0) / 2.0  # -> [0, 1]

            output[y:y + tile_size, x:x + tile_size] += pred_01 * weight
            weight_sum[y:y + tile_size, x:x + tile_size] += weight

    # Normalize by weights
    mask = weight_sum > 0
    output[mask] /= weight_sum[mask]

    return np.clip(output[:H, :W], 0, 1)


def main(config_path, checkpoint, output_dir):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    icfg = cfg.get("inference", {})

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    # Build model and load checkpoint
    pretrained_model = mcfg.get("pretrained_model", "stabilityai/sd-turbo")
    accelerator.print(f"Loading model from {checkpoint}...")
    net = Pix2Pix_Turbo(
        pretrained_path=checkpoint,
        pretrained_model=pretrained_model,
    )
    net.to(device)
    net.set_eval()

    # Tokenize prompt
    prompt = "brightfield to GFP fluorescence"
    prompt_tokens = net.tokenizer(
        prompt,
        max_length=net.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids  # (1, seq_len)

    # Find input files
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    stats_dir = os.path.join(data_dir, "stats")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    accelerator.print(f"Found {len(bf_files)} volumes to process")

    tile_size = icfg.get("tile_size", 512)
    tile_overlap = icfg.get("tile_overlap", 128)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for bf_path in tqdm(bf_files, desc="Predicting"):
            stem = os.path.splitext(os.path.basename(bf_path))[0]

            # Load stats
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            # Load and normalize BF to [0, 1]
            bf_raw = np.load(bf_path)
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=False)

            Z, H, W = bf.shape
            predictions = np.zeros((Z, H, W), dtype=np.float32)

            for z in tqdm(range(Z), desc=f"  {stem}", leave=False):
                predictions[z] = predict_slice_tiled(
                    net, bf[z], prompt_tokens, device,
                    tile_size=tile_size, tile_overlap=tile_overlap,
                )

            out_path = os.path.join(output_dir, f"{stem}.npy")
            np.save(out_path, predictions)
            accelerator.print(
                f"  {stem}: {predictions.shape}, "
                f"range=[{predictions.min():.3f}, {predictions.max():.3f}]"
            )

    accelerator.print("Prediction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict GFP with pix2pix-turbo")
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to .pkl model checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default="predictions/pix2pix_turbo",
        help="Directory for output predictions",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)
