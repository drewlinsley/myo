"""Inference script for brightfield -> fluorescence prediction."""

import os
import argparse
import numpy as np
import torch
from glob import glob

import segmentation_models_pytorch_3d as smp
from src import utils


def main(config):
    cfg = utils.read_config(config)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    icfg = cfg["inference"]

    os.makedirs(icfg["output_dir"], exist_ok=True)

    # Prepare environment
    accelerator, device, tqdm, TIMM = utils.prepare_env(mcfg["encoder"])
    mu = np.float32(TIMM["mean"][0])
    std = np.float32(TIMM["std"][0])

    # Build model
    model = smp.create_model(
        mcfg["arch"],
        encoder_name=mcfg["encoder"],
        encoder_weights=None,  # loading from checkpoint
        in_channels=mcfg["in_channels"],
        classes=mcfg["out_channels"],
        activation=None,
        time_kernel=mcfg.get("time_kernel", [32, 1, 1]),
    )

    # Load checkpoint
    ckpt = torch.load(icfg["checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    accelerator.print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', '?')})")

    model = accelerator.prepare(model)
    model.eval()

    # Find input files
    bf_files = sorted(glob(os.path.join(dcfg["brightfield_dir"], "**", dcfg["file_pattern"]), recursive=True))
    accelerator.print(f"Found {len(bf_files)} brightfield files to process")

    hw = dcfg["hw"]
    timesteps = dcfg["timesteps"]

    with torch.no_grad():
        for bf_path in tqdm(bf_files, desc="Predicting"):
            data = np.load(bf_path).astype(np.float32)
            if data.ndim == 3:
                data = data[..., np.newaxis]
            elif data.ndim == 4:
                data = data[..., :1]

            # Normalize
            dmin, dmax = data.min(), data.max()
            if dmax - dmin > 0:
                data = (data - dmin) / (dmax - dmin)
            data = (data - mu) / std

            T, H, W, C = data.shape
            predictions = np.zeros((T, H, W), dtype=np.float32)
            counts = np.zeros((T, H, W), dtype=np.float32)

            # Sliding window over time
            for t_start in range(0, max(1, T - timesteps + 1), timesteps // 2):
                t_end = min(t_start + timesteps, T)
                chunk = data[t_start:t_end]

                # Pad temporally if needed
                if len(chunk) < timesteps:
                    pad_t = timesteps - len(chunk)
                    chunk = np.pad(chunk, ((0, pad_t), (0, 0), (0, 0), (0, 0)))

                # Pad spatially to be divisible by 32
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                if pad_h > 0 or pad_w > 0:
                    chunk = np.pad(chunk, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

                # (T, H, W, C) -> (1, C, T, H, W)
                inp = torch.from_numpy(chunk.transpose(3, 0, 1, 2)[np.newaxis]).float().to(device)
                pred = model(inp)

                # Unpad and accumulate
                pred = pred[0, 0].cpu().numpy()  # (T_out, H_out, W_out) or similar
                # The time_kernel may reduce temporal dim - handle accordingly
                if pred.ndim == 3:
                    pt, ph, pw = pred.shape
                    pred = pred[:min(pt, t_end - t_start), :H, :W]
                    actual_t = pred.shape[0]
                    predictions[t_start:t_start + actual_t] += pred
                    counts[t_start:t_start + actual_t] += 1
                elif pred.ndim == 2:
                    # Single frame output per window
                    predictions[min(t_end - 1, T - 1)] += pred[:H, :W]
                    counts[min(t_end - 1, T - 1)] += 1

            # Average overlapping predictions
            mask = counts > 0
            predictions[mask] /= counts[mask]

            # Save
            rel_path = os.path.relpath(bf_path, dcfg["brightfield_dir"])
            out_path = os.path.join(icfg["output_dir"], rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, predictions)

    accelerator.print("Prediction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fluorescence from brightfield stacks")
    parser.add_argument('-c', '--config', required=True, type=str, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
