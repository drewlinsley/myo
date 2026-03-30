"""Evaluate multiple metrics on the val set.

Metrics computed (all within non-background GT mask unless noted):
  - Masked Pearson correlation
  - Masked MSE
  - Masked MAE
  - SSIM (full-image, per-slice average)

Usage:
    python eval_masked_pearson.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson_frac025/best.pth \
        --output results/frac025.json

    # For 0% baseline (untrained model):
    python eval_masked_pearson.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --no_checkpoint \
        --output results/frac000.json
"""

import os
import json
import argparse
import numpy as np
import torch

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint, make_train_val_split
from src.data.normalization import normalize

try:
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


def pearson_r(a, b):
    """Pearson correlation between two 1-D arrays."""
    if len(a) < 2:
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if den == 0:
        return float("nan")
    return float(num / den)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics on val set")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use untrained model (0%% baseline)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.no_checkpoint and args.checkpoint is None:
        parser.error("Either --checkpoint or --no_checkpoint is required")

    cfg = load_config(args.config)
    dims = cfg["model"]["dims"]
    assert dims == "2d", "Only 2D models supported for now"
    apply_timm = cfg["model"].get("encoder_weights") is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_checkpoint:
        model = build_model(cfg)
        print("Using untrained model (0% baseline)")
    else:
        cfg_copy = cfg.copy()
        cfg_copy["model"] = cfg["model"].copy()
        cfg_copy["model"]["encoder_weights"] = None
        model = build_model(cfg_copy)
        ckpt = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(val_loss={ckpt.get('val_loss', '?')})")

    model = model.to(device)
    model.eval()

    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]

    _, val_stems = make_train_val_split(
        stems, val_fraction=cfg["data"].get("val_fraction", 0.15),
        seed=cfg.get("seed", 42))

    print(f"Evaluating on {len(val_stems)} val volumes")
    if not HAS_SSIM:
        print("WARNING: scikit-image not installed, skipping SSIM")

    z_range = cfg["data"].get("z_range", None)

    all_pred_pixels = []
    all_gt_pixels = []
    all_ssim = []
    per_volume = {}

    with torch.no_grad():
        for stem in val_stems:
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            bf_raw = np.load(os.path.join(bf_dir, f"{stem}.npy"))
            gfp_raw = np.load(os.path.join(gfp_dir, f"{stem}.npy"))

            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
                gfp_raw = gfp_raw[z_lo:z_hi]

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)
            gfp = normalize(gfp_raw, stats["gfp"]["p_low"], stats["gfp"]["p_high"],
                            apply_timm=False)

            gfp_thresh = np.percentile(gfp_raw, 1)

            Z, H, W = bf.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32

            vol_pred = []
            vol_gt = []
            vol_ssim = []

            for z in range(Z):
                slc = bf[z]
                if pad_h > 0 or pad_w > 0:
                    slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")

                inp = torch.from_numpy(slc[np.newaxis, np.newaxis]).float().to(device)
                pred = model(inp)[0, 0].cpu().numpy()[:H, :W]

                gt = gfp[z]

                # SSIM on full image (clipped to [0,1])
                if HAS_SSIM:
                    s = compute_ssim(
                        np.clip(gt, 0, 1).astype(np.float64),
                        np.clip(pred, 0, 1).astype(np.float64),
                        data_range=1.0)
                    vol_ssim.append(s)

                # Masked metrics
                mask = gfp_raw[z] > gfp_thresh
                if mask.sum() < 2:
                    continue

                vol_pred.append(pred[mask])
                vol_gt.append(gt[mask])

            if vol_pred:
                vp = np.concatenate(vol_pred)
                vg = np.concatenate(vol_gt)
                vol_r = pearson_r(vp, vg)
                vol_mse = float(np.mean((vp - vg) ** 2))
                vol_mae = float(np.mean(np.abs(vp - vg)))
                all_pred_pixels.append(vp)
                all_gt_pixels.append(vg)
            else:
                vol_r = float("nan")
                vol_mse = float("nan")
                vol_mae = float("nan")

            vol_ssim_mean = float(np.mean(vol_ssim)) if vol_ssim else float("nan")
            if vol_ssim:
                all_ssim.extend(vol_ssim)

            n_px = sum(p.size for p in vol_pred)
            per_volume[stem] = {
                "masked_pearson": vol_r,
                "masked_mse": vol_mse,
                "masked_mae": vol_mae,
                "ssim": vol_ssim_mean,
                "n_masked_pixels": n_px,
            }
            print(f"  {stem}: Pearson={vol_r:.4f}  MSE={vol_mse:.4f}  "
                  f"MAE={vol_mae:.4f}  SSIM={vol_ssim_mean:.4f}  "
                  f"({n_px:,} masked px)")

    # Global metrics across all masked pixels
    if all_pred_pixels:
        global_pred = np.concatenate(all_pred_pixels)
        global_gt = np.concatenate(all_gt_pixels)
        overall_pearson = pearson_r(global_pred, global_gt)
        overall_mse = float(np.mean((global_pred - global_gt) ** 2))
        overall_mae = float(np.mean(np.abs(global_pred - global_gt)))
    else:
        overall_pearson = float("nan")
        overall_mse = float("nan")
        overall_mae = float("nan")

    overall_ssim = float(np.mean(all_ssim)) if all_ssim else float("nan")

    total_px = sum(v["n_masked_pixels"] for v in per_volume.values())
    print(f"\nOverall:  Pearson={overall_pearson:.4f}  MSE={overall_mse:.4f}  "
          f"MAE={overall_mae:.4f}  SSIM={overall_ssim:.4f}  "
          f"({total_px:,} total masked px)")

    results = {
        "overall_masked_pearson": overall_pearson,
        "overall_masked_mse": overall_mse,
        "overall_masked_mae": overall_mae,
        "overall_ssim": overall_ssim,
        "n_volumes": len(val_stems),
        "per_volume": per_volume,
        "checkpoint": args.checkpoint or "none (untrained)",
    }

    out_path = args.output
    if out_path is None:
        if args.checkpoint:
            ckpt_dir = os.path.dirname(args.checkpoint)
            out_path = os.path.join(ckpt_dir, "metrics.json")
        else:
            out_path = "metrics_frac000.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
