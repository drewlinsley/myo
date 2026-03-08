"""Evaluate predictions against ground truth GFP volumes.

Usage:
    python evaluate.py --pred_dir predictions/unet_2d_imagenet --data_dir data --output_dir results/unet_2d_imagenet
    python evaluate.py --compare results/unet_2d_imagenet results/unet_2d_random results/unet_3d_imagenet results/unet_3d_random
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob

from src.metrics import psnr, ssim, pearson_corr, mae
from src.data.normalization import normalize


def evaluate_volume(pred, gt):
    """Compute metrics between predicted and ground truth volumes (both in [0,1])."""
    return {
        "psnr": float(psnr(pred, gt)),
        "ssim": float(ssim(pred, gt)),
        "pearson": float(pearson_corr(pred, gt)),
        "mae": float(mae(pred, gt)),
    }


def evaluate_experiment(pred_dir, data_dir, output_dir):
    """Evaluate all predictions in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")
    pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))

    print(f"Evaluating {len(pred_files)} predictions from {pred_dir}")

    results = []
    for pred_path in pred_files:
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        gt_path = os.path.join(gfp_dir, f"{stem}.npy")
        stats_path = os.path.join(stats_dir, f"{stem}.json")

        if not os.path.exists(gt_path):
            print(f"  Skipping {stem}: no ground truth found")
            continue

        # Load prediction (already in [0,1])
        pred = np.load(pred_path).astype(np.float32)

        # Load and normalize ground truth to [0,1]
        gt_raw = np.load(gt_path)
        with open(stats_path) as f:
            stats = json.load(f)
        gt = normalize(gt_raw, stats["gfp"]["p_low"], stats["gfp"]["p_high"],
                       apply_timm=False)

        # Ensure same shape
        min_z = min(pred.shape[0], gt.shape[0])
        min_h = min(pred.shape[1], gt.shape[1])
        min_w = min(pred.shape[2], gt.shape[2])
        pred = pred[:min_z, :min_h, :min_w]
        gt = gt[:min_z, :min_h, :min_w]

        metrics = evaluate_volume(pred, gt)
        metrics["stem"] = stem
        results.append(metrics)
        print(f"  {stem}: PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
              f"Pearson={metrics['pearson']:.4f} MAE={metrics['mae']:.4f}")

    # Save per-volume results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "per_volume_metrics.csv"), index=False)

    # Summary
    if results:
        summary = {
            "n_volumes": len(results),
            "psnr_mean": float(df["psnr"].mean()),
            "psnr_std": float(df["psnr"].std()),
            "ssim_mean": float(df["ssim"].mean()),
            "ssim_std": float(df["ssim"].std()),
            "pearson_mean": float(df["pearson"].mean()),
            "pearson_std": float(df["pearson"].std()),
            "mae_mean": float(df["mae"].mean()),
            "mae_std": float(df["mae"].std()),
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary: PSNR={summary['psnr_mean']:.2f}±{summary['psnr_std']:.2f} "
              f"SSIM={summary['ssim_mean']:.4f}±{summary['ssim_std']:.4f}")

    # Save sample visualizations
    save_montages(pred_dir, data_dir, output_dir, max_files=5)

    return results


def save_montages(pred_dir, data_dir, output_dir, max_files=5):
    """Save prediction montages: BF | GT GFP | Predicted | |Error|."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping montages")
        return

    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")
    pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))[:max_files]

    montage_dir = os.path.join(output_dir, "montages")
    os.makedirs(montage_dir, exist_ok=True)

    for pred_path in pred_files:
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        bf_path = os.path.join(bf_dir, f"{stem}.npy")
        gt_path = os.path.join(gfp_dir, f"{stem}.npy")
        stats_path = os.path.join(stats_dir, f"{stem}.json")

        if not all(os.path.exists(p) for p in [bf_path, gt_path, stats_path]):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        bf_raw = np.load(bf_path)
        gt_raw = np.load(gt_path)
        pred = np.load(pred_path)

        bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"], apply_timm=False)
        gt = normalize(gt_raw, stats["gfp"]["p_low"], stats["gfp"]["p_high"], apply_timm=False)

        Z = min(bf.shape[0], gt.shape[0], pred.shape[0])
        z_indices = np.linspace(0, Z - 1, 5, dtype=int)

        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        for i, zi in enumerate(z_indices):
            axes[0, i].imshow(bf[zi], cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title(f"BF Z={zi}")
            axes[0, i].axis("off")

            axes[1, i].imshow(gt[zi], cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title(f"GT GFP Z={zi}")
            axes[1, i].axis("off")

            axes[2, i].imshow(pred[zi], cmap="gray", vmin=0, vmax=1)
            axes[2, i].set_title(f"Predicted Z={zi}")
            axes[2, i].axis("off")

            error = np.abs(gt[zi] - pred[zi])
            axes[3, i].imshow(error, cmap="hot", vmin=0, vmax=0.5)
            axes[3, i].set_title(f"|Error| Z={zi}")
            axes[3, i].axis("off")

        plt.suptitle(f"{stem}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(montage_dir, f"{stem}_montage.png"), dpi=150)
        plt.close()


def compare_experiments(result_dirs, output_path="results/comparison.csv"):
    """Create a comparison table across experiments."""
    rows = []
    for rdir in result_dirs:
        summary_path = os.path.join(rdir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"  Skipping {rdir}: no summary.json")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        summary["experiment"] = os.path.basename(rdir)
        rows.append(summary)

    if rows:
        df = pd.DataFrame(rows)
        cols = ["experiment", "n_volumes", "psnr_mean", "psnr_std",
                "ssim_mean", "ssim_std", "pearson_mean", "pearson_std",
                "mae_mean", "mae_std"]
        df = df[[c for c in cols if c in df.columns]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nComparison table saved to {output_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GFP predictions")
    parser.add_argument("--pred_dir", type=str, help="Directory with prediction .npy files")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory with gfp/ and stats/")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Output directory for results")
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="Compare multiple result directories")
    args = parser.parse_args()

    if args.compare:
        compare_experiments(args.compare)
    elif args.pred_dir:
        evaluate_experiment(args.pred_dir, args.data_dir, args.output_dir)
    else:
        parser.print_help()
