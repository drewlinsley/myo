"""Plot BF, GFP, and GFP thresholded at several percentiles for visual inspection.

Usage:
    python scripts/plot_thresholds.py --data_dir data --output_dir threshold_plots
    python scripts/plot_thresholds.py --data_dir data --percentiles 80 90 95 99 --max_files 5
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob


def plot_thresholds_for_volume(bf_path, gfp_path, stats_path, percentiles, output_dir):
    """Plot mid-Z slice: BF | GFP | thresholded at each percentile."""
    stem = os.path.splitext(os.path.basename(bf_path))[0]

    bf = np.load(bf_path).astype(np.float32)
    gfp = np.load(gfp_path).astype(np.float32)

    with open(stats_path) as f:
        stats = json.load(f)

    # Normalize BF and GFP to [0, 1] for display
    bf_disp = np.clip(bf, stats["bf"]["p_low"], stats["bf"]["p_high"])
    bf_disp = (bf_disp - stats["bf"]["p_low"]) / (stats["bf"]["p_high"] - stats["bf"]["p_low"] + 1e-8)

    gfp_disp = np.clip(gfp, stats["gfp"]["p_low"], stats["gfp"]["p_high"])
    gfp_disp = (gfp_disp - stats["gfp"]["p_low"]) / (stats["gfp"]["p_high"] - stats["gfp"]["p_low"] + 1e-8)

    # Compute percentile thresholds on raw GFP
    thresholds = {p: np.percentile(gfp, p) for p in percentiles}

    # Pick 5 evenly spaced Z-slices
    Z = bf.shape[0]
    z_indices = np.linspace(0, Z - 1, 5, dtype=int)

    n_cols = 2 + len(percentiles)  # BF, GFP, then one per threshold
    n_rows = len(z_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for row, zi in enumerate(z_indices):
        # BF
        axes[row, 0].imshow(bf_disp[zi], cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title("BF", fontsize=10)
        axes[row, 0].set_ylabel(f"Z={zi}", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # GFP
        axes[row, 1].imshow(gfp_disp[zi], cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, 1].set_title("GFP", fontsize=10)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # Thresholded
        for col, p in enumerate(percentiles):
            mask = (gfp[zi] >= thresholds[p]).astype(np.float32)
            frac = mask.mean() * 100

            axes[row, col + 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, col + 2].set_title(f"p{p}\n(thr={thresholds[p]:.0f})", fontsize=9)
            if row == n_rows - 1:
                axes[row, col + 2].set_xlabel(f"{frac:.1f}% pos", fontsize=8)
            axes[row, col + 2].set_xticks([])
            axes[row, col + 2].set_yticks([])

    plt.suptitle(f"{stem}  (Z={Z})", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{stem}_thresholds.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")

    # Print threshold summary
    for p in percentiles:
        mask = (gfp >= thresholds[p]).astype(np.float32)
        print(f"    p{p}: threshold={thresholds[p]:.0f}, positive={mask.mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot GFP threshold comparisons")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="threshold_plots")
    parser.add_argument("--percentiles", nargs="+", type=float,
                        default=[80, 90, 95, 99])
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bf_dir = os.path.join(args.data_dir, "bf")
    gfp_dir = os.path.join(args.data_dir, "gfp")
    stats_dir = os.path.join(args.data_dir, "stats")

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))[:args.max_files]
    print(f"Plotting {len(bf_files)} volumes with percentiles {args.percentiles}")

    for bf_path in bf_files:
        stem = os.path.splitext(os.path.basename(bf_path))[0]
        gfp_path = os.path.join(gfp_dir, f"{stem}.npy")
        stats_path = os.path.join(stats_dir, f"{stem}.json")

        if not os.path.exists(gfp_path):
            print(f"  Skipping {stem}: no GFP file")
            continue

        print(f"{stem}:")
        plot_thresholds_for_volume(bf_path, gfp_path, stats_path,
                                   args.percentiles, args.output_dir)


if __name__ == "__main__":
    main()
