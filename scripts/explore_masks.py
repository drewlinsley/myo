"""Explore BF-based foreground masks: percentile thresholds + algorithmic methods.

For each volume, saves:
  {stem}_masks.png      – grid: Z-slices x (BF, GFP, 14 mask methods)
  {stem}_histogram.png  – BF intensity histogram with threshold lines

Summary outputs:
  summary.json           – per-volume per-method stats + cross-volume means
  summary_fg_fractions.png – grouped bar chart of fg fraction by method

Usage:
    python scripts/explore_masks.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --output_dir results/mask_exploration/ \
        --n_volumes 5 \
        --n_z 5
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob

from scipy.ndimage import binary_dilation, generate_binary_structure, label as ndimage_label
from skimage.filters import (
    threshold_otsu,
    threshold_li,
    threshold_triangle,
    threshold_minimum,
    threshold_multiotsu,
)

from src.config import load_config

PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
ALGO_METHODS = ["otsu", "li", "triangle", "minimum", "multiotsu"]
# (dilation_iters, min_component_frac_of_slice_fg)
CLEANUP_CONFIGS = [
    (0, 0.01),   # no dilation, remove blobs <1% of fg
    (0, 0.05),   # no dilation, remove blobs <5% of fg
    (3, 0.01),   # small dilation + remove tiny
    (3, 0.05),   # small dilation + remove small
    (5, 0.01),   # medium dilation + remove tiny
    (5, 0.05),   # medium dilation + remove small
]
CLEANUP_METHODS = [f"minimum_d{d}_r{int(s*100):02d}" for d, s in CLEANUP_CONFIGS]
ALL_METHODS = [f"p{p}" for p in PERCENTILES] + ALGO_METHODS + CLEANUP_METHODS


def compute_volume_thresholds(bf_vol):
    """Compute all thresholds on a flattened BF volume.

    Returns dict[method_name, threshold_value].  NaN on failure.
    Dilation methods (minimum_dN) store the base minimum threshold;
    actual dilation is applied when building masks.
    """
    flat = bf_vol.ravel().astype(np.float64)
    thresholds = {}

    # Percentile-based
    for p in PERCENTILES:
        thresholds[f"p{p}"] = float(np.percentile(flat, p))

    # Algorithmic
    try:
        thresholds["otsu"] = float(threshold_otsu(flat))
    except Exception:
        thresholds["otsu"] = float("nan")

    try:
        thresholds["li"] = float(threshold_li(flat))
    except Exception:
        thresholds["li"] = float("nan")

    try:
        thresholds["triangle"] = float(threshold_triangle(flat))
    except Exception:
        thresholds["triangle"] = float("nan")

    try:
        thresholds["minimum"] = float(threshold_minimum(flat))
    except (RuntimeError, ValueError, Exception):
        thresholds["minimum"] = float("nan")

    try:
        # 3-class multi-otsu: take the lower threshold
        t = threshold_multiotsu(flat, classes=3)
        thresholds["multiotsu"] = float(t[0])
    except (ValueError, Exception):
        thresholds["multiotsu"] = float("nan")

    # Cleanup variants of minimum: same threshold, dilation + component removal later
    for d, s in CLEANUP_CONFIGS:
        thresholds[f"minimum_d{d}_r{int(s*100):02d}"] = thresholds["minimum"]

    return thresholds


def apply_mask(bf_vol, method, thresholds):
    """Build a boolean foreground mask for a given method.

    For minimum_dN_rMM methods: threshold with minimum, dilate by N iterations,
    then remove connected components smaller than MM% of per-slice foreground.
    Returns (Z, H, W) bool array.
    """
    thr = thresholds[method]
    if np.isnan(thr):
        return None
    mask = bf_vol > thr
    if method.startswith("minimum_d"):
        # Parse dilation iters and min component fraction
        # Format: minimum_d{N}_r{MM}
        parts = method.split("_")
        n_iter = int(parts[1][1:])   # d{N}
        min_frac = int(parts[2][1:]) / 100.0  # r{MM}
        struct = generate_binary_structure(2, 1)  # 2D cross
        for z in range(mask.shape[0]):
            if n_iter > 0:
                mask[z] = binary_dilation(
                    mask[z], structure=struct, iterations=n_iter)
            # Remove small connected components
            fg_count = mask[z].sum()
            if fg_count > 0 and min_frac > 0:
                labeled, n_components = ndimage_label(mask[z])
                min_pixels = fg_count * min_frac
                for comp_id in range(1, n_components + 1):
                    comp_mask = labeled == comp_id
                    if comp_mask.sum() < min_pixels:
                        mask[z][comp_mask] = False
    return mask


def plot_mask_grid(bf_vol, gfp_vol, thresholds, z_indices, stem, output_dir):
    """Create grid figure: rows=Z slices, cols=BF|GFP|14 masks.

    Returns per-method stats dict {method: {threshold, fg_fraction, fg_mean_bf, bg_mean_bf}}.
    """
    n_rows = len(z_indices)
    n_cols = 2 + len(ALL_METHODS)

    # Display-normalize BF with 1st/99th percentile clip
    p_lo, p_hi = np.percentile(bf_vol, [1, 99])
    bf_disp = np.clip(bf_vol.astype(np.float32), p_lo, p_hi)
    bf_disp = (bf_disp - p_lo) / (p_hi - p_lo + 1e-8)

    # Display-normalize GFP similarly
    gp_lo, gp_hi = np.percentile(gfp_vol, [1, 99])
    gfp_disp = np.clip(gfp_vol.astype(np.float32), gp_lo, gp_hi)
    gfp_disp = (gfp_disp - gp_lo) / (gp_hi - gp_lo + 1e-8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Compute volume-level stats per method (with dilation where applicable)
    stats = {}
    vol_masks = {}  # cache for rendering
    for method in ALL_METHODS:
        mask = apply_mask(bf_vol, method, thresholds)
        vol_masks[method] = mask
        if mask is None:
            stats[method] = dict(threshold=None, fg_fraction=None,
                                 fg_mean_bf=None, bg_mean_bf=None)
        else:
            fg_frac = float(mask.mean())
            fg_mean = float(bf_vol[mask].mean()) if mask.any() else 0.0
            bg_mean = float(bf_vol[~mask].mean()) if (~mask).any() else 0.0
            stats[method] = dict(threshold=float(thresholds[method]),
                                 fg_fraction=fg_frac,
                                 fg_mean_bf=fg_mean, bg_mean_bf=bg_mean)

    for row, zi in enumerate(z_indices):
        # BF reference
        axes[row, 0].imshow(bf_disp[zi], cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title("BF", fontsize=8, fontweight="bold")
        axes[row, 0].set_ylabel(f"Z={zi}", fontsize=7)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # GFP reference
        axes[row, 1].imshow(gfp_disp[zi], cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, 1].set_title("GFP", fontsize=8, fontweight="bold")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # Mask columns
        for col_idx, method in enumerate(ALL_METHODS):
            ax = axes[row, col_idx + 2]
            mask = vol_masks[method]

            if mask is None:
                ax.imshow(bf_disp[zi], cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "FAIL", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="red",
                        fontweight="bold")
                if row == 0:
                    ax.set_title(f"{method}\nFAIL", fontsize=7)
            else:
                # BF grayscale with semi-transparent red overlay on background
                bf_slice = bf_disp[zi]
                rgba = np.zeros((*bf_slice.shape, 4), dtype=np.float32)
                rgba[..., 0] = bf_slice
                rgba[..., 1] = bf_slice
                rgba[..., 2] = bf_slice
                rgba[..., 3] = 1.0

                # Red overlay on background (mask=0) pixels
                bg_mask = ~mask[zi]
                rgba[bg_mask, 0] = np.clip(rgba[bg_mask, 0] * 0.7 + 0.3, 0, 1)
                rgba[bg_mask, 1] = rgba[bg_mask, 1] * 0.7
                rgba[bg_mask, 2] = rgba[bg_mask, 2] * 0.7

                ax.imshow(rgba)

                if row == 0:
                    fg_pct = stats[method]["fg_fraction"]
                    fg_str = f"{fg_pct * 100:.1f}" if fg_pct is not None else "?"
                    thr = thresholds[method]
                    ax.set_title(f"{method}\nthr={thr:.0f} / fg={fg_str}%",
                                 fontsize=6)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"{stem}", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, f"{stem}_masks.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")

    return stats


def plot_histogram(bf_vol, thresholds, stem, output_dir):
    """BF intensity histogram with vertical lines for each threshold method."""
    flat = bf_vol.ravel()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(flat, bins=200, color="0.7", edgecolor="none", log=True)

    # Blue shades for percentiles
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(PERCENTILES)))
    for i, p in enumerate(PERCENTILES):
        method = f"p{p}"
        thr = thresholds[method]
        ax.axvline(thr, color=blues[i], linewidth=1.2, linestyle="--",
                   label=f"{method} = {thr:.0f}")

    # Distinct colors for algorithmic
    algo_colors = {"otsu": "red", "li": "green", "triangle": "orange",
                   "minimum": "purple", "multiotsu": "brown"}
    for method in ALGO_METHODS:
        thr = thresholds[method]
        if np.isnan(thr):
            continue
        ax.axvline(thr, color=algo_colors[method], linewidth=1.8, linestyle="-",
                   label=f"{method} = {thr:.0f}")

    ax.set_xlabel("BF intensity")
    ax.set_ylabel("Count (log)")
    ax.set_title(f"{stem} — BF intensity distribution")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{stem}_histogram.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def plot_summary(all_stats, output_dir):
    """Grouped bar chart: x=methods, y=fg_fraction, one bar per volume + mean line."""
    stems = list(all_stats.keys())
    methods = ALL_METHODS
    n_methods = len(methods)
    n_volumes = len(stems)

    fig, ax = plt.subplots(figsize=(14, 5))

    bar_width = 0.8 / max(n_volumes, 1)
    x = np.arange(n_methods)

    for vi, stem in enumerate(stems):
        fracs = []
        for m in methods:
            f = all_stats[stem].get(m, {}).get("fg_fraction")
            fracs.append(f if f is not None else 0.0)
        offset = (vi - n_volumes / 2 + 0.5) * bar_width
        ax.bar(x + offset, [f * 100 for f in fracs], bar_width,
               label=stem[:20], alpha=0.7)

    # Mean line
    means = []
    for m in methods:
        vals = [all_stats[s][m]["fg_fraction"] for s in stems
                if all_stats[s].get(m, {}).get("fg_fraction") is not None]
        means.append(np.mean(vals) * 100 if vals else 0.0)
    ax.plot(x, means, "k-o", linewidth=2, markersize=5, label="mean", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Foreground fraction (%)")
    ax.set_title("Foreground fraction by masking method")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "summary_fg_fractions.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore BF-based foreground masks")
    parser.add_argument("-c", "--config", required=True, help="YAML config")
    parser.add_argument("--output_dir", default="results/mask_exploration")
    parser.add_argument("--n_volumes", type=int, default=5,
                        help="Max volumes to process")
    parser.add_argument("--n_z", type=int, default=5,
                        help="Number of Z-slices per volume")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    z_range = cfg["data"].get("z_range", None)

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))[:args.n_volumes]
    print(f"Processing {len(bf_files)} volumes, {args.n_z} Z-slices each")

    all_stats = {}

    for bf_path in bf_files:
        stem = os.path.splitext(os.path.basename(bf_path))[0]
        gfp_path = os.path.join(gfp_dir, f"{stem}.npy")
        if not os.path.exists(gfp_path):
            print(f"  Skipping {stem}: no GFP file")
            continue

        print(f"\n{stem}:")

        # Load volumes
        bf_raw = np.load(bf_path).astype(np.float32)
        gfp_raw = np.load(gfp_path).astype(np.float32)

        # Apply z_range
        if z_range is not None:
            z_lo = max(0, z_range[0])
            z_hi = min(bf_raw.shape[0], z_range[1])
            bf_raw = bf_raw[z_lo:z_hi]
            gfp_raw = gfp_raw[z_lo:z_hi]

        Z = bf_raw.shape[0]
        z_indices = np.linspace(0, Z - 1, min(args.n_z, Z), dtype=int)

        # Compute thresholds
        thresholds = compute_volume_thresholds(bf_raw)
        for m, t in thresholds.items():
            if np.isnan(t):
                print(f"  {m}: FAILED")
            else:
                fg = (bf_raw > t).mean() * 100
                print(f"  {m}: thr={t:.0f}, fg={fg:.1f}%")

        # Plot grid + histogram
        vol_stats = plot_mask_grid(bf_raw, gfp_raw, thresholds, z_indices,
                                   stem, args.output_dir)
        plot_histogram(bf_raw, thresholds, stem, args.output_dir)

        all_stats[stem] = vol_stats

    # Summary JSON
    # Add cross-volume means
    summary = {"per_volume": all_stats}
    means = {}
    stems = list(all_stats.keys())
    for m in ALL_METHODS:
        vals = {}
        for key in ["threshold", "fg_fraction", "fg_mean_bf", "bg_mean_bf"]:
            v = [all_stats[s][m][key] for s in stems
                 if all_stats[s].get(m, {}).get(key) is not None]
            vals[key] = float(np.mean(v)) if v else None
        means[m] = vals
    summary["cross_volume_means"] = means

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")

    # Summary figure
    if all_stats:
        plot_summary(all_stats, args.output_dir)


if __name__ == "__main__":
    main()
