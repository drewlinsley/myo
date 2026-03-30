"""Plot metrics vs. training data fraction (power law curves).

Also plots val loss curves from each fraction's training log.
Fits log curves and extrapolates to 200%.

Usage:
    python plot_power_law.py --results_dir results/power_law/ --output results/power_law/power_law.png
"""

import os
import re
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (json_key, display_label, higher_is_better, transform)
# transform: None = use raw value, "one_minus" = plot 1 - value
METRICS = [
    ("overall_masked_pearson", "Pearson Loss (1 - r)", False, "one_minus"),
    ("overall_masked_mse",     "Masked MSE",           False, None),
    ("overall_masked_mae",     "Masked MAE",           False, None),
    ("overall_ssim",           "SSIM",                 True,  None),
]

FRAC_COLORS = {
    "frac001": "#e6194b",
    "frac025": "#f58231",
    "frac050": "#ffe119",
    "frac100": "#4363d8",
}

EXCLUDE_FRACS = {0.75}  # outlier


def get_n_train_100(ckpt_base):
    """Try to read total training volume count from split.json."""
    split_path = os.path.join(f"{ckpt_base}_frac100", "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            data = json.load(f)
        return len(data.get("train", []))
    # Fallback: try any other frac dir
    for tag in ["frac050", "frac025", "frac001"]:
        split_path = os.path.join(f"{ckpt_base}_{tag}", "split.json")
        if os.path.exists(split_path):
            with open(split_path) as f:
                data = json.load(f)
            frac = data.get("fraction", 1.0)
            if frac > 0:
                return int(len(data["train"]) / frac)
    return None


def plot_val_losses(ckpt_base, output_path):
    """Plot val loss curves from each fraction's log.csv."""
    fig, ax = plt.subplots(figsize=(8, 5))
    found_any = False

    for tag, color in FRAC_COLORS.items():
        log_path = os.path.join(f"{ckpt_base}_{tag}", "log.csv")
        if not os.path.exists(log_path):
            continue

        epochs, val_losses = [], []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row["epoch"]))
                    val_losses.append(float(row["val_loss"]))
                except (ValueError, KeyError):
                    continue

        if not epochs:
            continue

        pct = tag.replace("frac0", "").replace("frac", "")
        label = f"{int(pct)}%"
        ax.plot(epochs, val_losses, color=color, linewidth=1.5, label=label)
        found_any = True

    if not found_any:
        print("No log.csv files found — skipping val loss plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Loss", fontsize=11)
    ax.set_title("Validation Loss by Training Fraction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved val loss plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot power law curves")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing frac*.json result files")
    parser.add_argument("--ckpt_base", default="ckpts/unet_2d_imagenet_pearson",
                        help="Base checkpoint dir (without _fracXXX suffix)")
    parser.add_argument("--n_train_100", type=int, default=None,
                        help="Number of training volumes at 100%% (auto-detected if omitted)")
    parser.add_argument("--output", default="power_law.png",
                        help="Output plot path")
    args = parser.parse_args()

    # Plot val loss curves
    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    plot_val_losses(args.ckpt_base, os.path.join(out_dir, "val_losses.png"))

    # Detect n_train
    n_train_100 = args.n_train_100 or get_n_train_100(args.ckpt_base)
    if n_train_100:
        print(f"Training volumes at 100%: {n_train_100}")
    else:
        print("Could not detect n_train_100 — x-axis will show % only")

    # Collect results
    fractions = []
    all_metrics = {key: [] for key, _, _, _ in METRICS}

    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.endswith(".json"):
            continue
        m = re.match(r"frac(\d+)\.json", fname)
        if not m:
            continue
        frac_pct = int(m.group(1))
        frac = frac_pct / 100.0

        if frac in EXCLUDE_FRACS:
            print(f"  Excluding {frac_pct}% (outlier)")
            continue

        with open(os.path.join(args.results_dir, fname)) as f:
            data = json.load(f)

        fractions.append(frac)
        for key, _, _, transform in METRICS:
            val = data.get(key, float("nan"))
            if val is None:
                val = float("nan")
            if transform == "one_minus" and not np.isnan(val):
                val = 1.0 - val
            all_metrics[key].append(val)
        print(f"  {frac_pct}%: " + "  ".join(
            f"{label}={all_metrics[key][-1]:.4f}"
            for key, label, _, _ in METRICS))

    if len(fractions) < 2:
        print("Not enough data points to plot.")
        return

    fractions = np.array(fractions)
    order = np.argsort(fractions)
    fractions = fractions[order]
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])[order]

    active_metrics = [(k, l, h, t) for k, l, h, t in METRICS
                      if not np.all(np.isnan(all_metrics[k]))]

    n_metrics = len(active_metrics)
    cols = min(n_metrics, 2)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for idx, (key, label, higher_better, _transform) in enumerate(active_metrics):
        ax = axes[idx // cols, idx % cols]
        values = all_metrics[key]
        valid = ~np.isnan(values)

        # Separate 0% baseline from trained points
        is_zero = fractions == 0.0
        trained_mask = valid & ~is_zero
        zero_mask = valid & is_zero

        # Plot 0% as distinct marker
        if zero_mask.any():
            ax.plot(fractions[zero_mask] * 100, values[zero_mask], "D",
                    color="#999999", markersize=9, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5, label="0% (untrained)")
            for frac, val in zip(fractions[zero_mask], values[zero_mask]):
                ax.annotate(f"{val:.3f}", (frac * 100, val),
                            textcoords="offset points", xytext=(8, -5),
                            ha="left", fontsize=9, color="#666666")

        # Plot trained data points
        if trained_mask.any():
            ax.plot(fractions[trained_mask] * 100, values[trained_mask], "o",
                    color="#4363d8", markersize=8, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            for frac, val in zip(fractions[trained_mask], values[trained_mask]):
                ax.annotate(f"{val:.3f}", (frac * 100, val),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9)

        # Pareto-front fit: only use points on the improving frontier
        # so the fit is monotonically improving with more data.
        fit_fracs = fractions[trained_mask]
        fit_vals = values[trained_mask]

        # Sort by fraction (should already be, but be safe)
        sort_idx = np.argsort(fit_fracs)
        fit_fracs = fit_fracs[sort_idx]
        fit_vals = fit_vals[sort_idx]

        # Keep only Pareto-optimal points
        pareto_mask = np.zeros(len(fit_fracs), dtype=bool)
        if higher_better:
            best = -np.inf
            for i in range(len(fit_fracs)):
                if fit_vals[i] >= best:
                    best = fit_vals[i]
                    pareto_mask[i] = True
        else:
            best = np.inf
            for i in range(len(fit_fracs)):
                if fit_vals[i] <= best:
                    best = fit_vals[i]
                    pareto_mask[i] = True

        pareto_fracs = fit_fracs[pareto_mask]
        pareto_vals = fit_vals[pareto_mask]

        # Mark non-Pareto points with lighter color
        non_pareto = ~pareto_mask
        if non_pareto.any():
            ax.plot(fit_fracs[non_pareto] * 100, fit_vals[non_pareto], "o",
                    color="#4363d8", markersize=8, markeredgecolor="white",
                    markeredgewidth=1.5, alpha=0.3, zorder=4)

        # Log fit on Pareto front: y = a * ln(x) + b
        if len(pareto_fracs) >= 2:
            coeffs = np.polyfit(np.log(pareto_fracs), pareto_vals, 1)
            a, b = coeffs[0], coeffs[1]
            x_fit = np.linspace(pareto_fracs.min(), 2.0, 200)
            y_fit = a * np.log(x_fit) + b

            ax.plot(x_fit * 100, y_fit, "--", color="#4363d8", alpha=0.4,
                    linewidth=1.5, label="log fit (Pareto)")

            # Equation text + R^2
            a_sign = "+" if b >= 0 else "-"
            eq_str = f"y = {a:.4f} ln(x) {a_sign} {abs(b):.4f}"
            ss_res = np.sum((pareto_vals - (a * np.log(pareto_fracs) + b)) ** 2)
            ss_tot = np.sum((pareto_vals - pareto_vals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            eq_str += f"   ($R^2$ = {r2:.3f})"

            eq_y = 0.95 if not higher_better else 0.05
            eq_va = "top" if not higher_better else "bottom"
            ax.text(0.03, eq_y, eq_str, transform=ax.transAxes,
                    fontsize=9, verticalalignment=eq_va,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))

            # Projected value at 200%
            y_200 = a * np.log(2.0) + b
            ax.plot(200, y_200, "*", color="#e6194b", markersize=14,
                    markeredgecolor="white", markeredgewidth=1, zorder=6)

            n_200_str = ""
            if n_train_100:
                n_200_str = f"\n({n_train_100 * 2} volumes)"
            ax.annotate(f"{y_200:.3f}{n_200_str}", (200, y_200),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, color="#e6194b",
                        fontweight="bold")

        ax.set_xlabel("Training Data (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        arrow = "(higher=better)" if higher_better else "(lower=better)"
        ax.set_title(f"{label} {arrow}", fontsize=12)
        ax.set_xlim(-5, 215)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # Add 200% tick with volume count
        xticks = [0, 25, 50, 75, 100, 150, 200]
        xticklabels = [str(x) for x in xticks]
        if n_train_100:
            xticklabels[-1] = f"200\n({n_train_100 * 2} vol)"
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=9)

    for idx in range(n_metrics, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Performance vs. Training Data Size", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
