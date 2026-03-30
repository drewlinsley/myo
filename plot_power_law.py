"""Plot metrics vs. training data fraction (power law curves).

Usage:
    python plot_power_law.py --results_dir results/power_law/ --output results/power_law/power_law.png
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("overall_masked_pearson", "Masked Pearson r", True),
    ("overall_masked_mse",     "Masked MSE",       False),
    ("overall_masked_mae",     "Masked MAE",       False),
    ("overall_ssim",           "SSIM",             True),
]


def main():
    parser = argparse.ArgumentParser(description="Plot power law curves")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing frac*.json result files")
    parser.add_argument("--output", default="power_law.png",
                        help="Output plot path")
    args = parser.parse_args()

    fractions = []
    all_metrics = {key: [] for key, _, _ in METRICS}

    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.endswith(".json"):
            continue
        m = re.match(r"frac(\d+)\.json", fname)
        if not m:
            continue
        frac_pct = int(m.group(1))
        frac = frac_pct / 100.0

        with open(os.path.join(args.results_dir, fname)) as f:
            data = json.load(f)

        fractions.append(frac)
        for key, _, _ in METRICS:
            val = data.get(key, float("nan"))
            all_metrics[key].append(val if val is not None else float("nan"))
        print(f"  {frac_pct}%: " + "  ".join(
            f"{label}={data.get(key, float('nan')):.4f}"
            for key, label, _ in METRICS))

    if len(fractions) < 2:
        print("Not enough data points to plot.")
        return

    fractions = np.array(fractions)
    order = np.argsort(fractions)
    fractions = fractions[order]
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])[order]

    active_metrics = [(k, l, h) for k, l, h in METRICS
                      if not np.all(np.isnan(all_metrics[k]))]

    n_metrics = len(active_metrics)
    cols = min(n_metrics, 2)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for idx, (key, label, higher_better) in enumerate(active_metrics):
        ax = axes[idx // cols, idx % cols]
        values = all_metrics[key]
        valid = ~np.isnan(values)

        ax.plot(fractions[valid] * 100, values[valid], "o-", color="#4363d8",
                markersize=8, linewidth=2, markeredgecolor="white", markeredgewidth=1.5)

        for frac, val in zip(fractions[valid], values[valid]):
            ax.annotate(f"{val:.3f}", (frac * 100, val),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9)

        ax.set_xlabel("Training Data (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        arrow = "(higher=better)" if higher_better else "(lower=better)"
        ax.set_title(f"{label} {arrow}", fontsize=12)
        ax.set_xlim(-2, 105)
        ax.grid(True, alpha=0.3)

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
