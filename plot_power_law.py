"""Plot masked Pearson vs. training data fraction (power law curve).

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


def main():
    parser = argparse.ArgumentParser(description="Plot power law curve")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing frac*.json result files")
    parser.add_argument("--output", default="power_law.png",
                        help="Output plot path")
    args = parser.parse_args()

    # Collect results
    fractions = []
    pearsons = []

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

        r = data["overall_masked_pearson"]
        if r is None or (isinstance(r, float) and np.isnan(r)):
            print(f"  Skipping {fname}: NaN result")
            continue

        fractions.append(frac)
        pearsons.append(r)
        print(f"  {frac_pct}%: masked Pearson = {r:.4f}")

    if len(fractions) < 2:
        print("Not enough data points to plot.")
        return

    fractions = np.array(fractions)
    pearsons = np.array(pearsons)

    # Sort by fraction
    order = np.argsort(fractions)
    fractions = fractions[order]
    pearsons = pearsons[order]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions * 100, pearsons, "o-", color="#4363d8", markersize=8,
            linewidth=2, markeredgecolor="white", markeredgewidth=1.5)

    for frac, r in zip(fractions, pearsons):
        ax.annotate(f"{r:.3f}", (frac * 100, r),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xlabel("Training Data (%)", fontsize=12)
    ax.set_ylabel("Masked Pearson Correlation", fontsize=12)
    ax.set_title("Performance vs. Training Data Size", fontsize=13)
    ax.set_xlim(-2, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
