"""Plot scaling curves for the two-head GFP classifier.

Reads ckpts/gfp_classifier_frac*/best_metrics.json and plots
val accuracy (Exercise, Perturbation) vs training fraction.
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_glob", default="ckpts/gfp_classifier_frac*")
    p.add_argument("--output", default="results/gfp_classifier_scaling.png")
    args = p.parse_args()

    rows = []
    for d in sorted(glob(args.ckpt_glob)):
        mp = os.path.join(d, "best_metrics.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        rows.append(m)

    if not rows:
        raise SystemExit(f"No metrics found under {args.ckpt_glob}")

    rows.sort(key=lambda r: r["fraction"])
    fracs = [r["fraction"] for r in rows]
    acc_ex = [r["val_acc_exercise"] for r in rows]
    acc_pt = [r["val_acc_perturbation"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, acc_ex, "-o", label="Exercise", color="#3cb44b", linewidth=2)
    ax.plot(fracs, acc_pt, "-s", label="Perturbation", color="#e6194b", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="chance (binary)")
    ax.set_xscale("log")
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Val accuracy (best epoch)")
    ax.set_title("GFP classifier scaling")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
