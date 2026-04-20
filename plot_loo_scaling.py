"""Plot LOO classification accuracy vs BF->GFP training fraction.

Reads results/loo/frac{XXX}_{task}_{input}.json files; plots one line per task.
"""

import os
import re
import json
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAC_RE = re.compile(r"frac(\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/loo")
    p.add_argument("--input", default="bf")
    p.add_argument("--output", default="results/loo/loo_scaling.png")
    args = p.parse_args()

    # Collect accuracy per (task, fraction) for the requested input
    rows = {"exercise": [], "perturbation": []}
    for path in sorted(glob(os.path.join(args.results_dir, f"*_{args.input}.json"))):
        fname = os.path.basename(path)
        m = FRAC_RE.search(fname)
        if not m:
            continue
        frac = int(m.group(1)) / 100.0
        with open(path) as f:
            d = json.load(f)
        task = d.get("task")
        if task in rows:
            rows[task].append((frac, d["overall_accuracy"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"exercise": "#3cb44b", "perturbation": "#e6194b"}
    for task, data in rows.items():
        if not data:
            continue
        data.sort()
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.plot(xs, ys, "-o", label=task.capitalize(),
                color=colors[task], linewidth=2)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
               label="chance (binary)")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("BF→GFP training fraction (encoder pretraining)")
    ax.set_ylabel("LOO accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"LOO scaling (input={args.input})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
