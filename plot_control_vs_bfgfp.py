"""Grouped bar chart: GFP control vs BF->GFP + classifier, per task.

Reads four JSONs written by train_loo_classifier.py:
  {results_dir}/control_exercise_gfp.json
  {results_dir}/control_perturbation_gfp.json
  {results_dir}/bfgfp_exercise_bf.json
  {results_dir}/bfgfp_perturbation_bf.json

Each JSON has "overall_accuracy" and an optional "permutation_test" block.
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/loo")
    p.add_argument("--output", default="results/classifier/control_vs_bfgfp.png")
    p.add_argument("--control_tag", default="control")
    p.add_argument("--bfgfp_tag", default="bfgfp")
    args = p.parse_args()

    tasks = ["exercise", "perturbation"]
    arms = [("control", args.control_tag, "gfp"),
            ("bfgfp",   args.bfgfp_tag,   "bf")]

    accs = {arm: [] for arm, _, _ in arms}
    ps = {arm: [] for arm, _, _ in arms}
    for task in tasks:
        for arm, tag, inp in arms:
            path = os.path.join(args.results_dir,
                                f"{tag}_{task}_{inp}.json")
            d = load(path)
            if d is None:
                print(f"warn: missing {path}")
                accs[arm].append(np.nan)
                ps[arm].append(None)
                continue
            accs[arm].append(d.get("overall_accuracy", np.nan))
            pt = d.get("permutation_test") or {}
            ps[arm].append(pt.get("p_value"))

    x = np.arange(len(tasks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"control": "#e6194b", "bfgfp": "#3cb44b"}
    labels = {"control": "GFP control (no BF->GFP)",
              "bfgfp":   "BF->GFP encoder + classifier"}
    for i, (arm, _, _) in enumerate(arms):
        xs = x + (i - 0.5) * width
        ys = accs[arm]
        bars = ax.bar(xs, ys, width, label=labels[arm], color=colors[arm])
        for rect, y, pv in zip(bars, ys, ps[arm]):
            if np.isnan(y):
                continue
            txt = f"{y:.2f}"
            if pv is not None:
                txt += f"\np={pv:.3f}"
            ax.text(rect.get_x() + rect.get_width() / 2,
                    y + 0.02, txt, ha="center", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
               label="chance (binary)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() + " (yes/no)" for t in tasks])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("LOO accuracy")
    ax.set_title("GFP control vs BF->GFP + classifier (binary, LOO)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
