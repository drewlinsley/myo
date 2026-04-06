"""Plot leave-one-out k-NN accuracy vs. segmentation training-data size.

For each task (Exercise, Perturbation), shows one curve per k value
against the seg-data fraction used to train the feature extractor.

Usage:
    python plot_classification_scaling.py \
        --results_dir results/classifier/ \
        --output results/classifier/classification_scaling.png
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="classification_scaling.png")
    args = parser.parse_args()

    # Load all frac{XXX}.json files
    data = {}  # seg_pct → {task_name → task_data}
    for fname in sorted(os.listdir(args.results_dir)):
        m = re.match(r"frac(\d+)\.json", fname)
        if not m:
            continue
        seg_pct = int(m.group(1))
        with open(os.path.join(args.results_dir, fname)) as f:
            d = json.load(f)
        data[seg_pct] = d.get("tasks", {})

    if not data:
        print("No results found.")
        return

    # Union of all tasks + k values seen
    all_tasks = sorted({task for v in data.values() for task in v.keys()})
    all_k_values = set()
    for tasks in data.values():
        for task_data in tasks.values():
            for k_str in task_data.get("per_k", {}).keys():
                all_k_values.add(int(k_str))
    all_k_values = sorted(all_k_values)

    if not all_tasks or not all_k_values:
        print("No task data to plot.")
        return

    seg_pcts_sorted = sorted(data.keys())

    # Color per k value
    cmap = plt.get_cmap("viridis")
    denom = max(len(all_k_values) - 1, 1)
    colors = [cmap(0.05 + 0.85 * i / denom) for i in range(len(all_k_values))]

    # One subplot per task
    n_tasks = len(all_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5.5),
                             squeeze=False)
    axes = axes[0]

    for ax_idx, task in enumerate(all_tasks):
        ax = axes[ax_idx]

        for k_idx, k_val in enumerate(all_k_values):
            x_vals, y_vals = [], []
            key = str(k_val)

            for seg_pct in seg_pcts_sorted:
                task_data = data[seg_pct].get(task, {})
                per_k = task_data.get("per_k", {})
                k_data = per_k.get(key)
                if k_data is None:
                    continue
                x_vals.append(seg_pct)
                y_vals.append(k_data["accuracy"])

            if not x_vals:
                continue

            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)

            ax.plot(x_arr, y_arr, "o-", color=colors[k_idx],
                    markersize=8, linewidth=2, markeredgecolor="white",
                    markeredgewidth=1.2,
                    label=f"k={k_val}")

            # Annotate each point with its accuracy
            for x, y in zip(x_arr, y_arr):
                ax.annotate(f"{y:.2f}", (x, y),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color=colors[k_idx])

        # Find class info for title (any seg_pct that has this task)
        n_samples, n_classes = None, None
        for seg_pct in seg_pcts_sorted:
            td = data[seg_pct].get(task)
            if td:
                n_samples = td.get("n_samples")
                n_classes = td.get("n_classes")
                break

        title = f"Task: {task.capitalize()}"
        if n_samples is not None:
            title += f"  (n={n_samples}, {n_classes} classes)"

        ax.set_xlabel("Segmentation Training Data (%)", fontsize=11)
        ax.set_ylabel("Leave-One-Out k-NN Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Chance line
        if n_classes and n_classes > 1:
            ax.axhline(1.0 / n_classes, color="#999999", linestyle=":",
                       linewidth=1, alpha=0.7, label="chance")

        ax.legend(title="k-NN", fontsize=9, loc="best", title_fontsize=9)

    fig.suptitle(
        "k-NN Classification vs. Segmentation Training Size",
        fontsize=14, y=1.02)
    fig.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
