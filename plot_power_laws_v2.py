"""3-panel scaling plot:
   (1) LOO accuracy (mean ± SE over seeds) vs BF->GFP training fraction
   (2) MAE on held-out task vols vs fraction
   (3) SSIM on held-out task vols vs fraction
"""

import os
import re
import json
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAC_RE = re.compile(r"frac(\d{3})")


def parse_loo_json(path):
    with open(path) as f:
        d = json.load(f)
    task = d.get("task")
    acc = d.get("overall_accuracy")
    init_from = d.get("init_from")
    if init_from:
        m = FRAC_RE.search(os.path.basename(os.path.dirname(init_from)))
        frac = int(m.group(1)) / 100.0 if m else None
    else:
        frac = 0.0
    return task, frac, acc


def parse_metrics_json(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("holdout"), d.get("fraction"), d.get("mean", {})


def aggregate_loo(loo_dir):
    """Returns {task: {frac: [acc_seed0, acc_seed1, ...]}}."""
    agg = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob(os.path.join(loo_dir, "*.json"))):
        try:
            task, frac, acc = parse_loo_json(path)
        except Exception:
            continue
        if task is None or frac is None or acc is None:
            continue
        agg[task][frac].append(acc)
    return agg


def aggregate_metrics(metrics_dir):
    """Returns {task: {frac: {mae, ssim, pearson}}}.

    `holdout` maps to task: 'exercise' -> exercise; 'perturbation' -> perturbation.
    """
    out = {"exercise": {}, "perturbation": {}}
    for path in sorted(glob(os.path.join(metrics_dir, "*.json"))):
        holdout, frac, means = parse_metrics_json(path)
        if holdout in out and frac is not None:
            out[holdout][float(frac)] = means
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loo_dir", default="results/loo")
    p.add_argument("--metrics_dir", default="results/bfgfp_metrics")
    p.add_argument("--output", default="results/classifier/power_laws_v2.png")
    p.add_argument("--csv", default=None,
                   help="Optional CSV path for aggregated values")
    args = p.parse_args()

    loo = aggregate_loo(args.loo_dir)
    metrics = aggregate_metrics(args.metrics_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"exercise": "#3cb44b", "perturbation": "#e6194b"}
    tasks = ["exercise", "perturbation"]

    csv_rows = [("task", "fraction", "n_seeds", "acc_mean", "acc_se",
                 "mae", "ssim", "pearson")]

    # Panel 1: classification accuracy
    ax = axes[0]
    for task in tasks:
        if task not in loo:
            continue
        fracs = sorted(loo[task].keys())
        means, ses, ns = [], [], []
        for f in fracs:
            vals = np.array(loo[task][f], dtype=float)
            n = len(vals)
            means.append(float(vals.mean()))
            ses.append(float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
            ns.append(n)
        means = np.array(means); ses = np.array(ses)
        ax.plot(fracs, means, "-o", color=colors[task],
                label=f"{task.capitalize()} (n_seeds≤{max(ns) if ns else 0})",
                linewidth=2)
        ax.fill_between(fracs, means - ses, means + ses,
                        color=colors[task], alpha=0.2)
        for f, m, se, n in zip(fracs, means, ses, ns):
            mm = metrics.get(task, {}).get(f, {})
            csv_rows.append((task, f, n, float(m), float(se),
                             mm.get("mae"), mm.get("ssim"), mm.get("pearson")))
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
               label="chance (binary)")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("BF→GFP training fraction")
    ax.set_ylabel("LOO accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification (mean ± SE)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: MAE on held-out vols (one point per (task, frac))
    ax = axes[1]
    for task in tasks:
        m = metrics.get(task, {})
        if not m:
            continue
        fracs = sorted(m.keys())
        vals = [m[f].get("mae") for f in fracs]
        ax.plot(fracs, vals, "-o", color=colors[task],
                label=task.capitalize(), linewidth=2)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("BF→GFP training fraction")
    ax.set_ylabel("MAE (lower = better)")
    ax.set_title("Regression MAE on held-out task vols")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: SSIM
    ax = axes[2]
    for task in tasks:
        m = metrics.get(task, {})
        if not m:
            continue
        fracs = sorted(m.keys())
        vals = [m[f].get("ssim") for f in fracs]
        ax.plot(fracs, vals, "-o", color=colors[task],
                label=task.capitalize(), linewidth=2)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("BF→GFP training fraction")
    ax.set_ylabel("SSIM (higher = better)")
    ax.set_title("Regression SSIM on held-out task vols")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

    csv_path = args.csv or (os.path.splitext(args.output)[0] + ".csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
