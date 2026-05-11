"""3-panel scaling plot:
   (1) LOO accuracy (mean ± SE over seeds) vs BF->GFP training fraction
   (2) MAE on held-out task vols vs fraction
   (3) SSIM on held-out task vols vs fraction

Panel 1 supports multiple CV granularities (volume / replicate / date).
Color = task. Linestyle = cv_unit. Legend is split into two sections.
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
from matplotlib.lines import Line2D


FRAC_RE = re.compile(r"frac(\d{3})")

CV_LINESTYLE = {"volume": "-", "replicate": "--", "date": ":"}
CV_MARKER = {"volume": "o", "replicate": "s", "date": "^"}
TASK_COLOR = {"exercise": "#3cb44b", "perturbation": "#e6194b"}


def parse_loo_json(path):
    with open(path) as f:
        d = json.load(f)
    task = d.get("task")
    acc = d.get("overall_accuracy")
    init_from = d.get("init_from")
    cv_unit = d.get("cv_unit", "volume")  # older runs default to volume
    n_groups = d.get("n_groups")
    if init_from:
        m = FRAC_RE.search(os.path.basename(os.path.dirname(init_from)))
        frac = int(m.group(1)) / 100.0 if m else None
    else:
        frac = 0.0
    return task, cv_unit, frac, acc, n_groups


def parse_metrics_json(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("holdout"), d.get("fraction"), d.get("mean", {})


def aggregate_loo(loo_dir):
    """Returns {(task, cv_unit): {frac: {"accs": [...], "n_groups": int}}}."""
    agg = defaultdict(lambda: defaultdict(lambda: {"accs": [], "n_groups": None}))
    for path in sorted(glob(os.path.join(loo_dir, "*.json"))):
        try:
            task, cv, frac, acc, n_groups = parse_loo_json(path)
        except Exception:
            continue
        if task is None or frac is None or acc is None:
            continue
        cell = agg[(task, cv)][frac]
        cell["accs"].append(acc)
        if n_groups is not None and cell["n_groups"] is None:
            cell["n_groups"] = n_groups
    return agg


def aggregate_metrics(metrics_dir):
    """Returns {task: {frac: {mae, ssim, pearson}}}."""
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    tasks = ["exercise", "perturbation"]

    csv_rows = [("task", "cv_unit", "n_groups", "fraction", "n_seeds",
                 "acc_mean", "acc_se", "mae", "ssim", "pearson")]

    # Panel 1: classification accuracy, one line per (task, cv_unit)
    ax = axes[0]
    seen_cvs = set()
    seen_tasks = set()
    for (task, cv), per_frac in sorted(loo.items()):
        if task not in tasks or cv not in CV_LINESTYLE:
            continue
        seen_cvs.add(cv)
        seen_tasks.add(task)
        fracs = sorted(per_frac.keys())
        means, ses, ns, n_groups = [], [], [], []
        for f in fracs:
            cell = per_frac[f]
            vals = np.array(cell["accs"], dtype=float)
            n = len(vals)
            means.append(float(vals.mean()))
            ses.append(float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
            ns.append(n)
            n_groups.append(cell["n_groups"])
        means_a = np.array(means); ses_a = np.array(ses)
        ax.plot(fracs, means_a, linestyle=CV_LINESTYLE[cv],
                marker=CV_MARKER[cv], color=TASK_COLOR[task], linewidth=2)
        if any(s > 0 for s in ses):
            ax.fill_between(fracs, means_a - ses_a, means_a + ses_a,
                            color=TASK_COLOR[task], alpha=0.15)
        # Annotate n_groups on the rightmost point
        if fracs and n_groups[-1] is not None:
            ax.annotate(f"n={n_groups[-1]}",
                        xy=(fracs[-1], means_a[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=TASK_COLOR[task])
        for f, m, se, n, ng in zip(fracs, means_a, ses_a, ns, n_groups):
            mm = metrics.get(task, {}).get(f, {})
            csv_rows.append((task, cv, ng, f, n, float(m), float(se),
                             mm.get("mae"), mm.get("ssim"), mm.get("pearson")))

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("BF→GFP training fraction")
    ax.set_ylabel("LOO accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Classification (mean ± SE)")
    ax.grid(True, alpha=0.3)

    # Two-section legend: tasks (colors), cv units (linestyles)
    task_handles = [Line2D([0], [0], color=TASK_COLOR[t], linewidth=2,
                           label=t.capitalize())
                    for t in tasks if t in seen_tasks]
    cv_handles = [Line2D([0], [0], color="black",
                         linestyle=CV_LINESTYLE[cv], marker=CV_MARKER[cv],
                         linewidth=2, label=cv)
                  for cv in ["volume", "replicate", "date"] if cv in seen_cvs]
    chance = Line2D([0], [0], color="gray", linestyle=":",
                    label="chance (binary)")
    legend1 = ax.legend(handles=task_handles + [chance],
                        loc="lower right", fontsize=8, title="Task")
    ax.add_artist(legend1)
    if cv_handles:
        ax.legend(handles=cv_handles, loc="upper left", fontsize=8,
                  title="CV unit")

    # Panel 2: MAE on held-out vols
    ax = axes[1]
    for task in tasks:
        m = metrics.get(task, {})
        if not m:
            continue
        fracs = sorted(m.keys())
        vals = [m[f].get("mae") for f in fracs]
        ax.plot(fracs, vals, "-o", color=TASK_COLOR[task],
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
        ax.plot(fracs, vals, "-o", color=TASK_COLOR[task],
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
