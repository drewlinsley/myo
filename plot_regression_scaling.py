"""Scaling-law plots for LOO regression (Pearson + MSE in separate figures).

Reads `results/loo_reg/*.json` files written by `train_loo_regression.py`,
groups by (target_col, cv_unit), and plots mean ± SE over seeds vs BF→GFP
training fraction. One figure per metric.
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
CV_LINESTYLE = {"volume": "-", "replicate": "--"}
CV_MARKER = {"volume": "o", "replicate": "s"}


def parse_reg_json(path):
    with open(path) as f:
        d = json.load(f)
    init_from = d.get("init_from")
    if init_from:
        m = FRAC_RE.search(os.path.basename(os.path.dirname(init_from)))
        frac = int(m.group(1)) / 100.0 if m else None
    else:
        frac = 0.0
    return (d.get("target_col"),
            d.get("cv_unit", "volume"),
            frac,
            d.get("metrics", {}),
            d.get("n_volumes"))


def aggregate(loo_dir):
    """{(target, cv): {frac: {pearson: [...], mse: [...], rmse: [...], n: int}}}.

    `rmse` falls back to sqrt(mse) for older JSONs that don't store it.
    """
    agg = defaultdict(lambda: defaultdict(
        lambda: {"pearson": [], "mse": [], "rmse": [], "mae": [], "n": None}))
    for path in sorted(glob(os.path.join(loo_dir, "*.json"))):
        try:
            target, cv, frac, metrics, n = parse_reg_json(path)
        except Exception:
            continue
        if target is None or frac is None or not metrics:
            continue
        cell = agg[(target, cv)][frac]
        for k in ("pearson", "mse", "mae", "rmse"):
            if metrics.get(k) is not None:
                cell[k].append(float(metrics[k]))
        # Back-fill RMSE from MSE if missing in older JSONs.
        if metrics.get("rmse") is None and metrics.get("mse") is not None:
            cell["rmse"].append(float(np.sqrt(metrics["mse"])))
        if n is not None and cell["n"] is None:
            cell["n"] = n
    return agg


def render_panel(ax, agg, metric, ylabel, title, lower_is_better):
    for (target, cv), per_frac in sorted(agg.items()):
        if cv not in CV_LINESTYLE:
            continue
        fracs = sorted(per_frac.keys())
        means, ses = [], []
        for f in fracs:
            vals = np.array(per_frac[f][metric], dtype=float)
            if vals.size == 0:
                means.append(np.nan); ses.append(0.0); continue
            means.append(float(vals.mean()))
            ses.append(float(vals.std(ddof=1) / np.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0)
        means_a = np.array(means); ses_a = np.array(ses)
        ax.plot(fracs, means_a, linestyle=CV_LINESTYLE[cv],
                marker=CV_MARKER[cv], color="#4363d8", linewidth=2,
                label=f"{target} ({cv})")
        if any(s > 0 for s in ses):
            ax.fill_between(fracs, means_a - ses_a, means_a + ses_a,
                            color="#4363d8", alpha=0.15)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("BF→GFP training fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right" if not lower_is_better else "upper right",
              fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loo_dir", default="results/loo_reg")
    p.add_argument("--output_dir", default="results/classifier")
    p.add_argument("--prefix", default="regression",
                   help="Filename prefix; outputs <prefix>_pearson.png and _mse.png")
    args = p.parse_args()

    agg = aggregate(args.loo_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pearson
    fig, ax = plt.subplots(figsize=(7.5, 5))
    render_panel(ax, agg, metric="pearson",
                 ylabel="Pearson r (held-out)",
                 title="LOO regression: Pearson vs BF→GFP fraction",
                 lower_is_better=False)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    pearson_path = os.path.join(args.output_dir, f"{args.prefix}_pearson.png")
    fig.savefig(pearson_path, dpi=150)
    print(f"Saved {pearson_path}")

    # MSE
    fig, ax = plt.subplots(figsize=(7.5, 5))
    render_panel(ax, agg, metric="mse",
                 ylabel="MSE (held-out, target units²)",
                 title="LOO regression: MSE vs BF→GFP fraction",
                 lower_is_better=True)
    fig.tight_layout()
    mse_path = os.path.join(args.output_dir, f"{args.prefix}_mse.png")
    fig.savefig(mse_path, dpi=150)
    print(f"Saved {mse_path}")

    # RMSE (raw target units — interpretable "off by N units")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    render_panel(ax, agg, metric="rmse",
                 ylabel="RMSE (held-out, target units)",
                 title="LOO regression: RMSE vs BF→GFP fraction",
                 lower_is_better=True)
    fig.tight_layout()
    rmse_path = os.path.join(args.output_dir, f"{args.prefix}_rmse.png")
    fig.savefig(rmse_path, dpi=150)
    print(f"Saved {rmse_path}")


if __name__ == "__main__":
    main()
