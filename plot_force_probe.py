#!/usr/bin/env python
"""Plot what the DINOv2 -> force probe actually did.

Reads a directory of `probe_force_features.py` result JSONs (one per swept
config, as written by scripts/dino_force_sweep.sh) and renders one figure.

The figure is built around a specific failure mode. The headline number people
remember from this sweep is "spearman ~ 0.4", and 0.4 sounds like a result. It
is not, at n=22 replicates: the permutation null for these configs spans
roughly [-0.55, +0.63], so the smallest correlation this design could tell
apart from chance is about +0.54. A scatter plot alone would show a convincing
upward trend and say nothing about that. Every panel here therefore carries its
own null:

  A  predicted vs true, colored by plate   -- the effect, as an effect looks
  B  the permutation null for that config  -- what chance looks like
  C  every config, observed vs its null    -- the whole sweep at once
  D  the design facts and the family-wise p

Panel C is the one to put in front of a team. It shows the observed statistic
as a dot inside the interval chance would have produced, config by config, so
"we got 0.4" and "0.4 is inside the noise" appear in the same picture.

Usage
  python plot_force_probe.py --results_dir results/dino_sweep/<target>_... \
      --out results/figures
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --------------------------------------------------------------------------
def load_results(results_dir):
    """Every config JSON in a results dir, minus the controls and inventory."""
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(p)
        if base in ("metadata_inventory.json", "manifest.json"):
            continue
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if "spearman_pred_vs_force" not in d and "replicate_accuracy" not in d:
            continue
        d["_name"] = os.path.splitext(base)[0]
        d["_path"] = p
        out.append(d)
    return out


def split_controls(results):
    """Separate the shuffled-label canary from the real configs.

    The canary must NOT be ranked alongside the configs or fed to the
    family-wise correction: it is a different hypothesis (does the fold logic
    leak?) and including it in a max-statistic would let a leaking control
    raise the correction threshold for everyone else.
    """
    real, ctrl = [], []
    for d in results:
        (ctrl if d.get("shuffled") or d.get("canary") not in (None, "none")
         else real).append(d)
    return real, ctrl


def stat_keys(results):
    """Rank by rho for a numeric target, by accuracy for a categorical one."""
    categorical = any(d.get("target_type") == "categorical" for d in results)
    if categorical:
        return ("replicate_accuracy", "null_accuracy", "permutation_p_accuracy",
                "mde_accuracy_95", "accuracy")
    return ("spearman_pred_vs_force", "null_spearman", "permutation_p_spearman",
            "mde_spearman_95", "Spearman rho")


def family_wise_p(results, obs_key, null_key):
    """Max-statistic family-wise p across configs with matched nulls.

    All configs permute the SAME replicate labels with the SAME seed, so their
    null distributions are aligned permutation-by-permutation. The correction
    is then: for each permutation take the best statistic across configs, and
    rank the observed best against that distribution. This accounts for both
    the number of configs and the fact that they are highly correlated, which
    a Bonferroni over 8 near-duplicate configs would badly over-penalize.

    Returns (p, n_used, n_perm, obs_best, max_null) or None if the nulls are
    missing or ragged.
    """
    nulls, obs = [], []
    for d in results:
        n = d.get(null_key)
        o = d.get(obs_key)
        if not n or o is None or not np.isfinite(o):
            continue
        nulls.append(np.asarray(n, dtype=float))
        obs.append(float(o))
    if not nulls:
        return None
    k = min(len(n) for n in nulls)
    if k < 20:
        return None
    M = np.stack([n[:k] for n in nulls])          # (n_cfg, n_perm)
    max_null = M.max(axis=0)
    obs_best = max(obs)
    p = float((1 + (max_null >= obs_best).sum()) / (1 + len(max_null)))
    return p, len(obs), k, obs_best, max_null


# --------------------------------------------------------------------------
def short_plate_labels(plates):
    """Shorten plate ids to the fields that actually differ.

    Naive shortening (last dash-field) collapsed 260420-TC010-PL051-P14 and
    260426-TC010-PL051-P14 to the same "P14", so the legend showed two
    different plates under one name. Split on "-" and keep only the positions
    that vary across the plates present.
    """
    parts = [str(p).split("-") for p in plates]
    if len({len(x) for x in parts}) != 1:
        return {p: str(p) for p in plates}
    n = len(parts[0])
    keep = [i for i in range(n) if len({x[i] for x in parts}) > 1]
    if not keep:
        return {p: str(p) for p in plates}
    return {p: "-".join(x[i] for i in keep) for p, x in zip(plates, parts)}


def panel_scatter(ax, best, label, standalone=False, obs_key=None,
                  p_key=None, mde_key=None):
    """Held-out prediction vs measurement, one point per replicate.

    A replicate is one (plate, Tissue) pair -- one tissue. Its R001/R002/R003
    fields of view share a single force value, so they are not independent
    units and are averaged into the one point drawn here. Point area is
    proportional to how many FOVs went into it.
    """
    pr = best.get("per_replicate") or []
    if not pr:
        ax.text(0.5, 0.5, "no per-replicate predictions saved\n"
                          "(re-run the probe after this change)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    t = np.array([r["true_force"] for r in pr])
    s = np.array([r["pred_score"] for r in pr])
    nfov = np.array([int(r.get("n_fov", 0) or 0) for r in pr])
    sizes = (48 + 34 * (nfov - 1)) if nfov.max() > 0 else np.full(len(t), 48)
    plates = [str(r.get("plate", "?")) for r in pr]
    uniq = sorted(set(plates))
    short = short_plate_labels(uniq)
    cmap = plt.get_cmap("tab10")
    for i, u in enumerate(uniq):
        m = np.array([p == u for p in plates])
        ax.scatter(t[m], s[m], s=sizes[m], color=cmap(i % 10), edgecolor="k",
                   linewidth=0.5, label=short[u], zorder=3,
                   alpha=0.9 if standalone else 1.0)
    if len(t) > 2 and np.ptp(t) > 0:
        # Fit to ALL points, so it is descriptive only. The held-out evidence
        # is the rank correlation and its permutation p, not this line.
        b, a = np.polyfit(t, s, 1)
        xs = np.linspace(t.min(), t.max(), 2)
        ax.plot(xs, a + b * xs, "k--", lw=1.2, zorder=2,
                label="least squares (descriptive)" if standalone else None)

    dc = best.get("deconfound", "none")
    unit = ("within-plate centered" if dc == "plate" else "raw")
    ax.set_xlabel(f"measured {best.get('target_col','force')}  ({unit})")
    ax.set_ylabel("predicted, leave-one-replicate-out")
    ax.grid(alpha=0.25)
    ax.legend(title="plate (batch)", fontsize=7.5, title_fontsize=7.5,
              loc="best", framealpha=0.9)

    if not standalone:
        ax.set_title(f"A. one point per replicate (n={len(t)})", loc="left",
                     fontsize=10, fontweight="bold")
        return

    obs, pp = best.get(obs_key), best.get(p_key)
    mde = best.get(mde_key)
    box = [f"{label:<17} {_fmt(obs)}",
           f"{'p (permutation)':<17} {_fmt(pp)}",
           f"{'detectable only >':<17} {_fmt(mde)}"]
    ax.text(0.02, 0.98, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6",
                      alpha=0.92))
    nfovs = f"{nfov.min()}-{nfov.max()}" if nfov.max() else "?"
    ax.set_title(
        f"Held-out force prediction: one point per replicate\n"
        f"{len(t)} tissues on {best.get('n_plates','?')} plates  -  "
        f"{best.get('n_volumes','?')} FOV volumes ({nfovs} per tissue, "
        f"averaged)  -  point area proportional to FOVs",
        fontsize=10.5, fontweight="bold", loc="left")


def panel_null(ax, best, obs_key, null_key, p_key, mde_key, label):
    null = np.asarray(best.get(null_key) or [], dtype=float)
    obs = best.get(obs_key)
    if null.size == 0:
        ax.text(0.5, 0.5, "no permutation null saved", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.hist(null, bins=40, color="0.75", edgecolor="none",
            label=f"chance ({null.size} permutations)")
    mde = best.get(mde_key)
    if mde is not None:
        ax.axvline(mde, color="tab:orange", ls="--", lw=1.5,
                   label=f"detection threshold {mde:+.2f}")
    if obs is not None and np.isfinite(obs):
        ax.axvline(obs, color="tab:red", lw=2.5,
                   label=f"observed {obs:+.3f}")
    pp = best.get(p_key)
    ax.set_xlabel(label)
    ax.set_ylabel("permutations")
    ax.set_title("B. the same number, against chance"
                 + (f"   (p = {pp:.3f})" if pp is not None else ""),
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")


def panel_forest(ax, results, obs_key, null_key, mde_key, label):
    rows = []
    for d in results:
        o = d.get(obs_key)
        n = np.asarray(d.get(null_key) or [], dtype=float)
        if o is None or not np.isfinite(o):
            continue
        lo, hi = ((np.percentile(n, 2.5), np.percentile(n, 97.5))
                  if n.size else (np.nan, np.nan))
        rows.append((d["_name"], float(o), lo, hi, d.get(mde_key)))
    if not rows:
        ax.set_axis_off()
        return
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    for i, (_nm, o, lo, hi, mde) in enumerate(rows):
        if np.isfinite(lo):
            ax.plot([lo, hi], [i, i], color="0.75", lw=6, solid_capstyle="butt",
                    zorder=1)
        sig = mde is not None and o > mde
        ax.scatter([o], [i], s=55, zorder=3,
                   color="tab:red" if sig else "tab:blue",
                   edgecolor="k", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("gfp_tiled_", "") for r in rows],
                       fontsize=6.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(label)
    ax.set_title("C. every config, each against its own null", loc="left",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(handles=[
        Patch(facecolor="0.75", label="95% of chance outcomes"),
        Patch(facecolor="tab:blue", label="observed (inside chance)"),
        Patch(facecolor="tab:red", label="observed (clears its own threshold)"),
    ], fontsize=6.5, loc="lower right", framealpha=0.9)


def panel_facts(ax, best, results, ctrl, fw, obs_key, p_key, mde_key, label):
    ax.set_axis_off()
    L = []
    L.append(f"target      {best.get('target_col','?')}"
             f"  ({best.get('target_type','numeric')})")
    L.append(f"units       {best.get('n_replicates','?')} replicates"
             f" on {best.get('n_plates','?')} plates"
             f"  ({best.get('n_volumes','?')} volumes)")
    L.append(f"features    frozen DINOv2, {best.get('feature_dim','?')}-d"
             f"  ({best.get('modality','?')}, {best.get('model_class','?')})")
    L.append(f"confound    deconfound={best.get('deconfound','none')}"
             f"   eta2_plate={_fmt(best.get('eta2_plate'))}")
    L.append("")
    L.append(f"best config {best['_name']}")
    L.append(f"  {label:<14}{_fmt(best.get(obs_key))}")
    L.append(f"  p (this config alone)  {_fmt(best.get(p_key))}")
    L.append(f"  detectable only above  {_fmt(best.get(mde_key))}")
    fc = best.get("readout_fold_cosine")
    if fc is not None:
        L.append(f"  readout stability      {fc:+.3f} mean fold cosine"
                 + ("" if fc >= 0.9 else "   <-- unstable"))
    L.append("")
    if fw:
        p, n_used, n_perm, obs_best, _ = fw
        L.append(f"FAMILY-WISE over {n_used} configs ({n_perm} matched perms)")
        L.append(f"  best observed {obs_best:+.3f}   corrected p = {p:.4f}")
        L.append("  " + ("SURVIVES correction" if p < 0.05 else
                         "does NOT survive correction"))
    else:
        L.append("FAMILY-WISE p unavailable (no matched nulls on disk)")
    L.append("")
    for d in ctrl:
        o = d.get(obs_key)
        L.append(f"control  {d['_name']}: {label} = {_fmt(o)}"
                 f"  (shuffled labels; must be at chance)")
    if not ctrl:
        L.append("control  NONE FOUND -- the leak canary did not run, so a")
        L.append("         fold-logic leak would be invisible here.")
    ax.text(0.0, 1.0, "\n".join(L), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=7.6)
    ax.set_title("D. what this design could and could not see", loc="left",
                 fontsize=10, fontweight="bold")


def _fmt(v):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):+.3f}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="results/dino_sweep/<target>_<task>_... directory")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--tag", default=None,
                    help="filename stem (default: the results dir name)")
    ap.add_argument("--panels", choices=["combined", "split", "both"],
                    default="both",
                    help="'split' also writes each panel as its own figure")
    ap.add_argument("--select", default=None,
                    help="plot this config in panels A/B instead of the "
                         "top-ranked one (substring match on the file stem)")
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"no probe result JSONs in {args.results_dir}")
    real, ctrl = split_controls(results)
    if not real:
        raise SystemExit(f"{args.results_dir} contains only control runs")
    obs_key, null_key, p_key, mde_key, label = stat_keys(real)

    if args.select:
        cand = [d for d in real if args.select in d["_name"]]
        if not cand:
            raise SystemExit(f"--select {args.select!r} matched no config; "
                             f"have: {', '.join(d['_name'] for d in real)}")
        best = cand[0]
    else:
        # Rank on the observed statistic. This is selection on the test
        # statistic, which is exactly why panel D reports a family-wise p
        # rather than this config's own p.
        best = max(real, key=lambda d: (d.get(obs_key) if d.get(obs_key)
                                        is not None else -np.inf))

    fw = family_wise_p(real, obs_key, null_key)

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.normpath(args.results_dir))
    tgt = best.get("target_col", "force")
    verdict = ("NOT distinguishable from chance"
               if not fw or fw[0] >= 0.05 else "survives family-wise correction")

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fp = os.path.join(args.out, f"{name}.{ext}")
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  saved {fp}")
        plt.close(fig)

    if args.panels in ("combined", "both"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
        panel_scatter(axes[0][0], best, label)
        panel_null(axes[0][1], best, obs_key, null_key, p_key, mde_key, label)
        panel_forest(axes[1][0], real, obs_key, null_key, mde_key, label)
        panel_facts(axes[1][1], best, real, ctrl, fw, obs_key, p_key, mde_key,
                    label)
        fig.suptitle(f"Frozen DINOv2 -> {tgt}:  {verdict}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        _save(fig, f"probe_{tag}")

    if args.panels in ("split", "both"):
        # Each panel standalone, for slides. The scatter gets its own
        # treatment rather than being the combined panel blown up: on its own
        # it has to carry the sample structure and the significance that
        # panels B-D would otherwise supply, or it reads as a clean result.
        f1, a1 = plt.subplots(figsize=(7.6, 6.4))
        panel_scatter(a1, best, label, standalone=True, obs_key=obs_key,
                      p_key=p_key, mde_key=mde_key)
        _save(f1, f"probe_{tag}_scatter")

        f2, a2 = plt.subplots(figsize=(7.6, 5.0))
        panel_null(a2, best, obs_key, null_key, p_key, mde_key, label)
        a2.set_title(f"{tgt}: observed vs the permutation null", loc="left",
                     fontsize=10.5, fontweight="bold")
        _save(f2, f"probe_{tag}_null")

        n_cfg = sum(1 for d in real if d.get(obs_key) is not None)
        f3, a3 = plt.subplots(figsize=(8.6, 1.1 + 0.5 * max(3, n_cfg)))
        panel_forest(a3, real, obs_key, null_key, mde_key, label)
        a3.set_title(f"{tgt}: every config against its own null", loc="left",
                     fontsize=10.5, fontweight="bold")
        _save(f3, f"probe_{tag}_configs")

        f4, a4 = plt.subplots(figsize=(8.0, 5.2))
        panel_facts(a4, best, real, ctrl, fw, obs_key, p_key, mde_key, label)
        a4.set_title(f"{tgt}: what this design could and could not see",
                     loc="left", fontsize=10.5, fontweight="bold")
        _save(f4, f"probe_{tag}_design")


if __name__ == "__main__":
    main()
