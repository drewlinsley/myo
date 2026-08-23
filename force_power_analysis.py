"""How large an effect could this design have detected? (and how many tissues would it take)

A null force result is uninterpretable on its own. `scripts/dino_force_sweep.sh`
reporting "nothing survives correction" is consistent with two very different
worlds:

  (a) there is no relationship between the images and force, or
  (b) there is one, and 22 replicates cannot see it.

This script separates them. It simulates the ACTUAL analysis — the same
`run_loo` the probe uses, same replicate aggregation, same within-plate
deconfounding, same leave-one-replicate-out — on synthetic data with a KNOWN
planted effect, and reports the power to detect it at a range of sample sizes.

The planted effect is parameterized by `r`: the population correlation, at the
replicate level, between the best possible linear readout of the features and
force. r is therefore a CEILING — no model can beat it — which makes the output
read as "even a perfect model needs n >= X to see an effect of size r".

Two bars are reported, because the sweep applies both:
  single      one pre-registered config, alpha=0.05
  family-wise max-statistic over `--n_configs` correlated configs, which is what
              a sweep actually pays

Usage:
    python force_power_analysis.py                     # defaults, ~5 min
    python force_power_analysis.py --n_grid 22,40,60 --n_sim 100   # quicker
    python force_power_analysis.py --observed_rho 0.416 --n_obs 22
        (adds: what effect size would have been needed to explain that value)
"""

import os
import sys
import json
import argparse
import importlib.util

import numpy as np

_pf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probe_force_features.py")
_spec = importlib.util.spec_from_file_location("_probe", _pf_path)
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)


def make_dataset(rng, n_rep, r, n_plates=4, vols_per_rep=2.68, dim=768,
                 plate_sd=0.16, k_latent=20, decay=0.85, sig_idx=2,
                 vol_jitter=0.7, iso=0.05):
    """Synthesize one dataset with a planted replicate-level effect of size r.

    The feature model is LOW-RANK plus small isotropic noise, because that is
    what real DINOv2 features look like: a 768-d vector whose variation across
    59 volumes lives in a couple of dozen effective directions with a fast
    decaying spectrum. Planting the signal as a unit direction in isotropic
    768-d noise instead would make the simulation report ~0 power for ANY n --
    it would be measuring signal-to-noise in the ambient dimension rather than
    the effect of sample size, which is the question here.

    The signal occupies latent factor `sig_idx` (default 2, a mid-strength
    factor, not the dominant one). `vol_jitter` is per-FOV variation around the
    replicate's own latent score -- this is what makes averaging FOVs help, and
    what an unbalanced FOV count interacts with.

    Geometry mirrors the real drop: replicates nested in plates,
    vols_per_rep 2.68 = 59 volumes / 22 replicates. plate_sd 0.16 reproduces
    peak_amplitude_week1's measured plate structure (omega^2 ~ 0.026), i.e.
    almost none -- that target is not plate-confounded.
    """
    plate_of = [i % n_plates for i in range(n_rep)]
    plate_off = rng.normal(0, plate_sd, size=n_plates)

    y = rng.normal(size=n_rep) + plate_off[np.asarray(plate_of)]
    yz = (y - y.mean()) / (y.std() + 1e-12)

    # Replicate-level latent scores. Factor sig_idx is the only one tied to
    # force; corr(f_sig, y) = r at the replicate level, so r is a CEILING.
    F = rng.normal(size=(n_rep, k_latent))
    F[:, sig_idx] = r * yz + np.sqrt(max(0.0, 1 - r ** 2)) * rng.normal(size=n_rep)

    lam = np.sqrt(decay ** np.arange(k_latent))        # spectrum
    B = rng.normal(size=(k_latent, dim))
    B /= np.linalg.norm(B, axis=1, keepdims=True)      # unit loading directions

    vol_group, vol_force, dc, rows = [], [], [], []
    for g in range(n_rep):
        # Vary FOV count the way the real data does rather than fixing it: an
        # unbalanced design is part of what costs power here.
        k = max(1, int(rng.poisson(vols_per_rep - 1) + 1))
        for _ in range(k):
            f = F[g] + rng.normal(0, vol_jitter, size=k_latent)
            rows.append((f * lam) @ B + rng.normal(0, iso, size=dim))
            vol_group.append(f"r{g}")
            vol_force.append(y[g])
            dc.append(f"p{plate_of[g]}")
    return (np.asarray(rows), vol_group, np.asarray(vol_force), dc)


def rho_of(X, vol_group, vol_force, dc, pca_dim, alpha, seed, n_configs):
    """Observed spearman for each of n_configs correlated views of X.

    The configs are nested feature subsets, which is exactly how the real sweep
    is correlated (patch_mean_fg is a subset of patch_mean_fg+patch_std_fg).
    Returns the per-config values; the caller takes max() for the family bar.
    """
    d = X.shape[1]
    out = []
    for c in range(n_configs):
        # c=0 uses everything; later configs drop a growing tail.
        keep = d if c == 0 else max(8, int(d * (1.0 - 0.12 * c)))
        res = _pf.run_loo(X[:, :keep], vol_group, vol_force, vol_group,
                          "regression", 4, pca_dim, seed=seed,
                          fixed_alpha=alpha, deconfound=dc)
        out.append(_pf.spearman(res["true_force"], res["pred_score"]))
    return np.asarray([v if np.isfinite(v) else -1.0 for v in out])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_grid", default="22,30,40,60,100",
                   help="replicate counts to evaluate")
    p.add_argument("--r_grid", default="0.4,0.6,0.8",
                   help="planted ceiling correlations")
    p.add_argument("--n_sim", type=int, default=150,
                   help="datasets per cell (and for the null threshold)")
    p.add_argument("--n_configs", type=int, default=4,
                   help="correlated configs in the family-wise bar")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--pca_dim", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--observed_rho", type=float, default=None,
                   help="a rho you actually measured, to place on the curve")
    p.add_argument("--n_obs", type=int, default=22,
                   help="the n that --observed_rho came from")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    n_grid = [int(v) for v in args.n_grid.split(",") if v.strip()]
    r_grid = [float(v) for v in args.r_grid.split(",") if v.strip()]

    print("=" * 74)
    print(" Power to detect a force effect, simulating the real LOO analysis")
    print("=" * 74)
    print(f"  {args.n_sim} datasets/cell, dim={args.dim}, pca={args.pca_dim}, "
          f"deconfound=plate, {args.n_configs} configs in the family")
    print("  r = ceiling correlation between the best linear readout and force")
    print("      at the replicate level. No model can exceed it, so this is the")
    print("      most optimistic case, not a typical one.\n")

    out = {"args": vars(args), "cells": []}

    for n in n_grid:
        rng = np.random.default_rng(args.seed + n)

        # --- null: what does rho do when there is genuinely nothing? ---
        null_single, null_family = [], []
        for _ in range(args.n_sim):
            X, vg, vf, dc = make_dataset(rng, n, 0.0, dim=args.dim)
            v = rho_of(X, vg, vf, dc, args.pca_dim, args.alpha, args.seed,
                       args.n_configs)
            null_single.append(v[0])
            null_family.append(v.max())
        thr_s = float(np.percentile(null_single, 95))
        thr_f = float(np.percentile(null_family, 95))

        print(f"  n={n:>4} replicates   null rho: mean {np.mean(null_single):+.3f}"
              f"   detection bar: single {thr_s:+.3f}, "
              f"family-of-{args.n_configs} {thr_f:+.3f}")

        row = {"n": n, "null_mean": float(np.mean(null_single)),
               "threshold_single": thr_s, "threshold_family": thr_f,
               "power": {}}

        for r in r_grid:
            hit_s = hit_f = 0
            for _ in range(args.n_sim):
                X, vg, vf, dc = make_dataset(rng, n, r, dim=args.dim)
                v = rho_of(X, vg, vf, dc, args.pca_dim, args.alpha, args.seed,
                           args.n_configs)
                hit_s += v[0] > thr_s
                hit_f += v.max() > thr_f
            ps, pf_ = hit_s / args.n_sim, hit_f / args.n_sim
            row["power"][f"{r}"] = {"single": ps, "family": pf_}
            mark = "  <-- adequately powered" if pf_ >= 0.8 else ""
            print(f"            r={r:<4}  power: single {ps:5.1%}   "
                  f"family {pf_:5.1%}{mark}")
        out["cells"].append(row)
        print("")

    # --- place the value they actually measured ---
    if args.observed_rho is not None:
        cell = next((c for c in out["cells"] if c["n"] == args.n_obs), None)
        print("-" * 74)
        print(f"  Your measured best rho = {args.observed_rho:+.3f} at "
              f"n={args.n_obs}")
        if cell:
            print(f"    single-config bar at that n: {cell['threshold_single']:+.3f}")
            print(f"    family-wise bar at that n:   {cell['threshold_family']:+.3f}")
            verdict = ("BELOW both bars — indistinguishable from noise"
                       if args.observed_rho <= cell["threshold_single"]
                       else "above the single bar; check it against the family bar")
            print(f"    -> {verdict}")
        else:
            print(f"    (n={args.n_obs} not in --n_grid; add it to compare)")

    print("\n  how to read this")
    print("    The bar is set by n, not by the model. If power at your n is low")
    print("    even for a large r, then a null result says nothing about whether")
    print("    force is decodable — only that this experiment could not tell.")
    print("    In that case more tissues is the only fix; more architectures,")
    print("    more tokens, and more configs all make the family bar HIGHER.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  saved {args.output}")


if __name__ == "__main__":
    main()
