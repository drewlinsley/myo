"""Is there anything along z (or across the field) for a 3D aggregator to use?

`probe_force_features.py` collapses a volume's 144 view embeddings with a
weighted MEAN, which throws away where each view came from. Before building a
3D convnet over the (z, row, col) grid -- thousands of parameters against 22
labels -- it is worth knowing whether the grid axes carry any variance at all.

Optical sections 3 apart in a 35-slice band can be near-duplicates. If the
embedding barely changes with depth, a 3D aggregator has no substrate along the
axis that motivated it, and the honest move is to say so rather than to fit it.

Decomposes each volume's view-to-view variance into:
    across-z      same tile, different depth
    across-tile   same depth, different position in the field
    residual      interaction / everything else

Reads the cached .npz features, so it costs seconds and needs no GPU.

Usage:
    python diagnose_view_structure.py results/dino_features/gfp_tiled_volume_87a5619c
    python diagnose_view_structure.py <dir> --token patch_mean_fg --fg_min 0.75
"""

import os
import sys
import json
import glob
import argparse

import numpy as np


def decompose(v, vz, vy, vx, w):
    """Weighted variance of view embeddings split by grid axis.

    Returns fractions of total weighted view-to-view variance attributable to
    depth and to field position. Uses a two-way means decomposition: the
    variance of the per-z means (pooled over tiles) and of the per-tile means
    (pooled over z), both relative to the grand mean.
    """
    w = np.clip(np.asarray(w, float), 1e-12, None)
    w = w / w.sum()
    mu = (v * w[:, None]).sum(0)
    tot = float((w[:, None] * (v - mu) ** 2).sum())
    if tot <= 0:
        return None

    def axis_var(keys):
        ss = 0.0
        for k in set(map(tuple, keys)):
            m = np.array([tuple(q) == k for q in keys])
            wk = w[m].sum()
            if wk <= 0:
                continue
            mk = (v[m] * w[m][:, None]).sum(0) / wk
            ss += wk * float(((mk - mu) ** 2).sum())
        return ss

    vz_ss = axis_var(np.asarray(vz).reshape(-1, 1))
    vt_ss = axis_var(np.stack([vy, vx], axis=1))
    n = len(v)
    nz = len(set(map(int, vz)))
    nt = len({(int(a), int(b)) for a, b in zip(vy, vx)})
    # Same upward bias as eta^2: with k groups and n observations, a grouping
    # of PURE NOISE still explains (k-1)/(n-1) of the variance. Verified by
    # simulation -- isotropic noise on a 12x12 grid returns ~7.8%, and the
    # formula predicts 11/143 = 7.7%. Never read these shares against zero.
    return {"across_z": vz_ss / tot, "across_tile": vt_ss / tot,
            "null_z": (nz - 1) / (n - 1) if n > 1 else float("nan"),
            "null_tile": (nt - 1) / (n - 1) if n > 1 else float("nan"),
            "residual": max(0.0, 1.0 - (vz_ss + vt_ss) / tot),
            "n_views": int(n), "n_z": nz, "n_tiles": nt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feature_dir")
    p.add_argument("--token", default="patch_mean_fg")
    p.add_argument("--fg_min", type=float, default=0.75)
    p.add_argument("--max_volumes", type=int, default=0,
                   help="0 = all")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.feature_dir, "*.npz"))
                   if not os.path.basename(f).startswith("."))
    if args.max_volumes:
        files = files[:args.max_volumes]
    if not files:
        raise SystemExit(f"no .npz in {args.feature_dir}")

    rows, skipped = [], 0
    for f in files:
        z = np.load(f)
        if args.token not in z:
            raise SystemExit(f"{f}: no '{args.token}' (has {list(z.keys())})")
        if "view_z" not in z or "view_yx" not in z:
            raise SystemExit(
                f"{f}: no view_z/view_yx — these features predate grid "
                f"coordinates being stored. Re-extract.")
        v = np.asarray(z[args.token], dtype=np.float64)
        vz = np.asarray(z["view_z"]).astype(int)
        vyx = np.asarray(z["view_yx"]).astype(int)
        fg = (np.asarray(z["view_fg_frac"], float) if "view_fg_frac" in z
              else np.ones(len(v)))
        keep = fg >= args.fg_min if args.fg_min > 0 else np.ones(len(v), bool)
        if keep.sum() < 4:
            skipped += 1
            continue
        d = decompose(v[keep], vz[keep], vyx[keep, 0], vyx[keep, 1], fg[keep])
        if d:
            d["stem"] = os.path.basename(f)[:-4]
            rows.append(d)

    if not rows:
        raise SystemExit("no volume had >=4 views above --fg_min")

    az = np.array([r["across_z"] for r in rows])
    at = np.array([r["across_tile"] for r in rows])
    re_ = np.array([r["residual"] for r in rows])

    print("=" * 70)
    print(" View-embedding variance, by grid axis")
    print("=" * 70)
    print(f"  {len(rows)} volume(s), token={args.token}, fg_min={args.fg_min}"
          + (f", {skipped} skipped (<4 views)" if skipped else ""))
    print(f"  median views/volume {int(np.median([r['n_views'] for r in rows]))}"
          f"  z-levels {int(np.median([r['n_z'] for r in rows]))}"
          f"  tiles {int(np.median([r['n_tiles'] for r in rows]))}")
    print("")
    nz0 = float(np.mean([r["null_z"] for r in rows]))
    nt0 = float(np.mean([r["null_tile"] for r in rows]))
    print(f"    {'axis':30s} {'share':>7} {'null':>7} {'excess':>8}")
    for lbl, a, n0 in (("across z    (depth)", az, nz0),
                       ("across tile (field position)", at, nt0),
                       ("residual", re_, float("nan"))):
        ex = "" if np.isnan(n0) else f"{a.mean() - n0:>+8.1%}"
        n0s = "" if np.isnan(n0) else f"{n0:>7.1%}"
        print(f"    {lbl:30s} {a.mean():>7.1%} {n0s:>7} {ex:>8}")
    print("")
    print(f"  'null' is what a grouping of PURE NOISE explains, (k-1)/(n-1).")
    print(f"  Only 'excess' is structure. Judge on that column, never on share.")
    print("")
    # Everything below is judged on the BIAS-CORRECTED excess.
    ex_z, ex_t = az.mean() - nz0, at.mean() - nt0
    if ex_t > 0.02 and ex_z > 0.02:
        print(f"  field position carries {ex_t / max(ex_z, 1e-9):.1f}x the "
              f"structure that depth does")
        print("")
    if ex_z < 0.05:
        print("  VERDICT: depth carries almost none of the view-to-view")
        print("  variance. Adjacent optical sections are near-duplicates at")
        print("  this z_stride, so a 3D aggregator has little to learn ALONG Z.")
        print("  Reducing z_stride will add views (less noise in the volume")
        print("  mean) but not new structure. Spend the parameters on the")
        print("  field axes instead, or on nothing at all.")
    elif ex_z < 0.20:
        print("  VERDICT: depth carries a modest share. A z-trend term is")
        print("  worth adding to the linear probe; a full 3D convnet is hard")
        print("  to justify on this much structure at n=22.")
    else:
        print("  VERDICT: depth carries a substantial share of the variance.")
        print("  There IS something along z for an aggregator to use --")
        print("  structured pooling first, then a 3D model if it pays.")
    print("")
    print("  NOTE this measures variance in the FEATURES, not association with")
    print("  force. Structure here is necessary for a 3D model to help, not")
    print("  sufficient: the variance still has to be force-related.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"volumes": rows,
                       "mean": {"across_z": float(az.mean()),
                                "across_tile": float(at.mean()),
                                "residual": float(re_.mean())}}, f, indent=2)
        print(f"\n  saved {args.output}")


if __name__ == "__main__":
    main()
