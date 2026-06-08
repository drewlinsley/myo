"""Compute per-volume percentile stats for BF + GFP, write JSON sidecars.

For each stem present in {data_dir}/bf/ (and optionally /gfp/), writes
{data_dir}/stats/{stem}.json with:
    {
      "bf":  {"p_low": ..., "p_high": ...},
      "gfp": {"p_low": ..., "p_high": ...}   # omitted if no GFP file
    }

Idempotent — skips stems whose stats file already exists unless --force.

Usage:
    python compute_stats.py --data_dir data_new
    python compute_stats.py --data_dir data_new --percentile_clip 0.5 99.5
    python compute_stats.py --data_dir data_new --force          # rewrite
"""

import os
import json
import argparse
from glob import glob

import numpy as np


def percentile_pair(arr, lo, hi):
    arr = arr.astype(np.float64)
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Dataset root; expects {data_dir}/bf/ and /gfp/")
    p.add_argument("--percentile_clip", nargs=2, type=float,
                   default=[0.5, 99.5])
    p.add_argument("--force", action="store_true",
                   help="Recompute even if stats JSON exists")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Optional subset of stems to process")
    args = p.parse_args()

    bf_dir = os.path.join(args.data_dir, "bf")
    gfp_dir = os.path.join(args.data_dir, "gfp")
    stats_dir = os.path.join(args.data_dir, "stats")
    if not os.path.isdir(bf_dir):
        raise SystemExit(f"Missing {bf_dir}")
    os.makedirs(stats_dir, exist_ok=True)

    lo, hi = args.percentile_clip
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    if args.stems:
        keep = set(args.stems)
        bf_files = [f for f in bf_files
                    if os.path.splitext(os.path.basename(f))[0] in keep]

    n_written = 0
    n_skipped = 0
    for bf_path in bf_files:
        stem = os.path.splitext(os.path.basename(bf_path))[0]
        out_path = os.path.join(stats_dir, f"{stem}.json")
        if os.path.exists(out_path) and not args.force:
            n_skipped += 1
            continue

        bf = np.load(bf_path)
        bf_lo, bf_hi = percentile_pair(bf, lo, hi)
        stats = {"bf": {"p_low": bf_lo, "p_high": bf_hi}}

        gfp_path = os.path.join(gfp_dir, f"{stem}.npy")
        if os.path.exists(gfp_path):
            gfp = np.load(gfp_path)
            gfp_lo, gfp_hi = percentile_pair(gfp, lo, hi)
            stats["gfp"] = {"p_low": gfp_lo, "p_high": gfp_hi}
        else:
            print(f"  warn: no GFP for {stem}; writing BF-only stats")

        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        n_written += 1
        if stats.get("gfp"):
            print(f"  {stem}: bf=[{bf_lo:.1f},{bf_hi:.1f}] "
                  f"gfp=[{stats['gfp']['p_low']:.1f},"
                  f"{stats['gfp']['p_high']:.1f}]")
        else:
            print(f"  {stem}: bf=[{bf_lo:.1f},{bf_hi:.1f}]")

    print(f"\nWrote {n_written}, skipped {n_skipped} existing.")


if __name__ == "__main__":
    main()
