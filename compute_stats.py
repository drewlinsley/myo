"""Compute per-volume percentile stats for BF + GFP, write JSON sidecars.

For each stem present in {data_dir}/bf/ (and optionally /gfp/), writes
{data_dir}/stats/{stem}.json with:
    {
      "bf":  {"p_low": ..., "p_high": ...},
      "gfp": {"p_low": ..., "p_high": ...},  # omitted if no GFP file
      "z_auto": [lo, hi],                    # best z_window-slice band
      "z_auto_signal": "gfp_mean" | "bf_std"
    }

"z_auto" is the per-volume adaptive Z-band consumed by `z_range: auto` in the
configs: the contiguous window of --z_window slices with the highest mean GFP
(BF per-slice std when no GFP exists). Stacks vary in depth, so a fixed
[lo, hi] drops shallow stacks; the adaptive band uses every volume.

Idempotent — skips stems whose stats file already exists unless --force, but
UPGRADES existing stats that lack "z_auto" in place (band only, percentiles
untouched), so rerunning after a code update heals old stats dirs.

Usage:
    python compute_stats.py --data_dir data_new
    python compute_stats.py --data_dir data_new --percentile_clip 0.5 99.5
    python compute_stats.py --data_dir data_new --force          # rewrite
"""

import os
import sys
import json
import argparse
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.zband import compute_z_auto  # torch-free  # noqa: E402


def percentile_pair(arr, lo, hi):
    arr = arr.astype(np.float64)
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def z_band(bf_path, gfp_path, z_window):
    """(band, signal_name): per-slice mean GFP if available, else per-slice BF
    std (tissue has texture; empty planes are flat). mmap keeps RAM at one
    slice at a time."""
    if gfp_path and os.path.exists(gfp_path):
        vol = np.load(gfp_path, mmap_mode="r")
        profile = [float(np.asarray(vol[z], dtype=np.float32).mean())
                   for z in range(vol.shape[0])]
        return compute_z_auto(profile, z_window), "gfp_mean"
    vol = np.load(bf_path, mmap_mode="r")
    profile = [float(np.asarray(vol[z], dtype=np.float32).std())
               for z in range(vol.shape[0])]
    return compute_z_auto(profile, z_window), "bf_std"


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
    p.add_argument("--z_window", type=int, default=35,
                   help="Length of the adaptive z_auto band (slices)")
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
    n_upgraded = 0
    for bf_path in bf_files:
        stem = os.path.splitext(os.path.basename(bf_path))[0]
        out_path = os.path.join(stats_dir, f"{stem}.json")
        gfp_path = os.path.join(gfp_dir, f"{stem}.npy")

        if os.path.exists(out_path) and not args.force:
            # Upgrade-in-place: add the z_auto band to pre-existing stats
            # (percentiles untouched) so `z_range: auto` works on old dirs.
            with open(out_path) as f:
                stats = json.load(f)
            if "z_auto" in stats:
                n_skipped += 1
                continue
            band, signal = z_band(bf_path, gfp_path, args.z_window)
            stats["z_auto"] = band
            stats["z_auto_signal"] = signal
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
            n_upgraded += 1
            print(f"  {stem}: +z_auto={band} ({signal})")
            continue

        bf = np.load(bf_path)
        bf_lo, bf_hi = percentile_pair(bf, lo, hi)
        stats = {"bf": {"p_low": bf_lo, "p_high": bf_hi}}

        if os.path.exists(gfp_path):
            gfp = np.load(gfp_path)
            gfp_lo, gfp_hi = percentile_pair(gfp, lo, hi)
            stats["gfp"] = {"p_low": gfp_lo, "p_high": gfp_hi}
        else:
            print(f"  warn: no GFP for {stem}; writing BF-only stats")
        band, signal = z_band(bf_path, gfp_path, args.z_window)
        stats["z_auto"] = band
        stats["z_auto_signal"] = signal

        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        n_written += 1
        if stats.get("gfp"):
            print(f"  {stem}: bf=[{bf_lo:.1f},{bf_hi:.1f}] "
                  f"gfp=[{stats['gfp']['p_low']:.1f},"
                  f"{stats['gfp']['p_high']:.1f}] z_auto={band}")
        else:
            print(f"  {stem}: bf=[{bf_lo:.1f},{bf_hi:.1f}] z_auto={band}")

    print(f"\nWrote {n_written}, upgraded {n_upgraded} (added z_auto), "
          f"skipped {n_skipped} current.")


if __name__ == "__main__":
    main()
