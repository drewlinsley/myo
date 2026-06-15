"""Stage ND2 volumes from a Valo drop into the bf/ + gfp/ .npy layout
expected by compute_stats.py / eval_bfgfp_metrics.py / predict_classifier.py.

Each ND2 file is treated as ONE volume. Channels are split into:
    bf/<stem>.npy   - brightfield   (default --bf_channel 0)
    gfp/<stem>.npy  - fluorescence  (default --target_channel = last channel)
Output shape is always (Z, H, W) float32. T / P axes must be singleton.

Inspect a few files first to confirm channel order:
    pip install nd2
    python stage_nd2.py --src <valo_root> --inspect

Then stage:
    python stage_nd2.py \\
        --src data_phalloidin_mhc_051826/general/from-valo/skeletal_biowire_bf_fl/phalloidin_mhc_051826 \\
        --out data_phalloidin_mhc_051826_staged \\
        --bf_channel 0 --target_channel 2
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


def safe_stem(path, src_root):
    """Path-relative filename with separators flattened (keeps uniqueness)."""
    rel = os.path.relpath(path, src_root)
    stem = os.path.splitext(rel)[0]
    stem = re.sub(r"[\\/\s]+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]", "", stem)
    return stem


def read_nd2_volume(path):
    """Return numpy array shaped (Z, C, Y, X). Singleton dims kept; multi-T/P rejected."""
    import nd2
    with nd2.ND2File(path) as f:
        axes = list(f.sizes.keys())
        arr = np.asarray(f.asarray())

    keep_axes, keep_dims = [], []
    for ax, dim in zip(axes, arr.shape):
        if ax in ("T", "P"):
            if dim > 1:
                raise RuntimeError(
                    f"{path}: multi-{ax} ND2 (size {dim}) — one file should be one volume")
            continue
        keep_axes.append(ax)
        keep_dims.append(dim)
    arr = arr.reshape(keep_dims)

    canon = ["Z", "C", "Y", "X"]
    for c in canon:
        if c not in keep_axes:
            arr = np.expand_dims(arr, axis=-1)
            keep_axes.append(c)
    order = [keep_axes.index(c) for c in canon]
    return np.transpose(arr, order)  # (Z, C, Y, X)


def inspect(src):
    import nd2
    files = sorted(glob.glob(os.path.join(src, "**", "*.nd2"), recursive=True))
    print(f"Found {len(files)} .nd2 files under {src}")
    for fp in files[:5]:
        try:
            with nd2.ND2File(fp) as f:
                try:
                    ch_names = [c.channel.name for c in f.metadata.channels]
                except Exception:
                    ch_names = ["?"] * f.sizes.get("C", 1)
                print(f"  {os.path.relpath(fp, src)}: "
                      f"sizes={dict(f.sizes)} channels={ch_names}")
        except Exception as e:
            print(f"  {fp}: failed to open ({e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Root with .nd2 files (recursive)")
    p.add_argument("--out", default=None, help="Output root for {bf,gfp}/<stem>.npy")
    p.add_argument("--bf_channel", type=int, default=0)
    p.add_argument("--target_channel", type=int, default=-1,
                   help="Channel for the fluorescence target (-1 = last channel)")
    p.add_argument("--inspect", action="store_true",
                   help="Print shapes / channels for a few files then exit")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    try:
        import nd2  # noqa
    except ImportError:
        sys.exit("Need `pip install nd2` (Nikon NIS-Elements reader).")

    if args.inspect:
        inspect(args.src)
        return
    if not args.out:
        sys.exit("--out is required (omit only when using --inspect)")

    bf_dir = os.path.join(args.out, "bf")
    gfp_dir = os.path.join(args.out, "gfp")
    os.makedirs(bf_dir, exist_ok=True)
    os.makedirs(gfp_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.nd2"), recursive=True))
    if args.limit:
        files = files[:args.limit]
    print(f"Processing {len(files)} ND2 files")

    n_done = n_skip = n_err = 0
    for fp in files:
        stem = safe_stem(fp, args.src)
        bf_out = os.path.join(bf_dir, f"{stem}.npy")
        gfp_out = os.path.join(gfp_dir, f"{stem}.npy")
        if os.path.exists(bf_out) and os.path.exists(gfp_out) and not args.force:
            n_skip += 1
            continue
        try:
            vol = read_nd2_volume(fp)  # (Z, C, Y, X)
        except Exception as e:
            print(f"  ERR {os.path.relpath(fp, args.src)}: {e}")
            n_err += 1
            continue
        Z, C, H, W = vol.shape
        tgt = args.target_channel if args.target_channel >= 0 else C - 1
        if args.bf_channel >= C or tgt >= C:
            print(f"  ERR {stem}: file has {C} channel(s); "
                  f"asked for bf={args.bf_channel} target={tgt}")
            n_err += 1
            continue
        bf = vol[:, args.bf_channel].astype(np.float32)
        gfp = vol[:, tgt].astype(np.float32)
        np.save(bf_out, bf)
        np.save(gfp_out, gfp)
        n_done += 1
        if n_done <= 5 or n_done % 25 == 0:
            print(f"  [{n_done}/{len(files)}] {stem}: shape={bf.shape}")

    print(f"Done. wrote={n_done} skipped={n_skip} errors={n_err}")
    print(f"  bf/  : {bf_dir}")
    print(f"  gfp/ : {gfp_dir}")


if __name__ == "__main__":
    main()
