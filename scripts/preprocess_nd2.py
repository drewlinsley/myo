"""Convert .nd2 files to .npy arrays and compute per-volume percentile stats.

Usage:
    python scripts/preprocess_nd2.py --nd2_dir ../fixed_tissues --out_dir data

Output structure:
    data/
        bf/          # brightfield .npy files  (Z, H, W) uint16
        gfp/         # GFP .npy files          (Z, H, W) uint16
        stats/       # per-volume JSON stats
        manifest.json
"""

import os
import json
import argparse
import numpy as np

try:
    import nd2
except ImportError:
    raise ImportError("Install nd2: pip install nd2")


def compute_stats(volume, percentile_low=0.5, percentile_high=99.5):
    """Compute percentile-based normalization stats for a volume."""
    flat = volume.astype(np.float64).ravel()
    p_low, p_high = np.percentile(flat, [percentile_low, percentile_high])
    return {
        "p_low": float(p_low),
        "p_high": float(p_high),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def process_file(filepath, out_dir, percentile_clip=(0.5, 99.5)):
    """Process a single .nd2 file: extract BF and GFP channels, save as .npy."""
    stem = os.path.splitext(os.path.basename(filepath))[0]

    f = nd2.ND2File(filepath)
    data = f.asarray()  # (Z, C, H, W) typically
    sizes = dict(f.sizes)

    # Get channel names
    try:
        ch_names = [ch.channel.name for ch in f.metadata.channels]
    except Exception:
        ch_names = [f"ch{i}" for i in range(sizes.get("C", 1))]
    f.close()

    # Find BF and GFP channel indices
    bf_idx = None
    gfp_idx = None
    for i, name in enumerate(ch_names):
        name_lower = name.lower()
        if "bf" in name_lower or "bright" in name_lower:
            bf_idx = i
        elif "gfp" in name_lower or "green" in name_lower or "fluo" in name_lower:
            gfp_idx = i

    if bf_idx is None or gfp_idx is None:
        # Fallback: assume first channel is BF, second is GFP
        if len(ch_names) >= 2:
            bf_idx = 0
            gfp_idx = 1
            print(f"  Warning: Could not identify channels by name {ch_names}, "
                  f"using ch0=BF, ch1=GFP")
        else:
            print(f"  Skipping {stem}: need >=2 channels, found {ch_names}")
            return None

    # Extract channels: determine C axis position
    c_axis = list(sizes.keys()).index("C") if "C" in sizes else None
    if c_axis is None:
        print(f"  Skipping {stem}: no C dimension found")
        return None

    bf_slices = [slice(None)] * data.ndim
    bf_slices[c_axis] = bf_idx
    bf_vol = data[tuple(bf_slices)].squeeze()  # (Z, H, W)

    gfp_slices = [slice(None)] * data.ndim
    gfp_slices[c_axis] = gfp_idx
    gfp_vol = data[tuple(gfp_slices)].squeeze()  # (Z, H, W)

    # Save volumes
    bf_dir = os.path.join(out_dir, "bf")
    gfp_dir = os.path.join(out_dir, "gfp")
    stats_dir = os.path.join(out_dir, "stats")
    os.makedirs(bf_dir, exist_ok=True)
    os.makedirs(gfp_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    np.save(os.path.join(bf_dir, f"{stem}.npy"), bf_vol)
    np.save(os.path.join(gfp_dir, f"{stem}.npy"), gfp_vol)

    # Compute and save stats
    bf_stats = compute_stats(bf_vol, *percentile_clip)
    gfp_stats = compute_stats(gfp_vol, *percentile_clip)
    stats = {"bf": bf_stats, "gfp": gfp_stats}

    with open(os.path.join(stats_dir, f"{stem}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "stem": stem,
        "bf_shape": list(bf_vol.shape),
        "gfp_shape": list(gfp_vol.shape),
        "dtype": str(bf_vol.dtype),
        "channels": ch_names,
        "bf_idx": bf_idx,
        "gfp_idx": gfp_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess .nd2 files to .npy")
    parser.add_argument("--nd2_dir", type=str, default="../fixed_tissues",
                        help="Directory containing .nd2 files")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="Output directory for .npy files and stats")
    parser.add_argument("--percentile_low", type=float, default=0.5)
    parser.add_argument("--percentile_high", type=float, default=99.5)
    args = parser.parse_args()

    from glob import glob
    nd2_files = sorted(glob(os.path.join(args.nd2_dir, "*.nd2")))
    print(f"Found {len(nd2_files)} .nd2 files in {args.nd2_dir}")

    manifest = []
    for i, filepath in enumerate(nd2_files):
        print(f"[{i+1}/{len(nd2_files)}] {os.path.basename(filepath)}")
        info = process_file(filepath, args.out_dir,
                            (args.percentile_low, args.percentile_high))
        if info:
            manifest.append(info)
            print(f"  BF: {info['bf_shape']}, GFP: {info['gfp_shape']}, dtype: {info['dtype']}")

    # Save manifest
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Processed {len(manifest)}/{len(nd2_files)} files successfully")


if __name__ == "__main__":
    main()
