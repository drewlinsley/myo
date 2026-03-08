"""Data exploration script for .nd2 files in the fixed_tissues directory.

Run this on the VM:
    python explore_data.py --data_dir ../fixed_tissues

It will report:
    - Per-file: shape, dtype, channel names, dimension order, min/max/mean per channel
    - Summary: common patterns across all files
    - Sample visualizations saved to explore_output/
"""

import os
import argparse
import json
import numpy as np
from glob import glob

# Try nd2 library first, fall back to alternatives
try:
    import nd2
    HAS_ND2 = True
except ImportError:
    HAS_ND2 = False

try:
    from aicsimageio import AICSImage
    HAS_AICS = True
except ImportError:
    HAS_AICS = False


def explore_nd2_file(filepath):
    """Extract metadata and stats from a single .nd2 file."""
    info = {"file": os.path.basename(filepath)}

    if HAS_ND2:
        try:
            f = nd2.ND2File(filepath)
            info["shape"] = list(f.shape)
            info["ndim"] = f.ndim
            info["dtype"] = str(f.dtype)
            info["sizes"] = dict(f.sizes)  # e.g. {'T': 1, 'C': 3, 'Z': 20, 'Y': 512, 'X': 512}

            # Channel metadata
            try:
                info["channel_names"] = [ch.channel.name for ch in f.metadata.channels]
            except Exception:
                info["channel_names"] = "unavailable"

            # Pixel size / voxel info
            try:
                voxel = f.voxel_size()
                info["voxel_size_um"] = {"x": voxel.x, "y": voxel.y, "z": voxel.z}
            except Exception:
                info["voxel_size_um"] = "unavailable"

            # Read actual data for stats
            data = f.asarray()
            info["data_shape"] = list(data.shape)
            info["data_dtype"] = str(data.dtype)
            info["data_min"] = float(np.min(data))
            info["data_max"] = float(np.max(data))
            info["data_mean"] = float(np.mean(data))

            # Per-channel stats if C dimension exists
            sizes = f.sizes
            if "C" in sizes and sizes["C"] > 1:
                c_axis = list(sizes.keys()).index("C")
                per_channel = []
                for c in range(sizes["C"]):
                    slices = [slice(None)] * data.ndim
                    slices[c_axis] = c
                    ch_data = data[tuple(slices)]
                    ch_name = info["channel_names"][c] if isinstance(info["channel_names"], list) and c < len(info["channel_names"]) else f"ch{c}"
                    per_channel.append({
                        "channel": ch_name,
                        "shape": list(ch_data.shape),
                        "min": float(np.min(ch_data)),
                        "max": float(np.max(ch_data)),
                        "mean": float(np.mean(ch_data)),
                        "std": float(np.std(ch_data)),
                        "nonzero_fraction": float(np.count_nonzero(ch_data) / ch_data.size),
                    })
                info["per_channel_stats"] = per_channel

            f.close()
            return info

        except Exception as e:
            info["error_nd2"] = str(e)

    if HAS_AICS:
        try:
            img = AICSImage(filepath)
            info["shape"] = list(img.shape)
            info["dims"] = img.dims.order  # e.g. 'TCZYX'
            info["channel_names"] = img.channel_names
            info["physical_pixel_sizes"] = {
                "Z": img.physical_pixel_sizes.Z,
                "Y": img.physical_pixel_sizes.Y,
                "X": img.physical_pixel_sizes.X,
            }
            data = img.data
            info["data_shape"] = list(data.shape)
            info["data_dtype"] = str(data.dtype)
            info["data_min"] = float(np.min(data))
            info["data_max"] = float(np.max(data))
            info["data_mean"] = float(np.mean(data))

            # Per-channel stats
            c_idx = img.dims.order.index("C")
            per_channel = []
            for c in range(data.shape[c_idx]):
                slices = [slice(None)] * data.ndim
                slices[c_idx] = c
                ch_data = data[tuple(slices)]
                ch_name = img.channel_names[c] if c < len(img.channel_names) else f"ch{c}"
                per_channel.append({
                    "channel": ch_name,
                    "shape": list(ch_data.shape),
                    "min": float(np.min(ch_data)),
                    "max": float(np.max(ch_data)),
                    "mean": float(np.mean(ch_data)),
                    "std": float(np.std(ch_data)),
                    "nonzero_fraction": float(np.count_nonzero(ch_data) / ch_data.size),
                })
            info["per_channel_stats"] = per_channel
            return info

        except Exception as e:
            info["error_aics"] = str(e)

    # Fallback: try reading with nd2.imread
    if HAS_ND2:
        try:
            data = nd2.imread(filepath)
            info["data_shape"] = list(data.shape)
            info["data_dtype"] = str(data.dtype)
            info["data_min"] = float(np.min(data))
            info["data_max"] = float(np.max(data))
            info["data_mean"] = float(np.mean(data))
            return info
        except Exception as e:
            info["error_imread"] = str(e)

    info["error"] = "No supported library could read this file. Install nd2 or aicsimageio."
    return info


def save_sample_slices(filepath, output_dir, max_channels=6):
    """Save sample mid-slice images for visual inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    try:
        if HAS_ND2:
            f = nd2.ND2File(filepath)
            data = f.asarray()
            sizes = dict(f.sizes)
            try:
                ch_names = [ch.channel.name for ch in f.metadata.channels]
            except Exception:
                ch_names = [f"ch{i}" for i in range(sizes.get("C", 1))]
            f.close()
        else:
            return

        basename = os.path.splitext(os.path.basename(filepath))[0]

        # If there's a C dimension, show each channel's mid-Z slice
        if "C" in sizes and "Z" in sizes:
            c_axis = list(sizes.keys()).index("C")
            z_axis = list(sizes.keys()).index("Z")
            mid_z = data.shape[z_axis] // 2

            n_ch = min(sizes["C"], max_channels)
            fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 4))
            if n_ch == 1:
                axes = [axes]

            for c in range(n_ch):
                slices = [slice(None)] * data.ndim
                slices[c_axis] = c
                slices[z_axis] = mid_z
                img = data[tuple(slices)].squeeze()

                ax = axes[c]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"{ch_names[c] if c < len(ch_names) else f'ch{c}'}\nmin={img.min():.0f} max={img.max():.0f}")
                ax.axis("off")

            plt.suptitle(f"{basename} (mid-Z={mid_z})", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{basename}_channels.png"), dpi=150)
            plt.close()

        # Also save a Z-stack montage for the first fluorescence-looking channel
        if "Z" in sizes:
            z_axis = list(sizes.keys()).index("Z")
            n_z = sizes["Z"]
            # Pick a channel (try second channel if available, likely fluorescence)
            if "C" in sizes and sizes["C"] > 1:
                c_axis = list(sizes.keys()).index("C")
                # Show Z montage for each channel
                for c in range(min(sizes["C"], max_channels)):
                    slices = [slice(None)] * data.ndim
                    slices[c_axis] = c
                    ch_data = data[tuple(slices)].squeeze()

                    # Pick evenly spaced Z slices
                    n_show = min(n_z, 16)
                    z_indices = np.linspace(0, n_z - 1, n_show, dtype=int)
                    ncols = min(4, n_show)
                    nrows = (n_show + ncols - 1) // ncols

                    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
                    axes_flat = np.atleast_1d(axes).ravel()
                    for i, zi in enumerate(z_indices):
                        z_slices = [slice(None)] * ch_data.ndim
                        z_slices[z_axis if "C" not in sizes else z_axis - 1] = zi
                        try:
                            zimg = ch_data[tuple(z_slices)].squeeze()
                        except Exception:
                            zimg = ch_data[zi].squeeze() if ch_data.ndim >= 3 else ch_data.squeeze()
                        axes_flat[i].imshow(zimg, cmap="gray")
                        axes_flat[i].set_title(f"Z={zi}", fontsize=8)
                        axes_flat[i].axis("off")
                    for i in range(len(z_indices), len(axes_flat)):
                        axes_flat[i].axis("off")

                    ch_label = ch_names[c] if c < len(ch_names) else f"ch{c}"
                    plt.suptitle(f"{basename} - {ch_label} Z-stack", fontsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{basename}_{ch_label}_zstack.png"), dpi=100)
                    plt.close()

    except Exception as e:
        print(f"  Error saving visualizations for {filepath}: {e}")


def main(data_dir, output_dir="explore_output", max_vis=5):
    os.makedirs(output_dir, exist_ok=True)

    nd2_files = sorted(glob(os.path.join(data_dir, "*.nd2")))
    print(f"Found {len(nd2_files)} .nd2 files in {data_dir}")

    if not nd2_files:
        print("No .nd2 files found. Check the path.")
        return

    print(f"Libraries: nd2={HAS_ND2}, aicsimageio={HAS_AICS}")
    print("=" * 80)

    all_info = []
    for i, filepath in enumerate(nd2_files):
        print(f"\n[{i+1}/{len(nd2_files)}] {os.path.basename(filepath)}")
        info = explore_nd2_file(filepath)
        all_info.append(info)

        # Print key info
        for key in ["sizes", "channel_names", "data_shape", "data_dtype", "voxel_size_um"]:
            if key in info:
                print(f"  {key}: {info[key]}")
        if "per_channel_stats" in info:
            for ch in info["per_channel_stats"]:
                print(f"  Channel '{ch['channel']}': shape={ch['shape']} range=[{ch['min']:.1f}, {ch['max']:.1f}] mean={ch['mean']:.1f} std={ch['std']:.1f} nonzero={ch['nonzero_fraction']:.3f}")
        if "error" in info:
            print(f"  ERROR: {info['error']}")

        # Save visualizations for first few files
        if i < max_vis:
            save_sample_slices(filepath, output_dir)

    # Save full report
    report_path = os.path.join(output_dir, "exploration_report.json")
    with open(report_path, "w") as f:
        json.dump(all_info, f, indent=2, default=str)
    print(f"\n{'=' * 80}")
    print(f"Full report saved to: {report_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    shapes = set()
    channel_sets = set()
    dtypes = set()
    for info in all_info:
        if "data_shape" in info:
            shapes.add(tuple(info["data_shape"]))
        if "channel_names" in info and isinstance(info["channel_names"], list):
            channel_sets.add(tuple(info["channel_names"]))
        if "data_dtype" in info:
            dtypes.add(info["data_dtype"])

    print(f"Unique shapes: {shapes}")
    print(f"Unique channel sets: {channel_sets}")
    print(f"Unique dtypes: {dtypes}")

    if channel_sets:
        print(f"\nChannel names found:")
        for cs in channel_sets:
            print(f"  {list(cs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore .nd2 files")
    parser.add_argument("--data_dir", type=str, default="../fixed_tissues",
                        help="Directory containing .nd2 files")
    parser.add_argument("--output_dir", type=str, default="explore_output",
                        help="Directory to save exploration results")
    parser.add_argument("--max_vis", type=int, default=5,
                        help="Max number of files to generate visualizations for")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.max_vis)
