"""Extract encoder features from a trained 2D U-Net and visualize with UMAP/PLS.

Usage:
    python extract_features.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson/latest.pth \
        --metadata data/metadata.tsv \
        --output_dir features/ \
        --layers 5 \
        --aggregate none \
        --methods umap pls \
        --save_features
"""

import os
import csv
import json
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from collections import Counter
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint
from src.data.normalization import normalize

# Dataset column → which label column to use for that subset
DATASET_LABEL_COL = {
    "exercise": "Exercise",
    "perurbation": "Perturbation",   # matches typo in metadata
    "perturbation": "Perturbation",  # in case it gets fixed
}


def load_metadata(path):
    """Load metadata from .xlsx, .csv, or .tsv.

    Expected columns: File, Dataset, Exercise, Perturbation, ...
    All columns are preserved.  Numeric strings are converted to float;
    "NA" and empty strings become None.

    Returns:
        dict {stem: {"Dataset": str, "Exercise": str|None, ...}}
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        header = [str(c).strip() for c in next(rows)]
        data_rows = [{header[i]: (str(v).strip() if v is not None else "")
                      for i, v in enumerate(r)} for r in rows]
        wb.close()
    else:
        delimiter = "," if ext == ".csv" else "\t"
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data_rows = list(reader)

    mapping = {}
    for row in data_rows:
        nd2_name = row["File"].strip()
        stem = os.path.splitext(nd2_name)[0]
        entry = {}
        for col, val in row.items():
            if col == "File":
                continue
            val = val.strip() if val else ""
            if val == "" or val.upper() == "NA":
                entry[col] = None
                continue
            # Try numeric conversion
            try:
                entry[col] = float(val)
            except (ValueError, TypeError):
                entry[col] = val
        mapping[stem] = entry
    return mapping


def extract_encoder_features(model, bf_vol, device, layers, batch_size,
                             mask=None):
    """Run Z-slices through the encoder and pool feature maps.

    Args:
        model: 2D smp Unet (has .encoder attribute)
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device
        layers: list of encoder stage indices (0-5) to extract
        batch_size: slices per forward pass
        mask: optional (Z, H, W) boolean array — True = foreground.
              When provided, pools only over foreground spatial locations.

    Returns:
        features: (Z, D_feat) numpy array
    """
    Z, H, W = bf_vol.shape

    # Pad spatial dims to multiples of 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32

    all_features = []

    for start in range(0, Z, batch_size):
        end = min(start + batch_size, Z)
        batch_slices = []
        batch_masks = []
        for z in range(start, end):
            slc = bf_vol[z]
            if pad_h > 0 or pad_w > 0:
                slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")
            batch_slices.append(slc)
            if mask is not None:
                m = mask[z].astype(np.float32)
                if pad_h > 0 or pad_w > 0:
                    m = np.pad(m, ((0, pad_h), (0, pad_w)), mode="constant")
                batch_masks.append(m)

        # (B, 1, H', W')
        inp = torch.from_numpy(
            np.stack(batch_slices)[:, np.newaxis]
        ).float().to(device)

        # (B, 1, H', W') mask at input resolution
        if batch_masks:
            mask_t = torch.from_numpy(
                np.stack(batch_masks)[:, np.newaxis]
            ).float().to(device)
        else:
            mask_t = None

        encoder_out = model.encoder(inp)  # list of 6 feature maps

        layer_feats = []
        for li in layers:
            feat = encoder_out[li]  # (B, C, H', W')

            if mask_t is not None:
                # Downsample mask to match this feature map's spatial size
                mask_down = F.interpolate(
                    mask_t, size=feat.shape[2:], mode="nearest"
                )  # (B, 1, h, w)
                # Masked mean: sum(feat * mask) / sum(mask)
                numer = (feat * mask_down).sum(dim=(2, 3))   # (B, C)
                denom = mask_down.sum(dim=(2, 3)).clamp(min=1)  # (B, 1)
                pooled = numer / denom
            else:
                pooled = F.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)
                pooled = pooled.squeeze(-1).squeeze(-1)   # (B, C)

            layer_feats.append(pooled)

        # Concatenate across layers → (B, D_feat)
        combined = torch.cat(layer_feats, dim=1)
        all_features.append(combined.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (Z, D_feat)


def extract_encoder_features_3d(model, bf_vol, device, layers,
                                patch_depth=32, crop_size=256,
                                stride_z=None, stride_hw=None):
    """Patch-based feature extraction for a 3D U-Net encoder.

    Extracts non-overlapping (or strided) 3D patches, runs each through
    the encoder, global-avg-pools the requested feature maps, and returns
    (N_patches, D_feat).  Callers pool across patches via `_pool_volume`
    exactly like the 2D path pools across Z-slices.

    Args:
        model: 3D smp_3d U-Net (has .encoder attribute accepting (B, 1, H, W, D))
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device
        layers: list of encoder stage indices to concat
        patch_depth, crop_size: patch dims (must match 32-multiples)
        stride_z, stride_hw: grid strides (default = patch size, non-overlapping)

    Returns:
        features: (N_patches, D_feat) numpy array
    """
    import torch.nn.functional as F  # local import safe; F already imported above
    Z, H, W = bf_vol.shape
    pd, cs = patch_depth, crop_size
    sz = stride_z if stride_z else pd
    shw = stride_hw if stride_hw else cs

    # Reflect-pad to at least one patch in every dim
    pad_z = max(0, pd - Z)
    pad_h = max(0, cs - H)
    pad_w = max(0, cs - W)
    if pad_z or pad_h or pad_w:
        bf_vol = np.pad(bf_vol, ((0, pad_z), (0, pad_h), (0, pad_w)),
                        mode="reflect")
    Z, H, W = bf_vol.shape

    # Grid of patch starts (inclusive of last full-size patch)
    def grid(total, size, stride):
        if total <= size:
            return [0]
        out = list(range(0, total - size + 1, stride))
        if out[-1] != total - size:
            out.append(total - size)
        return out

    zs = grid(Z, pd, sz)
    ys = grid(H, cs, shw)
    xs = grid(W, cs, shw)

    all_feats = []
    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    patch = bf_vol[z0:z0+pd, y0:y0+cs, x0:x0+cs]
                    # (D, H, W) -> (1, 1, H, W, D) to match smp_3d BCHWD
                    patch_t = torch.from_numpy(
                        patch.transpose(1, 2, 0)[None, None].copy()
                    ).float().to(device)
                    encoder_out = model.encoder(patch_t)
                    layer_feats = []
                    for li in layers:
                        feat = encoder_out[li]  # (1, C, H', W', D')
                        pooled = F.adaptive_avg_pool3d(feat, 1).flatten(1)  # (1, C)
                        layer_feats.append(pooled)
                    all_feats.append(
                        torch.cat(layer_feats, dim=1).cpu().numpy())
    return np.concatenate(all_feats, axis=0)  # (N_patches, D_feat)


def run_umap(features, labels, dataset_name, label_col, layer_str, output_dir,
             n_neighbors, min_dist):
    """Fit UMAP on one dataset subset and save scatter plot."""
    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
        return

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   random_state=42)
    embedding = reducer.fit_transform(features)
    _save_scatter(embedding, labels, "UMAP", dataset_name, label_col, layer_str,
                  output_dir)


def run_pls(features, labels, dataset_name, label_col, layer_str, output_dir,
            n_components):
    """Fit PLS on one dataset subset and save scatter plot."""
    from sklearn.cross_decomposition import PLSRegression

    unique_vals = sorted(set(labels))
    if len(unique_vals) < 2:
        print(f"PLS ({dataset_name}): requires >=2 unique labels, "
              f"got {len(unique_vals)}. Skipping.")
        return

    cond_to_int = {c: i for i, c in enumerate(unique_vals)}
    y = np.array([cond_to_int[l] for l in labels], dtype=np.float64)

    n_comp = min(n_components, len(unique_vals) - 1, features.shape[1])
    pls = PLSRegression(n_components=n_comp)
    embedding = pls.fit_transform(features, y)[0]  # X scores

    if embedding.shape[1] == 1:
        embedding = np.column_stack([embedding, np.zeros(len(embedding))])

    _save_scatter(embedding, labels, "PLS", dataset_name, label_col, layer_str,
                  output_dir)


COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9",
]


def _save_scatter(embedding, labels, method, dataset_name, label_col, layer_str,
                  output_dir):
    """Create and save a labeled scatter plot."""
    unique_vals = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, val in enumerate(unique_vals):
        mask = np.array([l == val for l in labels])
        color = COLORS[i % len(COLORS)]
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, label=val, s=20, alpha=0.8, edgecolors="white",
                   linewidths=0.3)

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{method} — {dataset_name} — layers {layer_str} — by {label_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2)
    fig.tight_layout()

    fname = f"{method.lower()}_layers_{layer_str}_{dataset_name.lower()}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract encoder features and visualize with UMAP/PLS")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--metadata", required=True,
                        help="Tab-separated metadata file (columns: File, Exercise, Perturbation, ...)")
    parser.add_argument("--output_dir", default="features/",
                        help="Output directory for plots and features")
    parser.add_argument("--layers", type=int, nargs="+", default=[5],
                        help="Encoder stage indices to extract (0-5). Default: [5] (bottleneck)")
    parser.add_argument("--aggregate", choices=["none", "volume"], default="none",
                        help="'none'=per-slice, 'volume'=mean across Z per volume")
    parser.add_argument("--methods", nargs="+", choices=["umap", "pls"],
                        default=["umap", "pls"],
                        help="Dimensionality reduction methods to run")
    parser.add_argument("--save_features", action="store_true",
                        help="Save raw features as .npz")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Z-slices per forward pass")
    parser.add_argument("--umap_neighbors", type=int, default=5,
                        help="UMAP n_neighbors parameter (smaller=tighter clusters)")
    parser.add_argument("--umap_min_dist", type=float, default=0.01,
                        help="UMAP min_dist parameter (smaller=tighter clusters)")
    parser.add_argument("--pls_components", type=int, default=2,
                        help="Number of PLS components")
    args = parser.parse_args()

    # ── Config + model ──────────────────────────────────────────────
    cfg = load_config(args.config)
    dims = cfg["model"]["dims"]
    if dims != "2d":
        raise NotImplementedError(
            f"Feature extraction currently supports 2D models only, got dims='{dims}'")

    apply_timm = cfg["model"].get("encoder_weights") is not None

    for li in args.layers:
        if li == 0:
            warnings.warn("Stage 0 returns the input channels (not very informative).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with no pretrained weights (loading from checkpoint)
    cfg_copy = cfg.copy()
    cfg_copy["model"] = cfg["model"].copy()
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)

    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?')})")

    model = model.to(device)
    model.eval()

    # ── Metadata ────────────────────────────────────────────────────
    metadata = load_metadata(args.metadata)
    print(f"Loaded {len(metadata)} entries from {args.metadata}")

    # ── Data ────────────────────────────────────────────────────────
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    stats_dir = os.path.join(data_dir, "stats")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    print(f"Found {len(bf_files)} volumes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect features per volume, keyed by stem
    vol_features = {}   # stem → (N, D_feat) array  (includes nonzero-proportion)
    vol_meta = {}       # stem → metadata dict

    gfp_dir = os.path.join(data_dir, "gfp")
    z_range = cfg["data"].get("z_range", None)

    with torch.no_grad():
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]

            # Metadata lookup
            if stem in metadata:
                meta = metadata[stem]
            else:
                warnings.warn(f"Stem '{stem}' not in metadata — skipping")
                continue

            # Load stats
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            # Load and normalize BF
            bf_raw = np.load(bf_path)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            # Load GFP for non-zero proportion covariate
            gfp_path = os.path.join(gfp_dir, f"{stem}.npy")
            gfp_raw = np.load(gfp_path)
            if z_range is not None:
                gfp_raw = gfp_raw[z_lo:z_hi]

            # Extract features: (Z, D_feat)
            feats = extract_encoder_features(
                model, bf, device, args.layers, args.batch_size)

            # Compute proportion of non-zero GFP pixels per slice: (Z, 1)
            Z = feats.shape[0]
            nonzero_prop = np.array(
                [(gfp_raw[z] > 0).mean() for z in range(Z)],
                dtype=np.float32,
            ).reshape(-1, 1)

            if args.aggregate == "volume":
                feats = feats.mean(axis=0, keepdims=True)        # (1, D_feat)
                nonzero_prop = nonzero_prop.mean(axis=0, keepdims=True)  # (1, 1)

            # Append non-zero proportion as extra feature column
            feats = np.concatenate([feats, nonzero_prop], axis=1)

            vol_features[stem] = feats
            vol_meta[stem] = meta
            nzp = nonzero_prop.mean()
            print(f"  {stem}: {feats.shape[0]} rows, Dataset={meta['Dataset']}, "
                  f"nonzero_gfp={nzp:.3f}")

    layer_str = "_".join(str(l) for l in sorted(args.layers))

    # ── Group volumes by Dataset, fit separately ────────────────────
    # Gather unique dataset values
    datasets_seen = sorted(set(m["Dataset"].lower() for m in vol_meta.values()))
    print(f"\nDatasets found: {datasets_seen}")

    for ds in datasets_seen:
        label_col = DATASET_LABEL_COL.get(ds)
        if label_col is None:
            warnings.warn(f"Unknown dataset '{ds}' — no label column mapped. Skipping.")
            continue

        # Collect features + labels for this dataset subset
        ds_features = []
        ds_labels = []
        ds_stems = []
        for stem, meta in vol_meta.items():
            if meta["Dataset"].lower() != ds:
                continue
            ds_features.append(vol_features[stem])
            n = vol_features[stem].shape[0]
            ds_labels.extend([meta[label_col]] * n)
            ds_stems.extend([stem] * n)

        if not ds_features:
            continue

        features = np.concatenate(ds_features, axis=0)
        ds_name = ds.capitalize()

        print(f"\n── {ds_name} ({label_col}) ──")
        print(f"  Samples: {features.shape[0]}, feature dim: {features.shape[1]}")
        counts = Counter(ds_labels)
        for val, n in sorted(counts.items()):
            print(f"    {val}: {n}")

        # Save features per dataset
        if args.save_features:
            npz_path = os.path.join(
                args.output_dir, f"features_layers_{layer_str}_{ds}.npz")
            np.savez(npz_path,
                     features=features,
                     labels=np.array(ds_labels),
                     stems=np.array(ds_stems),
                     layer_indices=np.array(args.layers))
            print(f"  Saved {npz_path}")

        # Dim reduction + plots
        if "umap" in args.methods:
            print(f"  Running UMAP ({ds_name})...")
            run_umap(features, ds_labels, ds_name, label_col, layer_str,
                     args.output_dir, args.umap_neighbors, args.umap_min_dist)

        if "pls" in args.methods:
            print(f"  Running PLS ({ds_name})...")
            run_pls(features, ds_labels, ds_name, label_col, layer_str,
                    args.output_dir, args.pls_components)

    print("\nDone.")


if __name__ == "__main__":
    main()
