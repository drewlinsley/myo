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

# Metadata columns to use as label sets (each gets its own scatter plot)
LABEL_COLUMNS = ["Exercise", "Perturbation"]


def load_metadata(path):
    """Load metadata from .xlsx, .csv, or .tsv.

    Expected columns include 'File', 'Exercise', 'Perturbation'.
    Stems are derived by stripping the .nd2 extension from File.

    Returns:
        dict {stem: {col: value, ...}} for each LABEL_COLUMNS entry
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
        mapping[stem] = {col: row[col].strip() for col in LABEL_COLUMNS}
    return mapping


def extract_encoder_features(model, bf_vol, device, layers, batch_size):
    """Run Z-slices through the encoder and pool feature maps.

    Args:
        model: 2D smp Unet (has .encoder attribute)
        bf_vol: (Z, H, W) normalized float32 array
        device: torch device
        layers: list of encoder stage indices (0-5) to extract
        batch_size: slices per forward pass

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
        for z in range(start, end):
            slc = bf_vol[z]
            if pad_h > 0 or pad_w > 0:
                slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")
            batch_slices.append(slc)

        # (B, 1, H', W')
        inp = torch.from_numpy(
            np.stack(batch_slices)[:, np.newaxis]
        ).float().to(device)

        encoder_out = model.encoder(inp)  # list of 6 feature maps

        layer_feats = []
        for li in layers:
            feat = encoder_out[li]  # (B, C, H', W')
            pooled = F.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)
            pooled = pooled.squeeze(-1).squeeze(-1)   # (B, C)
            layer_feats.append(pooled)

        # Concatenate across layers → (B, D_feat)
        combined = torch.cat(layer_feats, dim=1)
        all_features.append(combined.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (Z, D_feat)


def run_umap(features, label_dict, layer_str, output_dir, n_neighbors, min_dist):
    """Fit UMAP once, then save a scatter plot per label column."""
    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
        return

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   random_state=42)
    embedding = reducer.fit_transform(features)

    for col, labels in label_dict.items():
        _save_scatter(embedding, labels, "UMAP", col, layer_str, output_dir)


def run_pls(features, label_dict, layer_str, output_dir, n_components):
    """Fit PLS separately per label column and save scatter plots."""
    from sklearn.cross_decomposition import PLSRegression

    for col, labels in label_dict.items():
        unique_vals = sorted(set(labels))
        if len(unique_vals) < 2:
            print(f"PLS ({col}): requires >=2 unique values, got {len(unique_vals)}. Skipping.")
            continue

        cond_to_int = {c: i for i, c in enumerate(unique_vals)}
        y = np.array([cond_to_int[l] for l in labels], dtype=np.float64)

        n_comp = min(n_components, len(unique_vals) - 1, features.shape[1])
        pls = PLSRegression(n_components=n_comp)
        embedding = pls.fit_transform(features, y)[0]  # X scores

        if embedding.shape[1] == 1:
            embedding = np.column_stack([embedding, np.zeros(len(embedding))])

        _save_scatter(embedding, labels, "PLS", col, layer_str, output_dir)


def _save_scatter(embedding, labels, method, color_col, layer_str, output_dir):
    """Create and save a labeled scatter plot."""
    unique_vals = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_vals), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, val in enumerate(unique_vals):
        mask = np.array([l == val for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], label=val, s=12, alpha=0.7)

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{method} — layers {layer_str} — colored by {color_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2)
    fig.tight_layout()

    fname = f"{method.lower()}_layers_{layer_str}_{color_col.lower()}.png"
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
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
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

    all_features = []
    # Per-column label lists: {col_name: [label, label, ...]}
    all_labels = {col: [] for col in LABEL_COLUMNS}
    all_stems = []
    all_z_indices = []

    z_range = cfg["data"].get("z_range", None)

    with torch.no_grad():
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]

            # Metadata lookup
            if stem in metadata:
                meta = metadata[stem]
            else:
                warnings.warn(f"Stem '{stem}' not in metadata — labeling 'unknown'")
                meta = {col: "unknown" for col in LABEL_COLUMNS}

            # Load stats
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            # Load and normalize
            bf_raw = np.load(bf_path)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            # Extract features: (Z, D_feat)
            feats = extract_encoder_features(
                model, bf, device, args.layers, args.batch_size)

            Z = feats.shape[0]
            if args.aggregate == "volume":
                feats = feats.mean(axis=0, keepdims=True)  # (1, D_feat)
                for col in LABEL_COLUMNS:
                    all_labels[col].append(meta[col])
                all_stems.append(stem)
                all_z_indices.append(-1)
            else:
                for col in LABEL_COLUMNS:
                    all_labels[col].extend([meta[col]] * Z)
                all_stems.extend([stem] * Z)
                z_start = z_range[0] if z_range else 0
                all_z_indices.extend(list(range(z_start, z_start + Z)))

            all_features.append(feats)
            print(f"  {stem}: {Z} slices, Exercise={meta['Exercise']}, "
                  f"Perturbation={meta['Perturbation']}")

    features = np.concatenate(all_features, axis=0)  # (N, D_feat)
    stems = all_stems
    z_indices = np.array(all_z_indices)

    print(f"\nTotal samples: {features.shape[0]}, feature dim: {features.shape[1]}")

    for col in LABEL_COLUMNS:
        counts = Counter(all_labels[col])
        print(f"\n  {col}:")
        for val, n in sorted(counts.items()):
            print(f"    {val}: {n}")

    # ── Layer string for filenames ──────────────────────────────────
    layer_str = "_".join(str(l) for l in sorted(args.layers))

    # ── Save features ───────────────────────────────────────────────
    if args.save_features:
        npz_path = os.path.join(args.output_dir, f"features_layers_{layer_str}.npz")
        save_dict = dict(
            features=features,
            stems=np.array(stems),
            z_indices=z_indices,
            layer_indices=np.array(args.layers),
        )
        for col in LABEL_COLUMNS:
            save_dict[col.lower()] = np.array(all_labels[col])
        np.savez(npz_path, **save_dict)
        print(f"Saved features to {npz_path}")

    # ── Dimensionality reduction + plots ────────────────────────────
    if "umap" in args.methods:
        print("\nRunning UMAP...")
        run_umap(features, all_labels, layer_str, args.output_dir,
                 args.umap_neighbors, args.umap_min_dist)

    if "pls" in args.methods:
        print("\nRunning PLS...")
        run_pls(features, all_labels, layer_str, args.output_dir,
                args.pls_components)

    print("\nDone.")


if __name__ == "__main__":
    main()
