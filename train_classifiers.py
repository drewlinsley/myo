"""Classify volumes via leave-one-out k-NN on encoder features.

For a given seg checkpoint, extracts volume-level encoder features, then
runs leave-one-out k-NN classification for each task (Exercise, Perturbation).
No train/test split — each volume is classified by its nearest neighbors
among all other volumes.  Truly zero-shot w.r.t. classifier training data.

Usage:
    python train_classifiers.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson_frac025/best.pth \
        --metadata data_mapping_drew.xlsx \
        --seg_tag frac025 \
        --output results/classifier/frac025.json

    # For untrained (0%) baseline:
    python train_classifiers.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --no_checkpoint \
        --metadata data_mapping_drew.xlsx \
        --seg_tag frac000 \
        --output results/classifier/frac000.json
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint
from src.data.normalization import normalize
from extract_features import (
    load_metadata, extract_encoder_features, DATASET_LABEL_COL
)


K_VALUES = [1, 3, 5]


def extract_volume_features_all(cfg, checkpoint, metadata_path, no_checkpoint,
                                layers, device):
    """Load seg model and extract volume-level features for every volume."""
    apply_timm = cfg["model"].get("encoder_weights") is not None

    if no_checkpoint:
        model = build_model(cfg)
        print("Using untrained seg model (0% baseline)")
    else:
        cfg_copy = cfg.copy()
        cfg_copy["model"] = cfg["model"].copy()
        cfg_copy["model"]["encoder_weights"] = None
        model = build_model(cfg_copy)
        ckpt = load_checkpoint(checkpoint, model)
        print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch', '?')})")

    model = model.to(device)
    model.eval()

    metadata = load_metadata(metadata_path)

    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")
    z_range = cfg["data"].get("z_range", None)

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))

    vol_features = []
    vol_meta = []

    with torch.no_grad():
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]
            if stem not in metadata:
                warnings.warn(f"Stem '{stem}' not in metadata — skipping")
                continue
            meta = metadata[stem]

            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            bf_raw = np.load(bf_path)
            gfp_raw = np.load(os.path.join(gfp_dir, f"{stem}.npy"))
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
                gfp_raw = gfp_raw[z_lo:z_hi]

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            feats = extract_encoder_features(model, bf, device, layers,
                                             batch_size=16)
            Z = feats.shape[0]
            nonzero_prop = np.array(
                [(gfp_raw[z] > 0).mean() for z in range(Z)], dtype=np.float32)

            # Aggregate to volume level
            vol_feat = feats.mean(axis=0)
            vol_feat = np.concatenate([vol_feat, [float(nonzero_prop.mean())]])

            vol_features.append(vol_feat)
            vol_meta.append(meta)
            print(f"  {stem}: Dataset={meta['Dataset']}, D={len(vol_feat)}")

    return np.array(vol_features, dtype=np.float32), vol_meta


def knn_leave_one_out(features, labels, k_values, metric="cosine"):
    """Leave-one-out k-NN classification at multiple k values.

    For each sample, finds its nearest neighbors among all OTHER samples
    and predicts the majority label.  No train/test split needed.

    Args:
        features: (N, D) array
        labels: list of N label strings
        k_values: list of k values to evaluate
        metric: "cosine" or "euclidean"

    Returns:
        dict with results per k value, or None if <2 classes.
    """
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        print("  <2 unique classes — skipping task")
        return None

    class_counts = np.bincount(y)
    N = len(y)

    # Compute pairwise distance matrix
    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features / norms
        sim = normed @ normed.T          # (N, N) cosine similarity
        dist = 1.0 - sim
    else:
        # Euclidean
        diff = features[:, None, :] - features[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Set self-distance to infinity so a sample can't be its own neighbor
    np.fill_diagonal(dist, np.inf)

    # Nearest neighbor indices sorted by distance
    nn_indices = np.argsort(dist, axis=1)  # (N, N)

    results = {
        "n_samples": int(N),
        "n_classes": int(n_classes),
        "classes": [str(c) for c in le.classes_],
        "class_counts": [int(c) for c in class_counts],
        "metric": metric,
        "per_k": {},
    }

    max_k = max(k_values)
    for k in k_values:
        if k > N - 1:
            print(f"  k={k}: not enough samples (N={N}), skipping")
            continue

        correct = 0
        per_sample = []
        for i in range(N):
            neighbors = nn_indices[i, :k]
            neighbor_labels = y[neighbors]
            counts = Counter(neighbor_labels)
            pred = counts.most_common(1)[0][0]
            is_correct = int(pred == y[i])
            correct += is_correct
            per_sample.append({
                "true": str(le.classes_[y[i]]),
                "pred": str(le.classes_[pred]),
                "correct": is_correct,
            })

        acc = correct / N
        results["per_k"][str(k)] = {
            "k": k,
            "accuracy": float(acc),
            "n_correct": int(correct),
            "n_total": int(N),
            "per_sample": per_sample,
        }
        print(f"  k={k}: acc={acc:.3f} ({correct}/{N})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out k-NN classification on seg-encoder features")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use untrained model (0%% seg baseline)")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--seg_tag", required=True,
                        help="Label for this seg model, e.g. frac025")
    parser.add_argument("--layers", type=int, nargs="+", default=[5],
                        help="Encoder stages to extract (default: [5])")
    parser.add_argument("--k_values", type=int, nargs="+", default=K_VALUES,
                        help="k values for k-NN (default: 1 3 5)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"],
                        default="cosine",
                        help="Distance metric (default: cosine)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.no_checkpoint and args.checkpoint is None:
        parser.error("Either --checkpoint or --no_checkpoint is required")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Extracting features for seg_tag={args.seg_tag}...")
    features, vol_meta = extract_volume_features_all(
        cfg, args.checkpoint, args.metadata, args.no_checkpoint,
        args.layers, device)

    if features.size == 0 or features.ndim < 2:
        raise RuntimeError(
            f"No volumes matched metadata for seg_tag={args.seg_tag}. "
            "Check --metadata path and dataset stems."
        )

    out = {
        "seg_tag": args.seg_tag,
        "checkpoint": args.checkpoint or "none (untrained)",
        "feature_dim": int(features.shape[1]),
        "n_volumes": int(features.shape[0]),
        "tasks": {},
    }

    datasets_seen = sorted(set(m["Dataset"].lower() for m in vol_meta))
    for ds in datasets_seen:
        label_col = DATASET_LABEL_COL.get(ds)
        if label_col is None:
            warnings.warn(f"Unknown dataset '{ds}' — skipping")
            continue

        mask = np.array([m["Dataset"].lower() == ds for m in vol_meta])
        ds_features = features[mask]
        ds_labels = [vol_meta[i][label_col] for i in range(len(vol_meta))
                     if mask[i]]

        print(f"\n── Task: {ds.capitalize()} ({label_col}) — {int(mask.sum())} volumes ──")
        results = knn_leave_one_out(
            ds_features, ds_labels, args.k_values, args.metric)
        if results is not None:
            out["tasks"][ds] = {"label_col": label_col, **results}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
