"""Generate side-by-side prediction montages for each data fraction.

For each val volume, saves a PNG with columns: BF | GT GFP | Mask | 0% | 1% | ... | 100%
Rows are evenly-spaced Z slices. Also saves per-volume mask PNGs.

Usage:
    python compare_predictions.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --output_dir comparisons/
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint, make_train_val_split
from src.data.normalization import normalize

FRACTIONS = [
    ("frac000", "0%"),
    ("frac001", "1%"),
    ("frac025", "25%"),
    ("frac050", "50%"),
    ("frac075", "75%"),
    ("frac100", "100%"),
]

N_ZSLICES = 5  # rows in the montage


def predict_volume(model, bf, device):
    """Run 2D model on each Z slice, return (Z, H, W) predictions."""
    Z, H, W = bf.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    preds = np.zeros((Z, H, W), dtype=np.float32)

    for z in range(Z):
        slc = bf[z]
        if pad_h > 0 or pad_w > 0:
            slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")
        inp = torch.from_numpy(slc[np.newaxis, np.newaxis]).float().to(device)
        preds[z] = model(inp)[0, 0].cpu().numpy()[:H, :W]

    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Compare predictions across data fractions")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--output_dir", default="comparisons/",
                        help="Output directory for montage PNGs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_timm = cfg["model"].get("encoder_weights") is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover data + val split
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]
    _, val_stems = make_train_val_split(
        stems, val_fraction=cfg["data"].get("val_fraction", 0.15),
        seed=cfg.get("seed", 42))

    z_range = cfg["data"].get("z_range", None)
    ckpt_base = cfg["training"]["checkpoint_dir"]

    # Load all fraction models
    models = {}
    for tag, label in FRACTIONS:
        if tag == "frac000":
            # Untrained model: ImageNet encoder + random decoder
            model = build_model(cfg)
            model = model.to(device)
            model.eval()
            models[tag] = (model, label)
            print(f"Built untrained model for {label}")
            continue

        ckpt_path = os.path.join(f"{ckpt_base}_{tag}", "best.pth")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found — skipping {label}")
            continue

        cfg_copy = cfg.copy()
        cfg_copy["model"] = cfg["model"].copy()
        cfg_copy["model"]["encoder_weights"] = None
        model = build_model(cfg_copy)
        load_checkpoint(ckpt_path, model)
        model = model.to(device)
        model.eval()
        models[tag] = (model, label)
        print(f"Loaded {tag} ({label})")

    if not models:
        print("No models available.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate montages for each val volume
    with torch.no_grad():
        for stem in val_stems:
            print(f"\nProcessing {stem}...")

            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            bf_raw = np.load(os.path.join(bf_dir, f"{stem}.npy"))
            gfp_raw = np.load(os.path.join(gfp_dir, f"{stem}.npy"))

            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
                gfp_raw = gfp_raw[z_lo:z_hi]

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)
            gfp = normalize(gfp_raw, stats["gfp"]["p_low"], stats["gfp"]["p_high"],
                            apply_timm=False)

            Z = bf.shape[0]
            z_indices = np.linspace(0, Z - 1, N_ZSLICES, dtype=int)

            # Compute mask (same as eval: 1st percentile threshold)
            gfp_thresh = np.percentile(gfp_raw, 1)
            mask_vol = gfp_raw > gfp_thresh  # (Z, H, W) bool

            # Save standalone mask montage
            mask_fig, mask_axes = plt.subplots(1, N_ZSLICES,
                                               figsize=(3 * N_ZSLICES, 3))
            if N_ZSLICES == 1:
                mask_axes = [mask_axes]
            for i, zi in enumerate(z_indices):
                mask_axes[i].imshow(mask_vol[zi], cmap="gray", vmin=0, vmax=1)
                mask_axes[i].set_title(f"Z={zi}", fontsize=9)
                mask_axes[i].set_xticks([])
                mask_axes[i].set_yticks([])
            mask_fig.suptitle(f"Eval mask (GT > p1): {stem}", fontsize=11)
            mask_fig.tight_layout()
            mask_path = os.path.join(args.output_dir, f"{stem}_mask.png")
            mask_fig.savefig(mask_path, dpi=150, bbox_inches="tight")
            plt.close(mask_fig)
            print(f"  Saved {mask_path}")

            # Predict with each fraction model
            predictions = {}
            for tag, (model, label) in models.items():
                predictions[tag] = predict_volume(model, bf, device)

            # Build montage: rows=z slices, cols=BF|GT|Mask|frac000|...|frac100
            n_cols = 3 + len(models)
            fig, axes = plt.subplots(N_ZSLICES, n_cols,
                                     figsize=(3 * n_cols, 3 * N_ZSLICES))
            if N_ZSLICES == 1:
                axes = axes[np.newaxis]

            # Undo TIMM for BF display
            if apply_timm:
                from src.data.normalization import TIMM_MEAN, TIMM_STD
                bf_display = np.clip(bf * TIMM_STD + TIMM_MEAN, 0, 1)
            else:
                bf_display = np.clip(bf, 0, 1)

            for row, zi in enumerate(z_indices):
                # BF
                axes[row, 0].imshow(bf_display[zi], cmap="gray", vmin=0, vmax=1)
                if row == 0:
                    axes[row, 0].set_title("BF", fontsize=10)
                axes[row, 0].set_ylabel(f"Z={zi}", fontsize=9)

                # GT GFP
                axes[row, 1].imshow(gfp[zi], cmap="gray", vmin=0, vmax=1)
                if row == 0:
                    axes[row, 1].set_title("GT GFP", fontsize=10)

                # Mask
                axes[row, 2].imshow(mask_vol[zi], cmap="gray", vmin=0, vmax=1)
                if row == 0:
                    axes[row, 2].set_title("Mask", fontsize=10)

                # Each fraction prediction
                for col_offset, (tag, (_, label)) in enumerate(models.items()):
                    pred_slice = np.clip(predictions[tag][zi], 0, 1)
                    axes[row, 3 + col_offset].imshow(pred_slice, cmap="gray",
                                                      vmin=0, vmax=1)
                    if row == 0:
                        axes[row, 3 + col_offset].set_title(label, fontsize=10)

                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.suptitle(f"Val volume: {stem}", fontsize=13)
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"{stem}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
