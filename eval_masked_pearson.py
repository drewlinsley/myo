"""Evaluate masked Pearson correlation (within non-zero GT pixels).

Runs inference on the val set and computes Pearson correlation only within
pixels where the ground-truth GFP is non-zero. Outputs a JSON results file.

Usage:
    python eval_masked_pearson.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson_frac025/best.pth \
        --output results/frac025.json
"""

import os
import json
import argparse
import numpy as np
import torch

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint, make_train_val_split
from src.data.normalization import normalize


def masked_pearson(pred, target, mask):
    """Pearson correlation computed only over masked pixels.

    Args:
        pred, target: (H, W) numpy arrays
        mask: boolean (H, W) array — True where GT is non-zero

    Returns:
        Pearson r (float), or NaN if mask is too small
    """
    p = pred[mask]
    t = target[mask]
    if len(p) < 2 or p.std() == 0 or t.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate masked Pearson correlation on val set")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <ckpt_dir>/masked_pearson.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dims = cfg["model"]["dims"]
    assert dims == "2d", "Only 2D models supported for now"
    apply_timm = cfg["model"].get("encoder_weights") is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    cfg_copy = cfg.copy()
    cfg_copy["model"] = cfg["model"].copy()
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)

    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?')})")

    model = model.to(device)
    model.eval()

    # Discover data
    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]

    # Get val stems (same split as training)
    _, val_stems = make_train_val_split(
        stems, val_fraction=cfg["data"].get("val_fraction", 0.15),
        seed=cfg.get("seed", 42))

    print(f"Evaluating on {len(val_stems)} val volumes")

    z_range = cfg["data"].get("z_range", None)
    all_pearsons = []
    per_volume = {}

    with torch.no_grad():
        for stem in val_stems:
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

            Z, H, W = bf.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32

            vol_pearsons = []
            for z in range(Z):
                slc = bf[z]
                if pad_h > 0 or pad_w > 0:
                    slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode="reflect")

                inp = torch.from_numpy(slc[np.newaxis, np.newaxis]).float().to(device)
                pred = model(inp)[0, 0].cpu().numpy()[:H, :W]

                gt = gfp[z]
                # Mask on raw GFP (before normalization) to find true signal pixels
                mask = gfp_raw[z] > 0
                if mask.sum() < 2:
                    continue

                r = masked_pearson(pred, gt, mask)
                if not np.isnan(r):
                    vol_pearsons.append(r)
                    all_pearsons.append(r)

            vol_mean = float(np.mean(vol_pearsons)) if vol_pearsons else float("nan")
            per_volume[stem] = {
                "mean_pearson": vol_mean,
                "n_slices": len(vol_pearsons),
            }
            print(f"  {stem}: masked Pearson = {vol_mean:.4f} ({len(vol_pearsons)} slices)")

    overall = float(np.mean(all_pearsons)) if all_pearsons else float("nan")
    print(f"\nOverall masked Pearson: {overall:.4f} ({len(all_pearsons)} total slices)")

    results = {
        "overall_masked_pearson": overall,
        "n_slices": len(all_pearsons),
        "n_volumes": len(val_stems),
        "per_volume": per_volume,
        "checkpoint": args.checkpoint,
    }

    out_path = args.output
    if out_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        out_path = os.path.join(ckpt_dir, "masked_pearson.json")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
