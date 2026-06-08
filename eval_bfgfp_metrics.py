"""Evaluate BF->GFP regression metrics on a set of held-out volumes.

Two modes:
  A. Sidecar mode (default): reads `heldout_stems.json` next to `best.pth`
     (written by `train_fraction.py --holdout`) — used by the power-laws pipeline.
  B. Explicit mode: pass `--data_dir`, optionally `--stems "a b c"`, to evaluate
     against any arbitrary dataset (new dataset, full pool, etc.). The script
     scans `{data_dir}/bf/` for vols that also have GFP + stats, runs inference,
     and computes MAE / SSIM / Pearson per vol.

Usage:
    # A — held-out from training:
    python eval_bfgfp_metrics.py \
        --ckpt ckpts/unet_3d_imagenet_pearson_frac100_holdEx/best.pth \
        --output results/bfgfp_metrics/.../result.json

    # B — new dataset:
    python eval_bfgfp_metrics.py \
        --ckpt ckpts/unet_3d_imagenet_pearson_frac100_holdPt/best.pth \
        --data_dir data_new \
        --output results/eval_new_dataset/holdPt_frac100_3d.json
"""

import os
import json
import argparse

import numpy as np
import torch

from src.config import load_config, resolve_ckpt_config
from src.utils import prepare_env, load_checkpoint
from src.models import build_model
from src.data.normalization import normalize
from src.metrics import mae as mae_metric, ssim as ssim_metric, pearson_corr
from predict import predict_3d, predict_2d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best.pth")
    p.add_argument("--output", required=True, help="Path to output JSON")
    p.add_argument("--config", default=None,
                   help="Override config (default: <ckpt_dir>/config.yaml)")
    p.add_argument("--data_dir", default=None,
                   help="Override dataset root; expects {data_dir}/bf/, /gfp/, /stats/. "
                        "If set, bypasses the heldout_stems.json sidecar.")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Subset of vol stems to evaluate (combine with --data_dir).")
    args = p.parse_args()

    ckpt_dir = os.path.dirname(args.ckpt)
    config_path = resolve_ckpt_config(ckpt_dir, args.config)

    heldout_stems = None
    holdout = "explicit"
    fraction = None
    if args.stems:
        heldout_stems = list(args.stems)
    elif args.data_dir is None:
        sidecar_path = os.path.join(ckpt_dir, "heldout_stems.json")
        if not os.path.exists(sidecar_path):
            raise SystemExit(
                f"Missing {sidecar_path}. Either train with --holdout, or "
                "pass --data_dir / --stems explicitly.")
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        heldout_stems = sidecar["heldout_stems"]
        holdout = sidecar["holdout"]
        fraction = sidecar.get("fraction")

    cfg = load_config(config_path)
    dims = cfg["model"]["dims"]
    apply_timm = cfg["model"].get("encoder_weights") is not None

    accelerator, device, tqdm = prepare_env(mixed_precision=False)

    cfg_copy = dict(cfg)
    cfg_copy["model"] = dict(cfg["model"])
    cfg_copy["model"]["encoder_weights"] = None
    model = build_model(cfg_copy)
    ckpt = load_checkpoint(args.ckpt, model)
    accelerator.print(
        f"Loaded {args.ckpt} epoch={ckpt.get('epoch', '?')} "
        f"holdout={holdout} fraction={fraction} "
        f"data_dir={args.data_dir or cfg['data']['data_dir']}")
    model = accelerator.prepare(model)
    model.eval()

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")
    z_range = cfg["data"].get("z_range", None)

    if heldout_stems is None:
        # Auto-discover from {data_dir}/bf/ — keep only stems that also have
        # GFP + stats so we can score.
        from glob import glob as _glob
        candidates = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in _glob(os.path.join(bf_dir, "*.npy")))
        heldout_stems = [
            s for s in candidates
            if os.path.exists(os.path.join(gfp_dir, f"{s}.npy"))
            and os.path.exists(os.path.join(stats_dir, f"{s}.json"))]
        if not heldout_stems:
            raise SystemExit(
                f"No vols with BF + GFP + stats found under {data_dir}.")
        accelerator.print(
            f"Auto-discovered {len(heldout_stems)} vol(s) in {data_dir}")

    per_volume = []
    with torch.no_grad():
        for stem in tqdm(heldout_stems, desc="Eval heldout"):
            bf_path = os.path.join(bf_dir, f"{stem}.npy")
            gfp_path = os.path.join(gfp_dir, f"{stem}.npy")
            if not (os.path.exists(bf_path) and os.path.exists(gfp_path)):
                accelerator.print(f"  skip missing {stem}")
                continue
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                stats = json.load(f)

            bf_raw = np.load(bf_path)
            gfp_raw = np.load(gfp_path)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
                gfp_raw = gfp_raw[z_lo:z_hi]

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)
            gfp = normalize(gfp_raw, stats["gfp"]["p_low"], stats["gfp"]["p_high"],
                            apply_timm=False)

            if dims == "2d":
                pred = predict_2d(model, bf, device)
            else:
                patch_depth = cfg["data"].get("patch_depth", 32)
                overlap = cfg.get("inference", {}).get("overlap", [16, 128, 128])
                inf_batch = cfg.get("inference", {}).get("batch_size", 4)
                spatial_tile = cfg.get("inference", {}).get(
                    "spatial_tile", cfg["data"].get("crop_size", 256))
                pred = predict_3d(model, bf, device, patch_depth, overlap,
                                  inf_batch, spatial_tile=spatial_tile)

            pred = np.asarray(pred, dtype=np.float32)
            gfp_arr = np.asarray(gfp, dtype=np.float32)
            # Clamp to comparable range for SSIM (data_range=1.0 default)
            pred_c = np.clip(pred, 0.0, 1.0)
            gfp_c = np.clip(gfp_arr, 0.0, 1.0)
            row = {
                "stem": stem,
                "mae": float(mae_metric(pred_c, gfp_c)),
                "ssim": float(ssim_metric(pred_c, gfp_c, data_range=1.0)),
                "pearson": float(pearson_corr(pred, gfp_arr)),
            }
            per_volume.append(row)
            accelerator.print(
                f"  {stem}: mae={row['mae']:.4f} ssim={row['ssim']:.3f} "
                f"pearson={row['pearson']:.3f}")

    if not per_volume:
        raise SystemExit("No held-out volumes evaluated.")

    summary = {
        "ckpt": args.ckpt,
        "data_dir": data_dir,
        "holdout": holdout,
        "fraction": fraction,
        "n_volumes": len(per_volume),
        "per_volume": per_volume,
        "mean": {
            "mae": float(np.mean([r["mae"] for r in per_volume])),
            "ssim": float(np.mean([r["ssim"] for r in per_volume])),
            "pearson": float(np.mean([r["pearson"] for r in per_volume])),
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(
        f"Mean: mae={summary['mean']['mae']:.4f} "
        f"ssim={summary['mean']['ssim']:.3f} "
        f"pearson={summary['mean']['pearson']:.3f}")
    accelerator.print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
