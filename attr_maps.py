"""Full-volume attribution maps for a trained GFP->force model, tiled at NATIVE
resolution.

Purpose
-------
1. Reference maps for shortcut decorrelation: train_decorr_force.py penalizes a
   new model whose input-attribution correlates with these maps (the shortcut
   features of an existing model).
2. High-quality saliency rendering: unlike gradviz_force.py's downsampled full
   view, tiles are crop_size windows at native resolution — the scale the model
   was trained on — stitched into one (Z, H, W) map. No resolution mismatch, no
   aliasing.

Each output is <output_dir>/<stem>.npy, float16 in [0, 1] (per-volume max-
normalized; the decorrelation penalty is a Pearson correlation, so scale is
irrelevant). Maps live in the SAME z-cropped frame as the training datasets
(z_range applied), so a training crop at (zd, yh, xw) indexes the map directly.

Usage
-----
    python attr_maps.py -c configs/gfp_classifier_3d.yaml \
        --ckpt results/force_from_gfp_new/force_ckpt_3d.pth \
        --data_dir data_phalloidin_mhc_051826_staged \
        --stems <stemA> <stemB> --output_dir results/force_decorr/attr_ref \
        --n_samples 8 --preview 4
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.utils import set_seed
from src.models.gfp_classifier import build_gfp_classifier
from gradviz_force import load_norm_volume, smoothgrad


def tile_starts(full, win):
    """Start offsets covering [0, full) with stride=win, tail-aligned (the last
    tile ends exactly at `full`, overlapping the previous one if needed)."""
    if full <= win:
        return [0]
    starts = list(range(0, full - win + 1, win))
    if starts[-1] + win < full:
        starts.append(full - win)
    return starts


def predict_target(model, vol, dims, pd, cs, device, fixed):
    """One volume-level target class: mean logits over a coarse tile sweep
    (no_grad), argmax — so every tile attributes the SAME class."""
    if fixed is not None:
        return fixed
    z, h, w = vol.shape
    pz = max(0, pd - z) if dims == "3d" else 0
    ph, pw = max(0, cs - h), max(0, cs - w)
    if pz or ph or pw:      # small FOVs: tiles must be full crop_size (/32)
        vol = np.pad(vol, ((0, pz), (0, ph), (0, pw)), mode="reflect")
        z, h, w = vol.shape
    logits_sum, n = None, 0
    with torch.no_grad():
        for yh in tile_starts(h, cs)[:2]:
            for xw in tile_starts(w, cs)[:2]:
                if dims == "3d":
                    zd = max(0, (z - pd) // 2)
                    t = vol[zd:zd + pd, yh:yh + cs, xw:xw + cs][None, None]
                else:
                    zb = int(np.argmax(vol.reshape(z, -1).mean(1)))
                    t = vol[zb, yh:yh + cs, xw:xw + cs][None, None]
                lg = model(torch.from_numpy(np.ascontiguousarray(t))
                           .float().to(device))[0]
                logits_sum = lg if logits_sum is None else logits_sum + lg
                n += 1
    return int((logits_sum / n).argmax(dim=1).item())


def volume_attr(model, vol, dims, pd, cs, n_samples, noise, target, device,
                z_stride=1):
    """Tiled SmoothGrad over the whole (Z, H, W) volume at native resolution.
    Overlapping (tail) regions are averaged. For 2D models, z_stride>1 computes
    every k-th slice and fills the gaps with the nearest computed slice."""
    z, h, w = vol.shape
    # pad up to tile size so every dim has >= one full window
    pz, ph, pw = max(0, pd - z) if dims == "3d" else 0, \
        max(0, cs - h), max(0, cs - w)
    if pz or ph or pw:
        vol = np.pad(vol, ((0, pz), (0, ph), (0, pw)), mode="reflect")
    zp, hp, wp = vol.shape

    acc = np.zeros(vol.shape, dtype=np.float32)
    cnt = np.zeros(vol.shape, dtype=np.uint8)
    if dims == "3d":
        for zd in tile_starts(zp, pd):
            for yh in tile_starts(hp, cs):
                for xw in tile_starts(wp, cs):
                    tile = vol[zd:zd + pd, yh:yh + cs, xw:xw + cs]
                    x = torch.from_numpy(tile[None, None].copy()).float()
                    sal, _, _ = smoothgrad(model, x, target, n_samples,
                                           noise, device)
                    acc[zd:zd + pd, yh:yh + cs, xw:xw + cs] += sal
                    cnt[zd:zd + pd, yh:yh + cs, xw:xw + cs] += 1
    else:
        done = sorted(set(range(0, zp, max(1, z_stride))) | {zp - 1})
        for zi in done:
            for yh in tile_starts(hp, cs):
                for xw in tile_starts(wp, cs):
                    tile = vol[zi, yh:yh + cs, xw:xw + cs]
                    x = torch.from_numpy(tile[None, None].copy()).float()
                    sal, _, _ = smoothgrad(model, x, target, n_samples,
                                           noise, device)
                    acc[zi, yh:yh + cs, xw:xw + cs] += sal
                    cnt[zi, yh:yh + cs, xw:xw + cs] += 1
        if len(done) < zp:          # nearest-fill the skipped slices
            done_arr = np.array(done)
            for zi in range(zp):
                if zi not in done:
                    src = int(done_arr[np.abs(done_arr - zi).argmin()])
                    acc[zi] = acc[src]
                    cnt[zi] = cnt[src]
    attr = acc / np.maximum(cnt, 1)
    return attr[:z, :h, :w]        # drop the pad back off


def save_preview(gfp, attr, stem, target, out_png):
    g, a = gfp.max(axis=0), attr.max(axis=0)      # depth MIPs

    def norm01(x):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].imshow(norm01(g), cmap="gray"); ax[0].set_title("GFP (MIP)")
    ax[1].imshow(norm01(a), cmap="magma"); ax[1].set_title("attribution (MIP)")
    ax[2].imshow(norm01(g), cmap="gray")
    ax[2].imshow(norm01(a), cmap="magma", alpha=0.5); ax[2].set_title("overlay")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"{stem}  (native-res tiled SmoothGrad, class={target})")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True,
                   help="Config matching the REFERENCE model architecture")
    p.add_argument("--ckpt", required=True, help="Trained force model .pth")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--input", choices=["bf", "gfp"], default=None,
                   help="Modality (default: from ckpt, else gfp)")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Volume stems (default: every staged volume)")
    p.add_argument("--n_samples", type=int, default=8,
                   help="SmoothGrad samples per tile (8 is plenty when maps "
                        "are averaged over tiles)")
    p.add_argument("--noise_level", type=float, default=0.1)
    p.add_argument("--target", choices=["pred", "high", "low"], default="pred",
                   help="Class logit to attribute (pred = volume-level argmax)")
    p.add_argument("--z_stride", type=int, default=1,
                   help="2D models only: compute every k-th slice (nearest-fill "
                        "between) to bound cost on deep stacks")
    p.add_argument("--preview", type=int, default=4,
                   help="Save overlay PNGs for the first N volumes (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Recompute maps that already exist")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = validate_config(load_config(args.config))
    dcfg, mcfg = cfg["data"], cfg["model"]
    dims = mcfg.get("dims", "2d")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    is_reg = ckpt.get("task") == "regression" or ckpt.get("n_out") == 1
    n_bins = int(ckpt.get("n_bins", 3))
    n_out = 1 if is_reg else n_bins
    modality = args.input or ckpt.get("input", "gfp")
    model = build_gfp_classifier(cfg, n_out, 2)
    missing, _ = model.load_state_dict(
        ckpt.get("model_state_dict", ckpt), strict=False)
    if any("encoder" in m for m in missing):
        raise SystemExit("ckpt missing encoder weights — wrong config?")
    model.to(device).eval()

    stats_dir = os.path.join(args.data_dir, "stats")
    mod_dir = os.path.join(args.data_dir, modality)
    z_range = dcfg.get("z_range", None)
    apply_timm = mcfg.get("encoder_weights") is not None
    cs = dcfg.get("crop_size", 256)
    pd = dcfg.get("patch_depth", 32)
    fixed = (0 if is_reg
             else {"pred": None, "low": 0, "high": n_bins - 1}[args.target])

    stems = args.stems or sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(mod_dir, "*.npy")))
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {"ckpt": args.ckpt, "config": args.config, "dims": dims,
            "input": modality, "z_range": z_range, "crop_size": cs,
            "patch_depth": pd, "n_samples": args.n_samples,
            "noise_level": args.noise_level, "target": args.target,
            "frame": "z_range-cropped, native H/W",
            "normalized": "per-volume max -> [0,1] float16"}
    with open(os.path.join(args.output_dir, "attr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"attribution maps ({dims}, input={modality}) for {len(stems)} "
          f"volume(s) -> {args.output_dir}/")
    for i, stem in enumerate(stems):
        out_npy = os.path.join(args.output_dir, f"{stem}.npy")
        if os.path.exists(out_npy) and not args.force:
            print(f"  [{i+1}/{len(stems)}] {stem}: exists, skipping")
            continue
        path = os.path.join(mod_dir, f"{stem}.npy")
        if not os.path.exists(path):
            print(f"  [{i+1}/{len(stems)}] {stem}: missing volume, skipping")
            continue
        vol = load_norm_volume(path, stats_dir, modality, z_range, apply_timm)
        tgt = predict_target(model, vol, dims, pd, cs, device, fixed)
        attr = volume_attr(model, vol, dims, pd, cs, args.n_samples,
                           args.noise_level, tgt, device,
                           z_stride=args.z_stride)
        peak = float(attr.max())
        if peak > 0:
            attr = attr / peak
        tmp = out_npy + ".tmp.npy"
        np.save(tmp, attr.astype(np.float16))
        os.replace(tmp, out_npy)
        print(f"  [{i+1}/{len(stems)}] {stem}: class={tgt} "
              f"shape={attr.shape} -> {out_npy}")
        if i < args.preview:
            save_preview(vol, attr, stem, tgt,
                         os.path.join(args.output_dir, f"{stem}_preview.png"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("done.")


if __name__ == "__main__":
    main()
