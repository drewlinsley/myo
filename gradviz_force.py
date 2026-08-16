"""SmoothGrad saliency for the GFP->force classifier: which parts of a GFP
volume drive the model's force-class prediction.

SmoothGrad (Smilkov et al. 2017): average the input-gradient of a target class
logit over many small-noise copies of the input, so the raw (noisy) gradient is
denoised into a stable saliency map.

    saliency(x) = mean_i | d logit_c / d (x + N(0, sigma)) |_i ,   i = 1..n_samples

The model is the one trained by train_split_force_classifier.py (or the LOO
trainer) with --save_ckpt: a GFPTwoHeadClassifier whose EXERCISE head carries the
force-class logits. We differentiate the target class logit (predicted class by
default) w.r.t. the normalized GFP input, exactly the preprocessing used in
training (z_range crop + percentile normalize + optional timm standardization).

2D model -> saliency on the highest-signal Z-slice (or --all_slices, averaged).
3D model -> saliency on a center patch (patch_depth x crop x crop); shown as a
depth max-projection.

Usage
-----
    python gradviz_force.py \
        -c configs/gfp_classifier_3d.yaml \
        --ckpt results/force_from_gfp_new/ckpt_3d.pth \
        --data_dir data_phalloidin_mhc_051826_staged \
        --stems <stemA> <stemB> --n_samples 25 --output_dir results/force_from_gfp_new/saliency_3d
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
from src.data.normalization import normalize
from src.data.zband import resolve_z_range
from src.models.gfp_classifier import build_gfp_classifier

MODALITY = "gfp"   # set in main(); used for panel labels


def load_norm_volume(path, stats_dir, modality, z_range, apply_timm):
    """Mirror VolumeRegressionDataset._load: crop z_range (fixed [lo, hi] or
    per-volume 'auto' band from the stats JSON), percentile-normalize."""
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(stats_dir, f"{stem}.json")) as f:
        st = json.load(f)
    raw = np.load(path)
    if z_range is not None:
        z_lo, z_hi = resolve_z_range(z_range, st, raw.shape[0])
        raw = raw[z_lo:z_hi]
    img = normalize(raw, st[modality]["p_low"], st[modality]["p_high"],
                    apply_timm=apply_timm)
    return img  # (Z, H, W) float32, normalized


def _center_crop2d(a, cs):
    h, w = a.shape[-2:]
    ph, pw = max(0, cs - h), max(0, cs - w)
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="reflect")
    h, w = a.shape[-2:]
    y0, x0 = (h - cs) // 2, (w - cs) // 2
    return a[y0:y0 + cs, x0:x0 + cs]


def _center_patch3d(vol, pd, cs):
    z, h, w = vol.shape
    pz, ph, pw = max(0, pd - z), max(0, cs - h), max(0, cs - w)
    if pz or ph or pw:
        vol = np.pad(vol, ((0, pz), (0, ph), (0, pw)), mode="reflect")
    z, h, w = vol.shape
    z0, y0, x0 = (z - pd) // 2, (h - cs) // 2, (w - cs) // 2
    return vol[z0:z0 + pd, y0:y0 + cs, x0:x0 + cs]


def _pad_mult(a, m=32):
    """Reflect-pad the last two dims up to a multiple of m (the encoder's stride),
    so the WHOLE plane can be run without cropping to a fixed window."""
    h, w = a.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        pad = [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)]
        a = np.pad(a, pad, mode="reflect")
    return a


def _fit_plane(a, max_hw):
    """Block-average-downsample the last two dims so max(H, W) <= max_hw. Keeps
    the WHOLE plane in view but bounds memory (a full-res forward+backward
    through the encoder OOMs — the model only ever saw crop_size windows).
    Averaging (not stride-decimating) is anti-aliased: naive a[::f, ::f] folds
    the high-frequency myotube texture into aliasing noise, which then shows up
    directly in the saliency. Returns (a at reduced resolution, the factor)."""
    hw = max(a.shape[-2:])
    if not max_hw or hw <= max_hw:
        return a, 1
    f = int(np.ceil(hw / max_hw))
    h, w = a.shape[-2:]
    th, tw = h - h % f, w - w % f
    a = a[..., :th, :tw]
    sh = a.shape[:-2] + (th // f, f, tw // f, f)
    return a.reshape(sh).mean(axis=(-3, -1), dtype=np.float32), f


def _center_depth(vol, pd):
    """Center pd-slice sub-stack (reflect-pad z if the stack is shallower)."""
    z = vol.shape[0]
    if z < pd:
        vol = np.pad(vol, ((0, pd - z), (0, 0), (0, 0)), mode="reflect")
        z = pd
    z0 = (z - pd) // 2
    return vol[z0:z0 + pd]


def smoothgrad(model, x, target_class, n_samples, noise_level, device):
    """x: (1,1,...) tensor already on device. Returns saliency np array shaped
    like x[0,0] and the clean-input logits (np)."""
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        clean_logits = model(x)[0]                       # exercise head
    if target_class is None:
        target_class = int(clean_logits.argmax(dim=1).item())
    rng = float((x.max() - x.min()).item())
    sigma = noise_level * (rng if rng > 0 else 1.0)
    grad_accum = torch.zeros_like(x)
    for _ in range(n_samples):
        noisy = (x + torch.randn_like(x) * sigma).detach().requires_grad_(True)
        logits = model(noisy)[0]
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        if noisy.grad is not None:
            noisy.grad = None
        score.backward()
        grad_accum += noisy.grad.detach()
    sal = (grad_accum / n_samples).abs()[0, 0].cpu().numpy()  # drop batch,chan
    return sal, target_class, clean_logits.detach().cpu().numpy()[0]


def overlay_panel(gfp2d, sal2d, title, subtitle, path):
    """3-panel: GFP, saliency, overlay. gfp2d/sal2d are 2D arrays."""
    def norm01(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    g = norm01(gfp2d)
    s = norm01(sal2d)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title(f"{MODALITY.upper()} (normalized)")
    ax[1].imshow(s, cmap="magma"); ax[1].set_title("SmoothGrad |saliency|")
    ax[2].imshow(g, cmap="gray")
    ax[2].imshow(s, cmap="magma", alpha=0.5)
    ax[2].set_title("overlay")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--ckpt", required=True, help="Saved force classifier .pth")
    p.add_argument("--data_dir", required=True, help="Staged root (<input>/ + stats/)")
    p.add_argument("--input", choices=["bf", "gfp"], default=None,
                   help="Modality fed to the model (default: from ckpt, else gfp)")
    p.add_argument("--stems", nargs="*", default=None,
                   help="Volume stems to visualize (default: first --limit found)")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=25)
    p.add_argument("--noise_level", type=float, default=0.15,
                   help="SmoothGrad sigma as a fraction of the input value range")
    p.add_argument("--target", choices=["pred", "high", "low"], default="pred",
                   help="Which force class logit to attribute (pred=argmax)")
    p.add_argument("--all_slices", action="store_true",
                   help="2D: average saliency over all Z-slices (else max-signal slice)")
    p.add_argument("--view", choices=["full", "crop"], default="full",
                   help="full = render the WHOLE H×W plane (padded to /32); "
                        "crop = a single center crop_size window (default: full)")
    p.add_argument("--max_hw", type=int, default=None,
                   help="full-view only: cap plane H/W (downsample above it) to "
                        "avoid OOM. Default 1024 (2D) / 512 (3D); 0 = no cap.")
    p.add_argument("--sal_smooth", type=float, default=1.0,
                   help="Gaussian sigma (px) applied to the final saliency map. "
                        "Raw |grad| of a strided CNN checkerboards at the pixel "
                        "level; a ~1-2 px blur shows the structure. 0 = off.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = validate_config(load_config(args.config))
    dcfg, mcfg = cfg["data"], cfg["model"]
    dims = mcfg.get("dims", "2d")

    global MODALITY
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    is_reg = ckpt.get("task") == "regression" or ckpt.get("n_out") == 1
    n_bins = int(ckpt.get("n_bins", 3))
    n_out = 1 if is_reg else n_bins
    classes = ckpt.get("classes") or [f"q{i}" for i in range(n_bins)]
    class_rep = np.array(ckpt["class_rep_force"], dtype=np.float64) \
        if ckpt.get("class_rep_force") is not None else None
    std = ckpt.get("standardize") or {"mu": 0.0, "sd": 1.0}   # regression only
    modality = args.input or ckpt.get("input", "gfp")
    MODALITY = modality
    state = ckpt.get("model_state_dict", ckpt)

    model = build_gfp_classifier(cfg, n_out, 2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any("encoder" in m for m in missing):
        raise SystemExit(f"ckpt is missing encoder weights ({len(missing)} keys) — "
                         "wrong config for this checkpoint?")
    model.to(device).eval()

    # regression -> attribute the single scalar output; classification -> a class.
    target_class = 0 if is_reg else {"pred": None, "low": 0,
                                     "high": n_bins - 1}[args.target]
    print(f"model: {'REGRESSION' if is_reg else f'{n_bins}-class'} "
          f"({dims}, input={modality})")

    stats_dir = os.path.join(args.data_dir, "stats")
    mod_dir = os.path.join(args.data_dir, modality)
    z_range = dcfg.get("z_range", None)
    apply_timm = mcfg.get("encoder_weights") is not None
    cs = dcfg.get("crop_size", 256)
    pd = dcfg.get("patch_depth", 32)
    max_hw = args.max_hw if args.max_hw is not None else (512 if dims == "3d" else 1024)

    if args.stems:
        stems = list(args.stems)
    else:
        stems = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob(os.path.join(mod_dir, "*.npy")))[:args.limit]
    if not stems:
        raise SystemExit(f"No volumes found under {mod_dir}/")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"SmoothGrad ({dims}, {modality}) on {len(stems)} vol(s); "
          f"n_samples={args.n_samples} noise={args.noise_level} target={args.target}")

    for stem in stems:
        path = os.path.join(mod_dir, f"{stem}.npy")
        if not os.path.exists(path):
            print(f"  skip {stem}: missing {path}"); continue
        vol = load_norm_volume(path, stats_dir, modality, z_range,
                               apply_timm)  # (Z,H,W)
        out_png = os.path.join(args.output_dir, f"{stem}.png")

        # full view: downsample above max_hw (bounds memory), then pad to /32.
        prep2d = ((lambda s: _pad_mult(_fit_plane(s, max_hw)[0])) if args.view == "full"
                  else (lambda s: _center_crop2d(s, cs)))
        ds = _fit_plane(vol[0] if dims == "2d" else vol, max_hw)[1] \
            if args.view == "full" else 1
        if dims == "2d":
            if args.all_slices:
                sals, gfps, tcs, logit_last = [], [], [], None
                for z in range(vol.shape[0]):
                    slc = prep2d(vol[z])
                    x = torch.from_numpy(slc[None, None].copy()).float()
                    sal, tc, lg = smoothgrad(model, x, target_class,
                                             args.n_samples, args.noise_level, device)
                    sals.append(sal); gfps.append(slc); tcs.append(tc); logit_last = lg
                sal = np.mean(sals, axis=0)
                gfp2d = np.mean(gfps, axis=0)
                tc = int(np.bincount(tcs).argmax())
                logits = logit_last
            else:
                zbest = int(np.argmax(vol.reshape(vol.shape[0], -1).mean(1)))
                slc = prep2d(vol[zbest])
                x = torch.from_numpy(slc[None, None].copy()).float()
                sal, tc, logits = smoothgrad(model, x, target_class,
                                             args.n_samples, args.noise_level, device)
                gfp2d = slc
            vw = (f"full-plane{f' /{ds}' if ds > 1 else ''}"
                  if args.view == "full" else f"{cs}px crop")
            title = (f"{stem}  (2D, {vw}, "
                     f"slice={'mean' if args.all_slices else 'max-signal'})")
        else:
            if args.view == "full":
                fit, _ = _fit_plane(vol, max_hw)                 # downsample H,W
                patch = _pad_mult(_center_depth(fit, pd))        # (D,fullH,fullW)
                vw = (f"{pd}×{patch.shape[1]}×{patch.shape[2]} full-plane"
                      f"{f' /{ds}' if ds > 1 else ''}")
            else:
                patch = _center_patch3d(vol, pd, cs)             # (D,cs,cs)
                vw = f"center patch {pd}x{cs}x{cs}"
            x = torch.from_numpy(patch[None, None].copy()).float()  # (1,1,D,H,W)
            sal3d, tc, logits = smoothgrad(model, x, target_class,
                                           args.n_samples, args.noise_level, device)
            sal = sal3d.max(axis=0)                              # depth MIP of saliency
            gfp2d = patch.max(axis=0)                            # depth MIP of GFP
            title = f"{stem}  (3D, {vw}, depth-MIP)"

        if args.sal_smooth > 0:
            from scipy.ndimage import gaussian_filter
            sal = gaussian_filter(sal, sigma=args.sal_smooth)

        if is_reg:
            pred_force = float(logits[0]) * std["sd"] + std["mu"]
            subtitle = f"predicted force = {pred_force:.2f}"
            tag = f"force={pred_force:.2f}"
        else:
            probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
            e_force = (float(np.dot(probs, class_rep))
                       if class_rep is not None else float("nan"))
            pstr = ", ".join(f"{classes[i]}={probs[i]:.2f}"
                             for i in range(len(classes)))
            subtitle = f"predicted={classes[tc]}  probs[{pstr}]  E[force]={e_force:.2f}"
            tag = f"pred={classes[tc]}"
        overlay_panel(gfp2d, sal, title, subtitle, out_png)
        print(f"  {stem}: {tag} -> {out_png}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. saliency maps in {args.output_dir}/")


if __name__ == "__main__":
    main()
