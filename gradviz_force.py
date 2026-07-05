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
from src.models.gfp_classifier import build_gfp_classifier


def load_norm_volume(path, stats_dir, modality, z_range, apply_timm):
    """Mirror VolumeRegressionDataset._load: crop z_range, percentile-normalize."""
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(stats_dir, f"{stem}.json")) as f:
        st = json.load(f)
    raw = np.load(path)
    if z_range is not None:
        z_lo = max(0, z_range[0])
        z_hi = min(raw.shape[0], z_range[1])
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


def overlay_panel(gfp2d, sal2d, title, path, classes, tc, logits, class_rep):
    """3-panel: GFP, saliency, overlay. gfp2d/sal2d are 2D arrays."""
    def norm01(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    g = norm01(gfp2d)
    s = norm01(sal2d)
    probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
    e_force = float(np.dot(probs, class_rep)) if class_rep is not None else float("nan")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title("GFP (normalized)")
    ax[1].imshow(s, cmap="magma"); ax[1].set_title("SmoothGrad |saliency|")
    ax[2].imshow(g, cmap="gray")
    ax[2].imshow(s, cmap="magma", alpha=0.5)
    ax[2].set_title("overlay")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    pstr = ", ".join(f"{classes[i]}={probs[i]:.2f}" for i in range(len(classes)))
    fig.suptitle(f"{title}\npredicted={classes[tc]}  probs[{pstr}]  "
                 f"E[force]={e_force:.2f}", fontsize=11)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = validate_config(load_config(args.config))
    dcfg, mcfg = cfg["data"], cfg["model"]
    dims = mcfg.get("dims", "2d")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_bins = int(ckpt.get("n_bins", 3))
    classes = ckpt.get("classes") or [f"q{i}" for i in range(n_bins)]
    class_rep = np.array(ckpt["class_rep_force"], dtype=np.float64) \
        if ckpt.get("class_rep_force") is not None else None
    modality = args.input or ckpt.get("input", "gfp")
    state = ckpt.get("model_state_dict", ckpt)

    model = build_gfp_classifier(cfg, n_bins, 2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any("encoder" in m for m in missing):
        raise SystemExit(f"ckpt is missing encoder weights ({len(missing)} keys) — "
                         "wrong config for this checkpoint?")
    model.to(device).eval()

    target_class = {"pred": None, "low": 0, "high": n_bins - 1}[args.target]

    stats_dir = os.path.join(args.data_dir, "stats")
    mod_dir = os.path.join(args.data_dir, modality)
    z_range = dcfg.get("z_range", None)
    apply_timm = mcfg.get("encoder_weights") is not None
    cs = dcfg.get("crop_size", 256)
    pd = dcfg.get("patch_depth", 32)

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

        if dims == "2d":
            if args.all_slices:
                sals, gfps, tcs, logit_last = [], [], [], None
                for z in range(vol.shape[0]):
                    slc = _center_crop2d(vol[z], cs)
                    x = torch.from_numpy(slc[None, None]).float()
                    sal, tc, lg = smoothgrad(model, x, target_class,
                                             args.n_samples, args.noise_level, device)
                    sals.append(sal); gfps.append(slc); tcs.append(tc); logit_last = lg
                sal = np.mean(sals, axis=0)
                gfp2d = np.mean(gfps, axis=0)
                tc = int(np.bincount(tcs).argmax())
                logits = logit_last
            else:
                zbest = int(np.argmax(vol.reshape(vol.shape[0], -1).mean(1)))
                slc = _center_crop2d(vol[zbest], cs)
                x = torch.from_numpy(slc[None, None]).float()
                sal, tc, logits = smoothgrad(model, x, target_class,
                                             args.n_samples, args.noise_level, device)
                gfp2d = slc
            title = f"{stem}  (2D, slice={'mean' if args.all_slices else 'max-signal'})"
        else:
            patch = _center_patch3d(vol, pd, cs)                 # (D,H,W)
            x = torch.from_numpy(patch[None, None]).float()      # (1,1,D,H,W)
            sal3d, tc, logits = smoothgrad(model, x, target_class,
                                           args.n_samples, args.noise_level, device)
            sal = sal3d.max(axis=0)                              # depth MIP of saliency
            gfp2d = patch.max(axis=0)                            # depth MIP of GFP
            title = f"{stem}  (3D, center patch {pd}x{cs}x{cs}, depth-MIP)"

        overlay_panel(gfp2d, sal, title, out_png, classes, tc, logits, class_rep)
        print(f"  {stem}: pred={classes[tc]} -> {out_png}")

    print(f"Done. saliency maps in {args.output_dir}/")


if __name__ == "__main__":
    main()
