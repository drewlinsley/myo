#!/usr/bin/env python
"""Is the brightfield foreground mask right-side-up?

The mask that everything downstream calls "foreground" is

    mask = bf_raw > threshold          (src/data/foreground_mask.py)

i.e. foreground is ASSUMED to be the brighter side of the brightfield
histogram. In transmitted-light brightfield that assumption is exactly
backwards whenever the empty background is the unobstructed illumination
(bright) and the tissue is the darker, scattering part. If that is the case
here, then every quantity named "fg" in this pipeline -- view_fg_frac, the
--fg_min view filter, the fgmean weighting, patch_mean_fg itself -- has been
selecting and weighting BACKGROUND.

This script settles it with the one signal that is unambiguous about where
tissue is: GFP. The phalloidin/MHC fluorescence is on the tissue by
construction, so

    mean GFP inside the mask  >  mean GFP outside   ->  mask is right
    mean GFP inside the mask  <  mean GFP outside   ->  mask is INVERTED

Two passes:

  1. cache-only (instant): the extractor stored view_fg_frac and
     view_mean_int (mean GFP per view) in every .npz. Their per-volume rank
     correlation says whether "more foreground" views are GFP-brighter or
     GFP-dimmer. Negative = inverted. This uses the EXACT masks the features
     were built with, via their per-view summaries.

  2. direct (seconds per volume): rebuild the mask on a few volumes, measure
     mean GFP inside/outside, and render BF / GFP / both mask polarities side
     by side so the verdict can be checked by eye.

Usage
  python check_mask_polarity.py --data_dir data_phalloidin_mhc_051826_staged \
      [--feature_dir results/dino_features/gfp_tiled_volume_<hash>] \
      [--n_volumes 6] [--out results/mask_check]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _by_path(mod_file, name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "data", mod_file)
    spec = importlib.util.spec_from_file_location(f"_chk_{name}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_zb = _by_path("zband.py", "zband")
resolve_z_range = _zb.resolve_z_range


def _rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    return r


def spearman(a, b):
    a, b = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def bf_threshold(band, method):
    """The same threshold the extractor computes, with a numpy fallback.

    threshold_li is what the extraction runs used (bf/li). If skimage is not
    installed wherever this check runs, fall back to Otsu implemented here --
    the POLARITY verdict does not depend on which histogram-split method chose
    the cut point, only on which side of it the tissue sits.
    """
    try:
        from skimage.filters import (threshold_li, threshold_otsu,
                                     threshold_triangle, threshold_minimum)
        fn = {"li": threshold_li, "otsu": threshold_otsu,
              "triangle": threshold_triangle,
              "minimum": threshold_minimum}[method]
        return float(fn(band.ravel().astype(np.float64))), method
    except ImportError:
        v = band.ravel().astype(np.float64)
        lo, hi = v.min(), v.max()
        hist, edges = np.histogram(v, bins=256, range=(lo, hi))
        mids = 0.5 * (edges[1:] + edges[:-1])
        w = hist.cumsum().astype(np.float64)
        m = (hist * mids).cumsum()
        w0, w1 = w[:-1], w[-1] - w[:-1]
        ok = (w0 > 0) & (w1 > 0)
        mu0 = np.where(ok, m[:-1] / np.maximum(w0, 1), 0)
        mu1 = np.where(ok, (m[-1] - m[:-1]) / np.maximum(w1, 1), 0)
        var = np.where(ok, w0 * w1 * (mu0 - mu1) ** 2, -1)
        return float(mids[int(np.argmax(var))]), "otsu(numpy fallback)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--feature_dir", default=None,
                    help="a cached DINO feature dir; default: the single "
                         "results/dino_features/gfp_tiled_* match, if any")
    ap.add_argument("--mask_source", default="bf")
    ap.add_argument("--modality", default="gfp",
                    help="the fluorescence channel used as tissue ground truth")
    ap.add_argument("--mask_method", default="li")
    ap.add_argument("--n_volumes", type=int, default=6)
    ap.add_argument("--z_stride", type=int, default=None,
                    help="default: the feature manifest's, else 1")
    ap.add_argument("--out", default="results/mask_check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    verdicts = {}

    # ---- pass 1: the cached per-view summaries --------------------------
    fdir = args.feature_dir
    if fdir is None:
        cands = sorted(glob.glob("results/dino_features/gfp_tiled_*"))
        fdir = cands[0] if len(cands) == 1 else None
        if len(cands) > 1:
            print(f"NOTE: {len(cands)} feature dirs match; pass --feature_dir "
                  f"to run pass 1 on one of them")
    man = {}
    if fdir and os.path.isdir(fdir):
        mp = os.path.join(fdir, "manifest.json")
        man = json.load(open(mp)) if os.path.exists(mp) else {}
        rhos = []
        for p in sorted(glob.glob(os.path.join(fdir, "*.npz"))):
            z = np.load(p)
            if "view_fg_frac" not in z or "view_mean_int" not in z:
                continue
            fg = np.asarray(z["view_fg_frac"], float)
            it = np.asarray(z["view_mean_int"], float)
            if len(fg) >= 8 and np.ptp(fg) > 1e-6:
                rhos.append(spearman(fg, it))
        rhos = np.asarray([r for r in rhos if np.isfinite(r)])
        if len(rhos):
            med = float(np.median(rhos))
            neg = int((rhos < 0).sum())
            print(f"pass 1  cached features ({os.path.basename(fdir)}):")
            print(f"  per-volume spearman(view_fg_frac, mean {args.modality} "
                  f"intensity): median {med:+.3f}  "
                  f"[IQR {np.percentile(rhos,25):+.3f}..{np.percentile(rhos,75):+.3f}]"
                  f"  negative in {neg}/{len(rhos)} volumes")
            print(f"  reading: 'foreground' views should be the {args.modality}"
                  f"-bright ones. A negative median means the stored fg is "
                  f"tracking {args.modality}-DIM views -> inverted.")
            verdicts["pass1_median_rho"] = med
            verdicts["pass1_n_volumes"] = int(len(rhos))
        else:
            print("pass 1  skipped: cached npz lack view_fg_frac/view_mean_int")
    else:
        print("pass 1  skipped: no cached feature dir")

    # ---- pass 2: rebuild the mask, judge it against GFP -----------------
    mod_dir = os.path.join(args.data_dir, args.modality)
    msk_dir = os.path.join(args.data_dir, args.mask_source)
    stats_dir = os.path.join(args.data_dir, "stats")
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(mod_dir, "*.npy")))
    if not stems:
        raise SystemExit(f"no volumes in {mod_dir}/")
    picks = stems[::max(1, len(stems) // max(1, args.n_volumes))][:args.n_volumes]
    z_stride = (args.z_stride if args.z_stride is not None
                else int(man.get("z_stride", 1)))

    print(f"\npass 2  rebuilding the {args.mask_source}/{args.mask_method} "
          f"mask on {len(picks)} volume(s), z_stride={z_stride}:")
    rows, ratios = [], []
    for stem in picks:
        st = json.load(open(os.path.join(stats_dir, f"{stem}.json")))
        bf = np.load(os.path.join(msk_dir, f"{stem}.npy"), mmap_mode="r")
        gf = np.load(os.path.join(mod_dir, f"{stem}.npy"), mmap_mode="r")
        n_z = bf.shape[0]
        z_lo, z_hi = resolve_z_range("auto", st, n_z)
        bfb = np.asarray(bf[z_lo:z_hi:max(1, z_stride)], dtype=np.float64)
        gfb = np.asarray(gf[z_lo:z_hi:max(1, z_stride)], dtype=np.float64)
        thr, used = bf_threshold(bfb, args.mask_method)
        bright = bfb > thr                       # the pipeline's current mask
        if not bright.any() or bright.all():
            print(f"  {stem}: degenerate threshold, skipped")
            continue
        g_in = float(gfb[bright].mean())         # GFP where mask says tissue
        g_out = float(gfb[~bright].mean())       # GFP where mask says empty
        ratio = g_in / max(g_out, 1e-9)
        ratios.append(ratio)
        ok = ratio > 1.0
        print(f"  {stem}: {args.modality} inside mask {g_in:8.1f}   "
              f"outside {g_out:8.1f}   ratio {ratio:5.2f}   "
              f"mask covers {bright.mean():.0%}   "
              f"[{used} thr={thr:.0f}]  -> "
              f"{'ok' if ok else 'INVERTED'}")
        zi = bfb.shape[0] // 2
        rows.append((stem, bfb[zi], gfb[zi], bright[zi], ratio))

    if not rows:
        raise SystemExit("no volume produced a usable mask")

    med_ratio = float(np.median(ratios))
    inverted = med_ratio < 1.0
    verdicts.update({"pass2_median_gfp_ratio": med_ratio,
                     "pass2_n_volumes": len(ratios),
                     "inverted": bool(inverted)})

    # ---- the figure ------------------------------------------------------
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.1 * n), squeeze=False)
    for i, (stem, bfs, gfs, mk, ratio) in enumerate(rows):
        blo, bhi = np.percentile(bfs, [1, 99])
        glo, ghi = np.percentile(gfs, [1, 99.5])
        axes[i][0].imshow(bfs, cmap="gray", vmin=blo, vmax=bhi)
        axes[i][0].set_ylabel(f"{stem}\nGFP in/out = {ratio:.2f}", fontsize=7)
        axes[i][1].imshow(gfs, cmap="gray", vmin=glo, vmax=ghi)
        # SHADE the selected region rather than outlining it: a mask and its
        # complement share the same boundary curve, so contours alone make the
        # two columns identical.
        red = np.zeros(mk.shape + (4,), np.float32); red[mk] = (1, 0, 0, 0.35)
        grn = np.zeros(mk.shape + (4,), np.float32); grn[~mk] = (0, 1, 0, 0.35)
        axes[i][2].imshow(gfs, cmap="gray", vmin=glo, vmax=ghi)
        axes[i][2].imshow(red)
        axes[i][3].imshow(gfs, cmap="gray", vmin=glo, vmax=ghi)
        axes[i][3].imshow(grn)
        for j, t in enumerate([
                "brightfield (mask is built from this)",
                "GFP (tissue ground truth)",
                "CURRENT mask: bf > threshold (red fill)",
                "flipped mask: bf < threshold (green fill)"]):
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(t, fontsize=9)
    v = ("INVERTED: the current mask sits on the GFP-dim side -- "
         "'foreground' has been selecting background"
         if inverted else
         "polarity ok: the current mask sits on the GFP-bright side")
    fig.suptitle(f"Brightfield mask polarity -- {v}\n"
                 f"(median GFP inside/outside the current mask: "
                 f"{med_ratio:.2f}; right column shows the flip)",
                 fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(args.out, "mask_polarity.png"), dpi=170,
                bbox_inches="tight")
    print(f"\n  saved {os.path.join(args.out, 'mask_polarity.png')}")
    json.dump(verdicts, open(os.path.join(args.out,
                                          "mask_polarity.json"), "w"),
              indent=2)

    print()
    if inverted:
        print("VERDICT: INVERTED (median GFP inside/outside = "
              f"{med_ratio:.2f} < 1).")
        print("  Every 'fg' quantity has been reversed: view_fg_frac counted")
        print("  background, --fg_min kept the most-background views and")
        print("  dropped tissue-rich ones, fgmean weighted views by how much")
        print("  background they contain, and patch_mean_fg pooled background")
        print("  tokens. To fix, re-extract and re-run:")
        print()
        print("    rm -rf results/dino_features results/dino_sweep")
        print("    MASK_POLARITY=dark bash scripts/dino_force_sweep.sh")
        print("    MASK_POLARITY=dark SKIP_SWEEP=1 bash scripts/explain_force.sh")
        print()
        print("  NOTE: any FG_MIN tuned against the inverted mask is tuned to")
        print("  BACKGROUND fraction. The sweep default is calibrated for the")
        print("  corrected polarity; if you overrode it, re-derive it from the")
        print("  view fg distribution the probe prints.")
        print()
        print("  (results so far were all null, so no conclusion flips --")
        print("   but every number was computed on background-weighted")
        print("   features and must be regenerated before being shown.)")
    else:
        print(f"VERDICT: polarity ok (median GFP inside/outside = "
              f"{med_ratio:.2f} > 1). The red/blue attribution pattern is")
        print("  then a property of the readout, not the mask -- the model")
        print("  ranks tissue-poor regions higher. That is worth its own")
        print("  look, but it is not a masking bug.")


if __name__ == "__main__":
    main()
