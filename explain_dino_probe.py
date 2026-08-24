#!/usr/bin/env python
"""What does the frozen-DINOv2 force probe actually read off the image?

The probe is a chain of affine maps -- weighted standardization, a weighted PCA
rotation, and a linear model -- so the whole readout collapses to one direction
`u` in DINOv2 feature space:

    prediction(volume) = u . x + const

`probe_force_features.py` writes that direction to a `<config>.readout.npz`
sidecar. Because the pooling that produced `x` is itself a weighted mean, this
attribution is EXACT rather than approximate:

    x   = sum_v  a_v e_v          a_v = view weight  (fgmean over tiles/slices)
    e_v = sum_p  b_p t_p          b_p = patch weight (foreground mask)

    => u . x = sum_v sum_p  a_v b_p (u . t_p)

so the prediction is a weighted average of a single scalar field over patch
tokens, s(p) = u . t_p. Every pixel's contribution literally sums to the
number the model produced. This is why no saliency/gradient method is used
here: for a linear readout on frozen features, gradients would be a strictly
worse estimate of a quantity available in closed form.

Two levels:
  --level view   uses the cached .npz features. No GPU. One value per
                 (z-slice, tile) -- coarse, but immediate.
  --level patch  re-runs DINOv2 on the selected volumes to recover the 37x37
                 token grid per tile. Dense maps, patch montages, and the
                 descriptor regression that puts the readout into words.

Reading the output honestly
---------------------------
Attribution explains a MODEL, never the biology, and it looks equally
convincing whether or not the model works. On this dataset the probe's best
config sits at spearman ~ +0.40 against a permutation null spanning about
[-0.55, +0.63], i.e. inside chance. So this script always renders the same
analysis for a readout fit on SHUFFLED labels (`--permuted_readout`) beside the
real one. If the two look alike, the maps are describing the dataset's texture
statistics, not force -- and that comparison is the point of showing them.

Usage
  # coarse, no GPU
  python explain_dino_probe.py --readout results/dino_sweep/<t>/<cfg>.readout.npz \
      --feature_dir results/dino_features/gfp_tiled_volume_<hash> \
      --data_dir data_phalloidin_mhc_051826_staged --level view --out results/xai

  # dense, needs the GPU
  python explain_dino_probe.py --readout ... --feature_dir ... --data_dir ... \
      --level patch --n_volumes 6 \
      --permuted_readout results/dino_sweep/<t>/shuffled_control.readout.npz
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import these BY PATH. src/data/__init__.py imports torch, and the
# view-level mode here is deliberately torch-free so it runs anywhere the
# cached features do -- same reason probe_force_features.py does this. The
# torch import happens only in the --level patch branch.
def _by_path(mod_file, name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "data", mod_file)
    spec = importlib.util.spec_from_file_location(f"_xai_{name}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_dv = _by_path("dino_views.py", "dino_views")
_zb = _by_path("zband.py", "zband")
_nm = _by_path("normalization.py", "normalization")
_fg = _by_path("foreground_mask.py", "foreground_mask")
plan_views, make_view = _dv.plan_views, _dv.make_view
area_resize = _dv.area_resize
resolve_z_range = _zb.resolve_z_range
normalize, global_percentiles = _nm.normalize, _nm.global_percentiles

TILE = 518


# --------------------------------------------------------------------------
# the readout
# --------------------------------------------------------------------------
def load_readout(path):
    z = np.load(path, allow_pickle=True)
    u = np.asarray(z["mean_direction"], dtype=np.float64)
    meta = {k: (z[k].item() if z[k].shape == () else z[k])
            for k in z.files if k not in ("mean_direction", "fold_directions",
                                          "feature_names")}
    return u, meta


def decomposable_block(u, token, dim):
    """The leading slice of `u` that distributes over views and patches.

    Only a MEAN-pooled token block decomposes: `patch_mean_fg` is a weighted
    mean over patch tokens, so u restricted to it distributes exactly. A
    `patch_std_fg` block, or the `+std` half of an `fgmean+std` aggregation, is
    a dispersion term -- quadratic in the tokens -- and there is no honest way
    to hand a single patch its share of a variance. Those dimensions are
    reported as unexplained rather than quietly attributed.

    Returns (u_block, fraction_of_squared_norm_covered).
    """
    toks = [t.strip() for t in str(token).split(",") if t.strip()]
    lead_is_mean = toks and toks[0] in ("patch_mean", "patch_mean_fg", "cls")
    if not lead_is_mean or u.size < dim:
        return None, 0.0
    ub = u[:dim]
    frac = float(ub @ ub) / float(u @ u) if u @ u > 0 else 0.0
    return ub, frac


# --------------------------------------------------------------------------
# view-level attribution, straight from the cached features
# --------------------------------------------------------------------------
def view_contributions(npz_path, u_block, token, fg_min):
    """Per-view contribution a_v * (u . e_v), matching the probe's pooling.

    The weights here MUST match load_dino_features(): views below `fg_min` are
    dropped outright (not down-weighted), and the survivors are weighted by
    foreground fraction. Using a different weighting would produce a map of a
    model nobody fit.
    """
    z = np.load(npz_path)
    tok0 = [t.strip() for t in str(token).split(",") if t.strip()][0]
    if tok0 not in z:
        raise SystemExit(f"{npz_path}: no '{tok0}' array (has {list(z.keys())})")
    e = np.asarray(z[tok0], dtype=np.float64)              # (n_views, D)
    fg = (np.asarray(z["view_fg_frac"], float) if "view_fg_frac" in z
          else np.ones(len(e)))
    keep = fg >= fg_min if fg_min > 0 else np.ones(len(e), bool)
    if not keep.any():
        return None
    w = np.clip(fg[keep], 0, None)
    if w.sum() <= 1e-8:
        return None
    a = w / w.sum()
    score = e[keep] @ u_block                              # u . e_v
    return {"contrib": a * score, "score": score, "weight": a,
            "z": np.asarray(z["view_z"])[keep],
            "yx": np.asarray(z["view_yx"])[keep],
            "fg": fg[keep], "total": float((a * score).sum())}


def paint_view_map(vc, H, W):
    """Tile contributions -> an (H, W) field that ADDS UP to the prediction.

    Deliberately no averaging. Contributions are summed into each tile's
    footprint, so dividing this map by the tile area and integrating over the
    field returns u . x exactly -- the number the model produced. The tile
    grid overlaps (tile_starts spaces 3 x 518 tiles over 1100 px), and in the
    overlap the two views genuinely both contribute, so they add rather than
    average. Averaging by coverage count, which is the obvious thing to do,
    breaks that identity and quietly rescales the overlap bands.

    Units: contribution per tile-sized region. Summed over z, because every
    z-slice of a tile is a separate view and all of them are pooled.
    """
    acc = np.zeros((H, W), np.float64)
    for c, (y, x) in zip(vc["contrib"], vc["yx"]):
        y, x = int(y), int(x)
        acc[y:min(y + TILE, H), x:min(x + TILE, W)] += c
    return acc


# --------------------------------------------------------------------------
# patch-level attribution: re-run the backbone
# --------------------------------------------------------------------------
def encode_slice_tokens(ctx, sl, specs, p_low, p_high, device, batch_size=8):
    """Patch tokens for every tile of one z-slice: (n_tiles, gh, gw, D).

    Everything downstream -- the readout response, the DINO foreground mask,
    the patch montage -- is a projection of these, so they are computed once
    per slice rather than once per quantity. One slice of a 4x3 tiling is
    12 x 37 x 37 x 768 float32 = 50 MB, which is why slices are streamed
    instead of held.
    """
    import torch
    views = [make_view(sl, sp, p_low, p_high, ctx["mean"], ctx["std"],
                       normalize) for sp in specs]
    n_prefix, patch = ctx["n_prefix"], ctx["patch"]
    out, gh, gw = [], None, None
    for i in range(0, len(views), batch_size):
        x = torch.from_numpy(np.stack(views[i:i + batch_size])).to(device)
        with torch.no_grad():
            tok = ctx["model"].forward_features(x).float()
        pt = tok[:, n_prefix:]                              # (B, gh*gw, D)
        gh, gw = x.shape[-2] // patch, x.shape[-1] // patch
        out.append(pt.reshape(pt.shape[0], gh, gw, pt.shape[-1])
                     .cpu().numpy().astype(np.float32))
    return np.concatenate(out), gh, gw


def fit_dino_foreground(ctx, band, zs, specs, p_low, p_high, device,
                        batch_size=8, n_slices=6):
    """A foreground mask read out of the DINOv2 tokens themselves.

    The first principal component of DINOv2 patch tokens separates object from
    background -- that is the standard published behaviour of these features,
    and it is attractive here because it lives in the SAME representation the
    probe reads. A brightfield-derived mask has to be geometrically aligned to
    the tiles and can disagree with what the tokens encode; PC1 cannot.

    PC1's sign is arbitrary, so it is fixed by data rather than assumed: the
    side whose patches are brighter on average is called foreground. The split
    point is Otsu on the PC1 scores rather than zero, because the two
    populations are not balanced and the centered zero-crossing sits wherever
    the foreground fraction puts it.

    Scope: fitted PER VOLUME, on a subsample of slices. That makes it adaptive,
    which is right for looking at one volume and WRONG for rebuilding features
    across the dataset -- a per-volume cutoff makes foreground fractions
    incomparable between volumes, and if tissue coverage tracks force, that
    bias tracks the label. Use it to inspect and to attribute; do not
    regenerate `patch_mean_fg` with it without moving the fit to a
    dataset-wide sample.

    Returns (pc1, threshold, sign) with mask = sign * (tokens @ pc1) > threshold.
    """
    picks = list(range(0, len(zs), max(1, len(zs) // max(1, n_slices))))[:n_slices]
    toks, ints = [], []
    for zi in picks:
        t, gh, gw = encode_slice_tokens(ctx, band[zi], specs, p_low, p_high,
                                        device, batch_size)
        toks.append(t.reshape(-1, t.shape[-1]))
        for sp in specs:
            tile = band[zi][sp["y"]:sp["y"] + TILE, sp["x"]:sp["x"] + TILE]
            ints.append(patch_descriptors(tile, ctx["patch"])["intensity"]
                        .ravel())
    T = np.concatenate(toks).astype(np.float64)
    I = np.concatenate(ints).astype(np.float64)
    T = T - T.mean(axis=0)
    # PC1 by power iteration on the Gram side: 768 dims, ~50k rows.
    _u, _s, vt = np.linalg.svd(T[np.random.default_rng(0).choice(
        len(T), size=min(len(T), 20000), replace=False)], full_matrices=False)
    pc1 = vt[0]
    proj = T @ pc1
    sign = 1.0 if np.corrcoef(proj, I)[0, 1] > 0 else -1.0
    thr = _otsu(sign * proj)
    return pc1, float(thr), float(sign)


def _otsu(v, bins=256):
    """Otsu split point on a 1-D score."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if not np.isfinite(lo) or hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    w = hist.cumsum().astype(np.float64)
    mids = 0.5 * (edges[1:] + edges[:-1])
    m = (hist * mids).cumsum()
    tot_w, tot_m = w[-1], m[-1]
    w0 = w[:-1]
    w1 = tot_w - w0
    ok = (w0 > 0) & (w1 > 0)
    if not ok.any():
        return float(np.median(v))
    mu0 = np.where(ok, m[:-1] / np.maximum(w0, 1), 0.0)
    mu1 = np.where(ok, (tot_m - m[:-1]) / np.maximum(w1, 1), 0.0)
    var = w0 * w1 * (mu0 - mu1) ** 2
    var[~ok] = -1.0
    return float(mids[int(np.argmax(var))])


def resolve_mask_threshold(man, data_dir, stems, feature_dir):
    """The ONE intensity cutoff the cached features were masked with.

    The probe weights patch tokens by the brightfield foreground mask, so any
    faithful per-patch attribution needs the same mask. The extractor computes
    a single dataset-wide threshold (so `view_fg_frac` means the same thing on
    every volume) but older runs did not record it, so recompute and cache it
    next to the features rather than silently using a per-volume threshold,
    which would shift the mask relative to the one the features were built
    with.
    """
    cfg = man.get("config", {})
    if man.get("mask_threshold") is not None:
        return float(man["mask_threshold"]), cfg
    cache = os.path.join(feature_dir, "mask_threshold.json")
    if os.path.exists(cache):
        return float(json.load(open(cache))["threshold"]), cfg
    if cfg.get("mask_scope") != "global":
        return None, cfg          # per-volume thresholds: recomputed below
    # Mirrors extract_dino_features.global_mask_threshold -- median of the
    # per-volume thresholds over a sample, giving ONE physical cutoff. Kept
    # here rather than imported because that module pulls in torch through
    # src.data, and everything except the DINOv2 forward pass in this script
    # runs without it. If that function changes, change this with it.
    src = cfg.get("mask_source", man.get("mask_source", "bf"))
    method = cfg.get("mask_method", man.get("mask_method", "li"))
    zr = man.get("z_range", "auto")
    zr = None if zr in (None, "none", "") else zr
    if zr not in (None, "auto"):
        zr = [int(v) for v in str(zr).split(",")]
    z_stride = int(man.get("z_stride", 1))
    ordered = sorted(stems)
    picks = ordered[::max(1, len(ordered) // 12)][:12]
    vals = []
    for stem in picks:
        mpath = os.path.join(data_dir, src, f"{stem}.npy")
        spath = os.path.join(data_dir, "stats", f"{stem}.json")
        if not (os.path.exists(mpath) and os.path.exists(spath)):
            continue
        st = json.load(open(spath))
        mv = np.load(mpath, mmap_mode="r")
        z_lo, z_hi = ((0, mv.shape[0]) if zr is None
                      else resolve_z_range(zr, st, mv.shape[0]))
        try:
            vals.append(_fg._threshold(
                np.asarray(mv[z_lo:z_hi][::max(1, z_stride)]), method))
        except Exception:
            continue
    thr = float(np.median(vals)) if vals else None
    if thr is not None:
        try:
            json.dump({"threshold": float(thr), "note":
                       "recomputed by explain_dino_probe.py; matches the "
                       "extractor's global_mask_threshold"}, open(cache, "w"))
        except OSError:
            pass
    return (None if thr is None else float(thr)), cfg


def load_mask_band(data_dir, cfg, man, stem, z_lo, z_hi, z_stride, threshold,
                   expect_shape):
    """The foreground mask over the same z-band the features used."""
    src = cfg.get("mask_source", man.get("mask_source", "bf"))
    if src in (None, "none"):
        return None
    mpath = os.path.join(data_dir, src, f"{stem}.npy")
    if not os.path.exists(mpath):
        return None
    mv = np.load(mpath, mmap_mode="r")
    if tuple(mv.shape) != tuple(expect_shape):
        raise SystemExit(
            f"{stem}: mask source '{src}' has shape {mv.shape} but the input "
            f"volume has {tuple(expect_shape)}. The tile geometry comes from "
            f"the input, so a mismatch silently misaligns every mask.")
    raw = np.asarray(mv[z_lo:z_hi][::max(1, z_stride)])
    dilate = int(cfg.get("mask_dilate", 0) or 0)
    min_frac = float(cfg.get("mask_min_frac", 0.0) or 0.0)
    # Same two branches as extract_dino_features.foreground_mask.
    if threshold is not None:
        band = raw > threshold
        if dilate or min_frac > 0:
            band = _fg._cleanup_per_slice(band, dilate, min_frac)
        return band
    return _fg.compute_bf_foreground_mask(
        raw, method=cfg.get("mask_method", man.get("mask_method", "li")),
        dilate=dilate, min_component_frac=min_frac)


def token_weights(mask_tile, g):
    """Mask -> per-token weights, exactly as encode_views built them.

    area_resize is a box filter, so a token's weight is the FRACTION of its
    14x14 pixel footprint that is foreground. A token entirely on background
    gets weight 0 and therefore contributes nothing -- which is the whole
    point: without this the attribution map paints a response onto pixels the
    model provably never pooled.
    """
    if mask_tile is None:
        return None
    w = np.clip(area_resize(mask_tile.astype(np.float32), g, g), 0, None)
    tot = float(w.sum())
    if tot <= 1e-6:
        # encode_views falls back to the unweighted mean for an empty view.
        return np.full((g, g), 1.0 / (g * g), dtype=np.float64)
    return (w / tot).astype(np.float64)


def patch_descriptors(tile, patch):
    """Interpretable per-patch image statistics, on the raw tile.

    These are the vocabulary the readout gets translated into: if s(p) is well
    predicted by them, the model's "features" have a plain-language name. If it
    is not, the honest statement is that the readout uses something these
    descriptors do not capture -- not that it uses nothing.

      intensity   mean signal
      contrast    within-patch std
      edge        mean gradient magnitude
      coherence   structure-tensor anisotropy in [0, 1]: 1 = every gradient in
                  the patch points the same way (aligned fibers), 0 = isotropic
    """
    t = np.asarray(tile, dtype=np.float64)
    gy, gx = np.gradient(t)
    ph, pw = t.shape[0] // patch, t.shape[1] // patch
    t = t[:ph * patch, :pw * patch]
    gy, gx = gy[:ph * patch, :pw * patch], gx[:ph * patch, :pw * patch]

    def blk(a):
        return a.reshape(ph, patch, pw, patch).transpose(0, 2, 1, 3
                                                         ).reshape(ph, pw, -1)
    tb, gyb, gxb = blk(t), blk(gy), blk(gx)
    jxx = (gxb ** 2).mean(-1)
    jyy = (gyb ** 2).mean(-1)
    jxy = (gxb * gyb).mean(-1)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-12)
    return {"intensity": tb.mean(-1), "contrast": tb.std(-1),
            "edge": np.sqrt(gxb ** 2 + gyb ** 2).mean(-1),
            "coherence": coh}


def describe_readout(S, D, fg):
    """Regress the patch score on the descriptors -> a plain-language summary.

    Weighted by foreground so background patches, which the probe down-weights
    to nothing, do not set the story. Reports each descriptor's own weighted
    correlation with s(p) AND the joint R^2, because the descriptors are
    themselves correlated (bright patches are also high-contrast).
    """
    names = list(D.keys())
    X = np.stack([D[k].ravel() for k in names], 1).astype(np.float64)
    y = S.ravel().astype(np.float64)
    w = np.clip(fg.ravel(), 0, None).astype(np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(1) & (w > 0)
    X, y, w = X[ok], y[ok], w[ok]
    if len(y) < 32:
        return None
    w = w / w.sum()

    def wstd(a):
        mu = (a * w[:, None]).sum(0) if a.ndim > 1 else float((a * w).sum())
        sd = np.sqrt(((a - mu) ** 2 * (w[:, None] if a.ndim > 1
                                       else w)).sum(0))
        return mu, np.where(np.asarray(sd) < 1e-12, 1.0, sd)

    mx, sx = wstd(X)
    my, sy = wstd(y)
    Xs, ys = (X - mx) / sx, (y - my) / sy
    marg = {n: float((Xs[:, i] * ys * w).sum()) for i, n in enumerate(names)}
    A = (Xs * w[:, None]).T @ Xs
    b = (Xs * w[:, None]).T @ ys
    beta = np.linalg.lstsq(A + 1e-8 * np.eye(len(names)), b, rcond=None)[0]
    resid = ys - Xs @ beta
    r2 = float(1.0 - (resid ** 2 * w).sum())
    return {"marginal_r": marg,
            "joint_beta": {n: float(v) for n, v in zip(names, beta)},
            "joint_r2": r2, "n_patches": int(len(y))}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _sym(a):
    v = float(np.nanpercentile(np.abs(a), 99)) or 1.0
    return -v, v


def fig_overlays(items, out_path, title, unit="contribution"):
    """Image + attribution, one row per volume, ordered by predicted force.

    Two color scales, deliberately:

      "contribution"  ONE scale shared by every row. Volumes differ mostly by
                      an overall offset -- that offset IS the prediction -- and
                      a per-row scale would renormalize it away, making a
                      weakly-scoring volume look as emphatic as a strong one.
      "within volume" each row's contribution minus its own mean, on its own
                      scale. This is the part that answers "which regions",
                      and it is only readable once the offset is removed.

    Showing only one of the two misleads in one direction or the other.
    """
    n = len(items)
    allmaps = np.concatenate([it["attr"].ravel() for it in items])
    gvmin, gvmax = _sym(allmaps)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.3 * n), squeeze=False)
    for i, it in enumerate(items):
        img, amap = it["image"], it["attr"]
        lo, hi = np.percentile(img, [1, 99.5])
        dev = amap - float(np.nanmean(amap))
        dmin, dmax = _sym(dev)
        axes[i][0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[i][0].set_ylabel(f"{it['stem']}\npred {it['pred']:+.3f}",
                              fontsize=7)
        axes[i][1].imshow(amap, cmap="coolwarm", vmin=gvmin, vmax=gvmax)
        axes[i][2].imshow(dev, cmap="coolwarm", vmin=dmin, vmax=dmax)
        axes[i][3].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[i][3].imshow(dev, cmap="coolwarm", vmin=dmin, vmax=dmax,
                          alpha=0.55)
        for j, t in enumerate(["signal (max projection)",
                               f"{unit} (shared scale +/-{gvmax:.2g})",
                               "within volume (own scale)",
                               "overlay of within-volume"]):
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(t, fontsize=9)
    fig.suptitle(title + "   (red = pushes prediction up, blue = down)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_extremes(top, bot, out_path, title, k=24):
    """Montage of the highest- and lowest-scoring image patches."""
    fig, axes = plt.subplots(2, k // 2, figsize=(k // 2 * 1.05, 2.6),
                             squeeze=False)
    for row, (bank, name) in enumerate(((top, "highest"), (bot, "lowest"))):
        for j in range(k // 2):
            ax = axes[row][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(bank):
                ax.imshow(bank[j]["patch"], cmap="gray")
                ax.set_xlabel(f"{bank[j]['score']:+.1f}", fontsize=5.5)
            else:
                ax.set_axis_off()
        axes[row][0].set_ylabel(name, fontsize=8)
    fig.suptitle(title + ":  patches the readout scores highest (top) "
                 "and lowest (bottom)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_masks(items, out_path, mask_mode, fg_min):
    """What the model was actually allowed to look at.

    Four columns per volume, because "the background is driving it" and "the
    mask is broken" are different claims and only this separates them:

      GFP / BF     the two channels, so a region that is dark in GFP but real
                   tissue in brightfield is visible as such. The mask is built
                   from BF; judging it against the GFP projection alone will
                   call correct foreground "background".
      mask         the actual weights, projected over z
      tiles        the 4x3 grid, each tile labeled with how many of its
                   z-slices cleared --fg_min. A tile at 0 contributed nothing
                   to the prediction at all.
    """
    from matplotlib.patches import Rectangle
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.3 * n), squeeze=False)
    for i, it in enumerate(items):
        img, msk = it["image"], it.get("mask")
        lo, hi = np.percentile(img, [1, 99.5])
        axes[i][0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[i][0].set_ylabel(f"{it['stem']}\npred {it['pred']:+.3f}",
                              fontsize=7)
        if it.get("bf") is not None:
            b = it["bf"]
            blo, bhi = np.percentile(b, [1, 99.5])
            axes[i][1].imshow(b, cmap="gray", vmin=blo, vmax=bhi)
        else:
            axes[i][1].text(0.5, 0.5, "no brightfield volume", ha="center",
                            va="center", transform=axes[i][1].transAxes,
                            fontsize=8)
        if msk is not None:
            frac = float((msk > 0.5).mean())
            axes[i][2].imshow(msk, cmap="viridis", vmin=0, vmax=1)
            axes[i][2].set_xlabel(f"foreground {frac:.0%} of field", fontsize=7)
        axes[i][3].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        if msk is not None:
            axes[i][3].contour(msk, levels=[0.5], colors="lime",
                               linewidths=0.7)
        nz = max(1, it.get("n_z", 1))
        for (y, x, kept) in it.get("tiles", []):
            r = kept / nz
            axes[i][3].add_patch(Rectangle(
                (x, y), TILE, TILE, fill=False, lw=1.1,
                edgecolor=("tab:red" if r == 0 else
                           "tab:orange" if r < 0.5 else "white"),
                linestyle="--" if r == 0 else "-"))
            axes[i][3].text(x + 12, y + 46, f"{kept}/{nz}", fontsize=6,
                            color=("tab:red" if r == 0 else "white"))
        for j, t in enumerate([
                "GFP (max projection)",
                "brightfield -- the mask is built from THIS",
                f"foreground mask ({mask_mode})",
                f"mask outline + tiles clearing fg_min={fg_min}"]):
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(t, fontsize=9)
    fig.suptitle("What the model was allowed to look at", fontsize=11,
                 fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_descriptors(desc, desc_ctrl, out_path):
    names = list(desc["marginal_r"].keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, [desc["marginal_r"][n] for n in names], 0.4,
           label=f"real labels (joint R2 = {desc['joint_r2']:.2f})",
           color="tab:red")
    if desc_ctrl:
        ax.bar(x + 0.2, [desc_ctrl["marginal_r"][n] for n in names], 0.4,
               label=f"shuffled labels (joint R2 = {desc_ctrl['joint_r2']:.2f})",
               color="0.7")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("weighted correlation with patch contribution")
    ax.set_title("What the readout direction tracks, in image terms",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved {out_path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True,
                    help="<config>.readout.npz from probe_force_features.py")
    ap.add_argument("--permuted_readout", default=None,
                    help="the shuffled-label control's readout. Strongly "
                         "recommended: it is the only way to see whether these "
                         "maps mean anything.")
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--modality", default="gfp")
    ap.add_argument("--level", choices=["view", "patch"], default="view")
    ap.add_argument("--mask_mode", choices=["bf", "dino", "none"],
                    default="bf",
                    help="which foreground mask weights the patch "
                         "tokens in --level patch. 'bf' rebuilds the "
                         "brightfield mask the features were extracted "
                         "with (faithful to the fitted model). 'dino' "
                         "derives it from PC1 of the patch tokens. "
                         "'none' shows the unweighted response and "
                         "WILL paint background the model never pooled.")
    ap.add_argument("--n_volumes", type=int, default=6,
                    help="how many volumes to render (half highest-predicted, "
                         "half lowest)")
    ap.add_argument("--n_patch_examples", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="results/xai")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    u, meta = load_readout(args.readout)
    token = str(meta.get("token", "patch_mean_fg"))
    fg_min = float(meta.get("fg_min", 0.0))
    target = str(meta.get("target_col", "force"))

    man_path = os.path.join(args.feature_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    dim = int(man.get("dim", 768))
    ub, frac = decomposable_block(u, token, dim)
    if ub is None:
        raise SystemExit(
            f"--readout was fit on token='{token}', whose leading block is not "
            f"a mean-pooled one. Per-view/per-patch attribution only "
            f"decomposes over a mean. Re-run the probe with "
            f"--token patch_mean_fg (that is also the best config in the "
            f"sweep) and explain that model.")
    print(f"readout: {os.path.basename(args.readout)}")
    print(f"  target={target} token={token} fg_min={fg_min} dim={dim}")
    print(f"  attributable share of the readout: {frac:.1%} of ||u||^2"
          + ("" if frac > 0.98 else
             "  <-- the rest lives in a dispersion (std) block, which is "
             "quadratic in the tokens and has no per-patch decomposition"))
    fc = meta.get("fold_cosine")
    if fc is not None and np.isfinite(float(fc)):
        fc = float(fc)
        print(f"  readout stability across LOO folds: cosine {fc:+.3f}"
              + ("" if fc >= 0.9 else
                 "  <-- UNSTABLE. These maps describe one resample."))

    ub_ctrl = None
    if args.permuted_readout:
        uc, mc = load_readout(args.permuted_readout)
        ub_ctrl, _ = decomposable_block(uc, str(mc.get("token", token)), dim)
        if ub_ctrl is None:
            print("  WARNING: permuted readout is not decomposable; "
                  "control panels will be skipped")

    # ---- rank volumes by their own predicted score ----
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in sorted(glob.glob(os.path.join(args.feature_dir, "*.npz")))]
    stems = [s for s in stems if s != "manifest"]
    if not stems:
        raise SystemExit(f"no cached features in {args.feature_dir}")
    scored = []
    for s in stems:
        vc = view_contributions(os.path.join(args.feature_dir, f"{s}.npz"),
                                ub, token, fg_min)
        if vc is not None:
            scored.append((s, vc))
    if not scored:
        raise SystemExit(f"no volume kept a view at fg_min={fg_min}")
    scored.sort(key=lambda t: t[1]["total"])
    k = max(1, args.n_volumes // 2)
    picks = scored[:k] + scored[-k:]
    print(f"  {len(scored)} volumes scored; rendering {len(picks)} "
          f"({k} lowest, {k} highest predicted)")

    mod_dir = os.path.join(args.data_dir, args.modality)
    stats_dir = os.path.join(args.data_dir, "stats")
    z_range = man.get("z_range", "auto")
    z_stride = int(man.get("z_stride", 3))
    gpct = (global_percentiles(stats_dir, args.modality)
            if man.get("norm_scope") == "global" else None)

    mask_thr, mcfg, mask_projs = None, {}, {}
    if args.level == "patch" and args.mask_mode == "bf":
        mask_thr, mcfg = resolve_mask_threshold(man, args.data_dir, stems,
                                                args.feature_dir)
        print(f"  mask: {mcfg.get('mask_source','bf')}/"
              f"{mcfg.get('mask_method','li')} "
              f"scope={mcfg.get('mask_scope','global')} "
              f"threshold={'per-volume' if mask_thr is None else f'{mask_thr:.1f}'}")
    elif args.level == "patch":
        print(f"  mask: {args.mask_mode}")

    ctx = device = None
    if args.level == "patch":
        import torch
        from extract_dino_features import build_dino
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("WARNING: no CUDA -- --level patch will be very slow")
        ctx = build_dino(str(man.get("model", meta.get("model", ""))) or
                         "vit_base_patch14_reg4_dinov2.lvd142m", device)

    items, items_ctrl = [], []
    banks = {"top": [], "bot": []}
    D_acc, S_acc, F_acc = [], [], []
    S_ctrl_acc = []

    for stem, vc in picks:
        vol = np.load(os.path.join(mod_dir, f"{stem}.npy"), mmap_mode="r")
        st = json.load(open(os.path.join(stats_dir, f"{stem}.json")))
        n_z, H, W = vol.shape
        zr = None if z_range in (None, "none", "") else z_range
        if zr not in (None, "auto"):
            zr = [int(v) for v in str(zr).split(",")]
        z_lo, z_hi = (0, n_z) if zr is None else resolve_z_range(zr, st, n_z)
        zs = list(range(z_lo, z_hi, max(1, z_stride)))
        p_low, p_high = (gpct if gpct else
                         (float(st[args.modality]["p_low"]),
                          float(st[args.modality]["p_high"])))
        band = np.asarray(vol[z_lo:z_hi:max(1, z_stride)])
        proj = band.max(axis=0)

        if args.level == "view":
            items.append({"stem": stem, "pred": vc["total"],
                          "image": proj, "attr": paint_view_map(vc, H, W)})
            continue

        # ---- dense ----
        gx, gy = man.get("tile_grid", [4, 3])
        specs = plan_views(H, W, "tiled", tile_size=TILE, tile_grid=(gx, gy))
        amap = np.zeros((H, W), np.float64)
        mproj = np.zeros((H, W), np.float32)
        # How much each view was actually pooled, straight from the cached
        # features. A view below --fg_min contributed NOTHING to the
        # prediction, so painting its response would show the model reading a
        # region it never saw. Those get weight 0 and stay blank.
        vw = {(int(z), int(y), int(x)): float(w) for z, (y, x), w
              in zip(vc["z"], vc["yx"], vc["weight"])}

        # The mask that decides which tokens the prediction is built from.
        bf_band = None
        pc1 = pc_thr = pc_sign = None
        if args.mask_mode == "bf":
            bf_band = load_mask_band(args.data_dir, mcfg, man, stem, z_lo,
                                     z_hi, z_stride, mask_thr, vol.shape)
            if bf_band is None:
                print(f"    {stem}: no brightfield mask available; "
                      f"falling back to --mask_mode none")
        elif args.mask_mode == "dino":
            pc1, pc_thr, pc_sign = fit_dino_foreground(
                ctx, band, zs, specs, p_low, p_high, device, args.batch_size)

        for zi, _z in enumerate(zs):
            sl = band[zi]
            toks, gh, gw = encode_slice_tokens(ctx, sl, specs, p_low, p_high,
                                               device, args.batch_size)
            grids = toks @ ub                                # (n_tiles, gh, gw)
            gctrl = toks @ ub_ctrl if ub_ctrl is not None else None
            for ti, sp in enumerate(specs):
                y, x = sp["y"], sp["x"]
                tile = sl[y:y + TILE, x:x + TILE]
                if tile.shape != (TILE, TILE):
                    continue

                # b_p: the within-view token weights. THIS is what was missing
                # before -- without it the map paints a response onto
                # background tokens whose pooling weight is exactly zero, which
                # reads as "the model is driven by background" when the model
                # never saw those tokens at all.
                if bf_band is not None:
                    mtile = bf_band[zi][y:y + TILE, x:x + TILE]
                elif pc1 is not None:
                    mtile = np.kron(
                        (pc_sign * (toks[ti] @ pc1) > pc_thr).astype(np.float32),
                        np.ones((ctx["patch"], ctx["patch"]), np.float32))
                else:
                    mtile = None
                b = token_weights(mtile, gh)
                if mtile is not None:
                    np.maximum(mproj[y:y + TILE, x:x + TILE],
                               np.asarray(mtile, np.float32),
                               out=mproj[y:y + TILE, x:x + TILE])

                a_v = vw.get((int(_z), int(y), int(x)), 0.0)
                if a_v > 0:
                    # Exact per-patch contribution: a_v * b_p * (u . t_p).
                    # Summed over patches this is a_v * (u . patch_mean_fg),
                    # i.e. the view's contribution, so the whole map integrates
                    # to the prediction the same way the view-level one does.
                    contrib = grids[ti] * (b if b is not None
                                           else 1.0 / (gh * gw)) * a_v
                    up = np.kron(contrib, np.ones((ctx["patch"], ctx["patch"])))
                    hh = min(up.shape[0], H - y); ww = min(up.shape[1], W - x)
                    amap[y:y + hh, x:x + ww] += up[:hh, :ww]

                # descriptors + extreme-patch reservoir, middle slice only:
                # every slice would multiply the work for a montage that only
                # needs a few dozen tiles.
                if zi == len(zs) // 2:
                    d = patch_descriptors(tile, ctx["patch"])
                    D_acc.append(d)
                    S_acc.append(grids[ti])
                    # Vote weight for the descriptor regression: the SAME token
                    # weights the model pools with, so background patches get
                    # no say -- previously this was an intensity percentile
                    # stand-in, which let the excluded region shape the story.
                    F_acc.append(b if b is not None
                                 else np.full((gh, gw), 1.0 / (gh * gw)))
                    if gctrl is not None:
                        S_ctrl_acc.append(gctrl[ti])
                    # Rank patches the model actually pooled. An unweighted
                    # argsort surfaces the most extreme BACKGROUND patches,
                    # which is exactly the artifact under investigation.
                    g_ = np.where((b if b is not None else 1.0) > 0,
                                  grids[ti], np.nan)
                    flat = g_.ravel()
                    order = np.argsort(np.where(np.isnan(flat), -np.inf, flat))
                    valid = int(np.isfinite(flat).sum())
                    for idx in order[max(0, valid - 4):valid]:
                        i0, j0 = divmod(int(idx), g_.shape[1])
                        banks["top"].append({
                            "score": float(flat[idx]),
                            "patch": tile[i0 * ctx["patch"]:(i0 + 1) * ctx["patch"],
                                          j0 * ctx["patch"]:(j0 + 1) * ctx["patch"]]})
                    for idx in order[:min(4, valid)]:
                        i0, j0 = divmod(int(idx), g_.shape[1])
                        banks["bot"].append({
                            "score": float(flat[idx]),
                            "patch": tile[i0 * ctx["patch"]:(i0 + 1) * ctx["patch"],
                                          j0 * ctx["patch"]:(j0 + 1) * ctx["patch"]]})
        mask_projs[stem] = mproj
        kept_by_tile = {}
        for zz, (yy, xx) in zip(vc["z"], vc["yx"]):
            kept_by_tile[(int(yy), int(xx))] = \
                kept_by_tile.get((int(yy), int(xx)), 0) + 1
        bf_path = os.path.join(args.data_dir,
                               mcfg.get("mask_source", "bf"), f"{stem}.npy")
        bf_proj = None
        if os.path.exists(bf_path):
            _bv = np.load(bf_path, mmap_mode="r")
            if tuple(_bv.shape) == tuple(vol.shape):
                bf_proj = np.asarray(_bv[z_lo:z_hi:max(1, z_stride)]).max(axis=0)
        items.append({"stem": stem, "pred": vc["total"], "image": proj,
                      "attr": amap, "mask": mproj, "bf": bf_proj,
                      "tiles": [(sp["y"], sp["x"],
                                 kept_by_tile.get((sp["y"], sp["x"]), 0))
                                for sp in specs],
                      "n_z": len(zs)})

    tag = os.path.basename(args.readout).replace(".readout.npz", "")
    if args.level == "view":
        # Exact: these values sum to the prediction.
        unit = "contribution per tile-sized region"
        sub = "tile-level contributions -- these add up to the prediction"
    elif args.mask_mode == "none":
        unit = "unweighted readout response"
        sub = ("patch-level response with NO token weighting -- shows regions "
               "the model never pooled")
    elif args.mask_mode == "bf":
        # Exact: a_v * b_p * (u . t_p) sums to the prediction, same as the
        # view-level map, because b_p is the weight the features were built
        # with.
        unit = "contribution per pixel"
        sub = ("patch-level contributions (brightfield mask, as fitted) -- "
               "these add up to the prediction")
    else:
        # A DIFFERENT mask from the one the features were pooled with, so the
        # map answers "what would this readout draw from under a token-derived
        # mask", not "what did it draw from". It does not sum to the fitted
        # prediction and must not be presented as if it did.
        unit = "contribution per pixel (DINO mask)"
        sub = ("patch-level contributions under a token-derived mask -- NOT "
               "the mask the features were pooled with")
    fig_overlays(items, os.path.join(args.out, f"attr_{tag}_{args.level}.png"),
                 f"{target}: {sub}", unit=unit)

    summary = {"readout": args.readout, "target": target, "token": token,
               "level": args.level, "attributable_norm_fraction": frac,
               "fold_cosine": (float(fc) if fc is not None else None),
               "volumes": [{"stem": s, "pred": v["total"]} for s, v in scored]}

    if args.level == "patch":
        fig_masks(items, os.path.join(args.out, f"masks_{tag}.png"),
                  args.mask_mode, fg_min)
        summary["mask_mode"] = args.mask_mode
        summary["mask_foreground_fraction"] = {
            it["stem"]: float((it["mask"] > 0.5).mean())
            for it in items if it.get("mask") is not None}
        summary["tiles_clearing_fg_min"] = {
            it["stem"]: [int(k) for _y, _x, k in it.get("tiles", [])]
            for it in items}

    if args.level == "patch" and D_acc:
        keys = list(D_acc[0].keys())
        D = {k: np.concatenate([d[k].ravel() for d in D_acc]) for k in keys}
        S = np.concatenate([s.ravel() for s in S_acc])
        F = np.concatenate([f.ravel() for f in F_acc]).astype(float)
        desc = describe_readout(S, D, F)
        desc_ctrl = None
        if S_ctrl_acc:
            desc_ctrl = describe_readout(
                np.concatenate([s.ravel() for s in S_ctrl_acc]), D, F)
        if desc:
            fig_descriptors(desc, desc_ctrl,
                            os.path.join(args.out, f"descriptors_{tag}.png"))
            summary["descriptors"] = desc
            summary["descriptors_shuffled_control"] = desc_ctrl
            print(f"  readout explained by image descriptors: "
                  f"R2 = {desc['joint_r2']:.3f}"
                  + (f"   (shuffled-label control: "
                     f"{desc_ctrl['joint_r2']:.3f})" if desc_ctrl else ""))
        banks["top"].sort(key=lambda d: -d["score"])
        banks["bot"].sort(key=lambda d: d["score"])
        fig_extremes(banks["top"], banks["bot"],
                     os.path.join(args.out, f"patches_{tag}.png"), target,
                     k=args.n_patch_examples)

    with open(os.path.join(args.out, f"xai_{tag}_{args.level}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {os.path.join(args.out, f'xai_{tag}_{args.level}.json')}")


if __name__ == "__main__":
    main()
