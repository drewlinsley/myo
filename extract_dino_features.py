"""Frozen DINOv2 features per staged volume, cached to .npz for cheap probing.

Why a separate extraction step
------------------------------
Each LOO arm of the GPU trainers costs 20 model trainings, so nothing could be
swept. Extracting features once and fitting linear probes over them turns every
subsequent question — framing, pooling, foreground threshold, target column —
into a sub-second experiment, which is what makes label-permutation nulls and
honest multiple-comparison control affordable (see probe_force_features.py).

Design notes that matter scientifically
---------------------------------------
* 2D only. DINOv2 has no 3D variant, and the 2D arm was already the better one.
* `tiled` is the default and the one to trust: 518x518 crops at NATIVE
  resolution whose union covers the whole 1600x1100 field, so averaging the
  per-tile embeddings already gives a whole-FOV representation with no resize,
  no aliasing, and no aspect distortion. Force being a tissue-level property is
  handled by the aggregation, not by squeezing the field into one view.
  `whole` (opt-in) downsamples ~3x into a single view. The ONLY thing it adds
  is attention ACROSS the field in one forward pass; it pays for that by
  discarding the fine texture, so treat it as a comparison arm, not a default.
* Views are stored UNPOOLED (one vector per view, not per volume), and each
  view keeps SPATIAL STATISTICS of its patch tokens, not just their mean.
  Mean-pooling a token grid discards spatial heterogeneity across the field —
  the same thing AdaptiveAvgPool was already discarding in the ResNeXt path —
  and "how uniformly aligned and how densely packed are the fibers" is a
  plausible force correlate. Be precise about the scale, though: under the
  default `tiled` framing a view is one 518px tile (9% of the field), so
  `patch_std` is WITHIN-tile heterogeneity. ACROSS-field heterogeneity is a
  different statistic — the std of patch_mean across views — which the probe
  computes via `--agg mean+std`. Sweep both; they are not interchangeable.
  `--grid_pool N` keeps a coarse NxN grid for genuinely spatial probes.
* `view_fg_frac` is always stored even when foreground weighting is off, so a
  probe can drop background tiles later without re-extraction.

Usage
-----
    python extract_dino_features.py --data_dir data_phalloidin_mhc_051826_staged \
        --input gfp --framing tiled --norm_scope volume \
        --output_dir results/dino_features
    # inspect the plan without loading weights or touching the GPU:
    python extract_dino_features.py ... --limit 2 --dry_run
"""

import os
import json
import glob
import hashlib
import argparse

import numpy as np

from src.data.normalization import normalize, global_percentiles
from src.data.zband import resolve_z_range
from src.data.dino_views import (plan_views, make_view, view_foreground,
                                 area_resize, fit_into_square)

DEFAULT_MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"


def build_dino(model_name, device):
    """Frozen DINOv2 + its own resolved normalization stats.

    Register variants (reg4) are preferred. The mechanism is not magnitude —
    after the final LayerNorm all tokens are renormalized to roughly equal norm.
    It is that artifact tokens carry GLOBAL information in place of local patch
    content, which distorts the direction of any pooled patch vector. Registers
    give that global information somewhere else to live.
    """
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().requires_grad_(False).to(device)
    cfg = timm.data.resolve_model_data_config(model)
    patch = model.patch_embed.patch_size
    patch = patch[0] if isinstance(patch, (tuple, list)) else patch
    img_size = cfg["input_size"][-1]
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    dim = model.num_features
    has_cls = getattr(model, "cls_token", None) is not None
    # forward_features returns POST-final-LayerNorm tokens only when
    # global_pool == 'token' (DINOv2). For a global_pool='avg' checkpoint timm
    # moves the LN into fc_norm, which forward_features does not apply, and the
    # tokens come back un-normalized at a wildly different scale.
    if getattr(model, "global_pool", "token") == "avg":
        print("WARNING: this checkpoint uses global_pool='avg', so "
              "forward_features returns PRE-norm tokens. patch_std in "
              "particular will not be comparable to a DINOv2 run.")
    return {"model": model, "mean": cfg["mean"], "std": cfg["std"],
            "patch": int(patch), "img_size": int(img_size), "has_cls": has_cls,
            "n_prefix": int(n_prefix), "dim": int(dim), "name": model_name}


def encode_views(ctx, views, view_masks, device, batch_size, amp=True,
                 grid_pool=0):
    """(N, 3, S, S) -> per-view SPATIAL statistics of the patch tokens.

    Returns float16 arrays:
      cls           (N, D)          the class token
      patch_mean    (N, D)          mean over spatial tokens
      patch_std     (N, D)          std over spatial tokens  <- heterogeneity
      patch_mean_fg (N, D)          foreground-weighted mean (if a mask exists)
      patch_grid    (N, g, g, D)    coarse pooled token grid (if grid_pool=g)

    patch_std collapses a 37x37 token grid to per-dimension dispersion rather
    than just its mean, keeping heterogeneity that AdaptiveAvgPool discarded in
    the ResNeXt path. Because forward_features returns POST-final-LayerNorm
    tokens, every token has roughly equal norm, so this is an angular-dispersion
    measure rather than a norm-dominated one — which is why the known high-norm
    artifact tokens do not dominate it here. Scale caveat: this is dispersion
    WITHIN one view; across-view dispersion is `--agg mean+std` in the probe.

    Patch tokens are taken from index `num_prefix_tokens` onward — hard-coding
    tok[:, 1:] would silently fold 4 register tokens into every statistic on a
    reg4 model.
    """
    import torch
    import torch.nn.functional as F

    model = ctx["model"]
    n_prefix, patch = ctx["n_prefix"], ctx["patch"]
    out_cls, out_pm, out_ps, out_fg, out_grid = [], [], [], [], []
    use_amp = amp and device.type == "cuda"

    for i in range(0, len(views), batch_size):
        chunk = np.stack(views[i:i + batch_size])
        x = torch.from_numpy(chunk).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = model.forward_features(x)
            else:
                tok = model.forward_features(x)
        tok = tok.float()
        if not torch.isfinite(tok).all():
            raise SystemExit(
                "non-finite tokens from the backbone (fp16 overflow?) — one bad "
                "view NaNs the whole volume mean and surfaces much later as a "
                "confusing sklearn error. Rerun with --no_amp.")
        if not ctx.get("has_cls", True):
            raise SystemExit(f"{ctx['name']} has no class token; use "
                             "--token patch_mean,patch_std instead of cls")
        cls = tok[:, 0]
        pt = tok[:, n_prefix:]                      # (B, gh*gw, D)
        gh = x.shape[-2] // patch
        gw = x.shape[-1] // patch
        if pt.shape[1] != gh * gw:
            raise SystemExit(
                f"token count {pt.shape[1]} != {gh}x{gw}={gh*gw}; "
                f"num_prefix_tokens={n_prefix} may be wrong for this model")
        out_cls.append(cls.cpu().numpy())
        out_pm.append(pt.mean(dim=1).cpu().numpy())
        out_ps.append(pt.std(dim=1).cpu().numpy())

        if grid_pool:
            g = int(grid_pool)
            grid = pt.transpose(1, 2).reshape(pt.shape[0], pt.shape[2], gh, gw)
            grid = F.adaptive_avg_pool2d(grid, (g, g))       # (B, D, g, g)
            out_grid.append(grid.permute(0, 2, 3, 1).cpu().numpy())

        if view_masks is not None:
            mgrid = np.stack([area_resize(m, gh, gw)
                              for m in view_masks[i:i + batch_size]])
            w = torch.from_numpy(mgrid.reshape(len(mgrid), -1)).to(pt.device)
            w = w.clamp(min=0).unsqueeze(-1)         # (B, gh*gw, 1)
            wsum = w.sum(dim=1)                       # (B, 1)
            fg = (pt * w).sum(dim=1) / wsum.clamp(min=1e-6)
            # A view with no foreground would otherwise become an all-zero
            # vector — not neutral, but an out-of-manifold point that pulls the
            # volume mean toward the origin. Fall back to the plain mean.
            empty = (wsum.squeeze(-1) < 1e-3)
            if empty.any():
                fg[empty] = pt[empty].mean(dim=1)
            out_fg.append(fg.cpu().numpy())

    res = {"cls": np.concatenate(out_cls).astype(np.float16),
           "patch_mean": np.concatenate(out_pm).astype(np.float16),
           "patch_std": np.concatenate(out_ps).astype(np.float16)}
    if out_fg:
        res["patch_mean_fg"] = np.concatenate(out_fg).astype(np.float16)
    if out_grid:
        res["patch_grid"] = np.concatenate(out_grid).astype(np.float16)
    return res


def foreground_mask(raw, method, dilate, min_frac):
    """BF/GFP foreground mask, tolerant of threshold failures.

    threshold_minimum raises RuntimeError when it cannot find two histogram
    maxima, which happens on near-black GFP backgrounds. Fall back rather than
    killing a multi-hour extraction, and record that it happened.
    """
    from src.data.foreground_mask import compute_bf_foreground_mask
    try:
        return compute_bf_foreground_mask(raw, method=method, dilate=dilate,
                                          min_component_frac=min_frac), None
    except Exception as e:
        try:
            return compute_bf_foreground_mask(raw, method="li", dilate=dilate,
                                              min_component_frac=min_frac), \
                f"{method} failed ({type(e).__name__}), used li"
        except Exception as e2:
            return None, f"{method} and li both failed ({type(e2).__name__})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--framing", default="tiled",
                   help="Comma list of tiled|whole. 'tiled' crops at native "
                        "resolution and covers the full FOV once its views are "
                        "averaged — no resize, so no aliasing or aspect "
                        "distortion. 'whole' resizes the field into one view; "
                        "its only advantage is cross-field attention, at the "
                        "cost of ~3x downsampling. NOTE: each framing "
                        "re-reads the volume, so listing two doubles the disk "
                        "and masking cost.")
    p.add_argument("--fov_fit", choices=["pad", "squash"], default="pad",
                   help="whole framing: 'pad' preserves the 1.45 aspect ratio "
                        "(myotube alignment is plausibly the signal); 'squash' "
                        "fills the square and wastes no tokens on padding.")
    p.add_argument("--tile_grid", default="4,3", help="tiled framing: gx,gy")
    p.add_argument("--norm_scope", choices=["volume", "global"], default="volume",
                   help="'global' pools percentiles across the dataset so "
                        "absolute brightness survives. NOTE: probe_intensity.sh "
                        "found intensity does not predict force, so 'volume' is "
                        "the sensible default.")
    p.add_argument("--z_range", default="auto")
    p.add_argument("--z_stride", type=int, default=3,
                   help="Adjacent slices in a ~35-slice band are near-duplicates; "
                        "3 cuts extraction ~3x at negligible information cost.")
    p.add_argument("--mask_source", choices=["bf", "gfp", "none"], default="bf",
                   help="Foreground mask modality. 'bf' even for a GFP run: a "
                        "BF-derived tissue mask is independent of the GFP "
                        "intensity being measured.")
    p.add_argument("--mask_method", default="li",
                   choices=["minimum", "otsu", "li", "triangle"])
    p.add_argument("--mask_dilate", type=int, default=0)
    p.add_argument("--mask_min_frac", type=float, default=0.0)
    p.add_argument("--grid_pool", type=int, default=0,
                   help="Also store an NxN average-pooled patch-token grid per "
                        "view (0 = off), for probes that need spatial layout "
                        "rather than a single pooled vector.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="Print the per-volume plan and exit — no weights, no GPU.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    framings = [f.strip() for f in args.framing.split(",") if f.strip()]
    for f in framings:
        if f not in ("whole", "tiled"):
            raise SystemExit(f"--framing must be whole|tiled, got {f!r}")
    gx, gy = (int(v) for v in args.tile_grid.split(","))

    data_dir = args.data_dir
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)
    mask_dir = (os.path.join(data_dir, args.mask_source)
                if args.mask_source != "none" else None)

    stems = args.stems or sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(mod_dir, "*.npy")))
    if args.limit:
        stems = stems[:args.limit]
    if not stems:
        raise SystemExit(f"no volumes in {mod_dir}/")

    z_range = None if args.z_range in ("none", "null", "") else args.z_range
    if z_range not in (None, "auto"):
        z_range = [int(v) for v in str(z_range).split(",")]

    gpct = (global_percentiles(stats_dir, args.input)
            if args.norm_scope == "global" else None)
    if gpct:
        print(f"global percentiles ({args.input}): {gpct}")

    print(f"{len(stems)} volume(s)  framings={framings}  z_stride={args.z_stride}"
          f"  mask={args.mask_source}/{args.mask_method}")

    ctx = None
    device = None
    if not args.dry_run:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("WARNING: no CUDA — this will be very slow. "
                  "Run scripts/check_gpu.py.")
        ctx = build_dino(args.model, device)
        print(f"model={ctx['name']} dim={ctx['dim']} patch={ctx['patch']} "
              f"img_size={ctx['img_size']} n_prefix={ctx['n_prefix']}")

    # Every setting that changes the FEATURES goes into the directory key.
    cfg_key = {
        "model": args.model, "input": args.input,
        "norm_scope": args.norm_scope, "global_pct": gpct,
        "z_range": args.z_range, "z_stride": args.z_stride,
        "fov_fit": args.fov_fit, "tile_grid": [gx, gy],
        "grid_pool": args.grid_pool, "mask_source": args.mask_source,
        "mask_method": args.mask_method, "mask_dilate": args.mask_dilate,
        "mask_min_frac": args.mask_min_frac,
    }
    cfg_hash = hashlib.sha1(
        json.dumps(cfg_key, sort_keys=True).encode()).hexdigest()[:8]
    print(f"config hash: {cfg_hash}  (feature dirs are keyed by it, so a "
          f"different backbone or geometry cannot silently reuse cached "
          f"features)")

    manifest = {"config_hash": cfg_hash, "config": cfg_key,
                "model": args.model, "input": args.input,
                "norm_scope": args.norm_scope, "global_pct": gpct,
                "z_range": args.z_range, "z_stride": args.z_stride,
                "fov_fit": args.fov_fit, "tile_grid": [gx, gy],
                "mask_source": args.mask_source, "mask_method": args.mask_method,
                "volumes": {}, "warnings": []}
    if ctx:
        manifest.update({"dim": ctx["dim"], "patch": ctx["patch"],
                         "img_size": ctx["img_size"],
                         "n_prefix": ctx["n_prefix"]})

    for si, stem in enumerate(stems):
        with open(os.path.join(stats_dir, f"{stem}.json")) as f:
            st = json.load(f)
        path = os.path.join(mod_dir, f"{stem}.npy")
        vol = np.load(path, mmap_mode="r")
        n_z, H, W = vol.shape
        z_lo, z_hi = ((0, n_z) if z_range is None
                      else resolve_z_range(z_range, st, n_z))
        zs = list(range(z_lo, z_hi, max(1, args.z_stride)))

        if gpct:
            p_low, p_high = gpct
        else:
            p_low = float(st[args.input]["p_low"])
            p_high = float(st[args.input]["p_high"])

        for framing in framings:
            out_dir = os.path.join(args.output_dir,
                                   f"{args.input}_{framing}_{args.norm_scope}"
                                   f"_{cfg_hash}")
            out_npz = os.path.join(out_dir, f"{stem}.npz")
            if os.path.exists(out_npz) and not args.force:
                manifest["volumes"].setdefault(stem, {})[framing] = {
                    "cached": True}
                print(f"  [{si+1}/{len(stems)}] {stem} {framing}: cached")
                continue

            specs = plan_views(H, W, framing, fov_size=518, tile_size=518,
                               tile_grid=(gx, gy), fov_fit=args.fov_fit)
            n_views = len(zs) * len(specs)
            if args.dry_run:
                print(f"  [{si+1}/{len(stems)}] {stem} ({n_z},{H},{W}) "
                      f"z_auto={st.get('z_auto')} -> z[{z_lo}:{z_hi}] "
                      f"{len(zs)} slices x {len(specs)} views = {n_views}"
                      f"  [{framing}]")
                continue

            raw_band = np.asarray(vol[z_lo:z_hi][::max(1, args.z_stride)])
            mask_band, warn = None, None
            if mask_dir:
                mpath = os.path.join(mask_dir, f"{stem}.npy")
                if os.path.exists(mpath):
                    mv = np.load(mpath, mmap_mode="r")
                    if mv.shape != vol.shape:
                        raise SystemExit(
                            f"{stem}: mask source '{args.mask_source}' has "
                            f"shape {mv.shape} but '{args.input}' has "
                            f"{vol.shape}. The z-band and view geometry come "
                            f"from the input volume, so a mismatch silently "
                            f"misaligns every mask.")
                    mraw = np.asarray(mv[z_lo:z_hi][::max(1, args.z_stride)])
                    mask_band, warn = foreground_mask(
                        mraw, args.mask_method, args.mask_dilate,
                        args.mask_min_frac)
                    if warn:
                        manifest["warnings"].append(f"{stem}: {warn}")

            views, vmasks, v_z, v_yx, v_fg, v_int = [], [], [], [], [], []
            for zi, z in enumerate(zs):
                sl = raw_band[zi]
                msl = mask_band[zi] if mask_band is not None else None
                for sp in specs:
                    views.append(make_view(sl, sp, p_low, p_high,
                                           ctx["mean"], ctx["std"], normalize))
                    if msl is not None:
                        # The mask MUST go through the same geometry as the
                        # image. Previously the resize path stored the raw mask
                        # while the image was aspect-preserved and padded, so
                        # the two were stretched apart by 518/356 = 1.455 —
                        # a ~250px misalignment at the extremes, silently
                        # corrupting every foreground-weighted feature.
                        if sp["mode"] == "crop":
                            mm = msl[sp["y"]:sp["y"] + sp["th"],
                                     sp["x"]:sp["x"] + sp["tw"]].astype(np.float32)
                            if mm.shape != (sp["th"], sp["tw"]):
                                pad = np.zeros((sp["th"], sp["tw"]), np.float32)
                                pad[:mm.shape[0], :mm.shape[1]] = mm
                                mm = pad
                        else:
                            mm = fit_into_square(msl.astype(np.float32),
                                                 sp["out"], sp.get("fit", "pad"),
                                                 pad_value=0.0)
                        vmasks.append(mm)
                    v_z.append(z)
                    v_yx.append([sp["y"], sp["x"]])
                    v_fg.append(view_foreground(msl, sp) if msl is not None else 1.0)
                    sub = (sl[sp["y"]:sp["y"] + sp["th"],
                              sp["x"]:sp["x"] + sp["tw"]]
                           if sp["mode"] == "crop" else sl)
                    v_int.append(float(np.mean(sub)) if sub.size else 0.0)

            feats = encode_views(ctx, views, vmasks or None, device,
                                 args.batch_size, grid_pool=args.grid_pool)
            os.makedirs(out_dir, exist_ok=True)
            tmp = os.path.join(out_dir, f".{stem}.partial")
            # Dot-prefix the temp file. np.savez_compressed always appends
            # ".npz", so any suffix scheme still matches the probe's
            # glob("*.npz") — but glob skips dotfiles, so an orphan left by a
            # killed run cannot surface as a phantom stem.
            np.savez_compressed(
                tmp + ".npz", view_z=np.asarray(v_z, np.int32),
                view_yx=np.asarray(v_yx, np.int32),
                view_fg_frac=np.asarray(v_fg, np.float32),
                view_mean_int=np.asarray(v_int, np.float32),
                z_lo=z_lo, z_hi=z_hi, n_z_total=n_z,
                p_low_used=p_low, p_high_used=p_high, **feats)
            os.replace(tmp + ".npz", out_npz)
            manifest["volumes"].setdefault(stem, {})[framing] = {
                "n_views": int(n_views), "z": [int(z_lo), int(z_hi)],
                "shape": [int(n_z), int(H), int(W)]}
            print(f"  [{si+1}/{len(stems)}] {stem} {framing}: "
                  f"{n_views} views -> {out_npz}")

    if not args.dry_run:
        # One manifest per hashed feature dir. A single shared manifest was
        # clobbered by the next modality/model and could describe a config its
        # neighbouring .npz files were not produced with.
        for framing in framings:
            fdir = os.path.join(args.output_dir,
                                f"{args.input}_{framing}_{args.norm_scope}"
                                f"_{cfg_hash}")
            if not os.path.isdir(fdir):
                continue
            mp = os.path.join(fdir, "manifest.json")
            merged = dict(manifest)
            if os.path.exists(mp):          # keep earlier volumes + warnings
                try:
                    prev = json.load(open(mp))
                    merged["volumes"] = {**prev.get("volumes", {}),
                                         **manifest["volumes"]}
                    merged["warnings"] = (prev.get("warnings", [])
                                          + manifest["warnings"])
                except Exception:
                    pass
            with open(mp, "w") as f:
                json.dump(merged, f, indent=2)
        mp = os.path.join(args.output_dir, "manifest.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nmanifest -> {mp}")
        if manifest["warnings"]:
            print(f"  {len(manifest['warnings'])} warning(s), e.g. "
                  f"{manifest['warnings'][0]}")


if __name__ == "__main__":
    main()
