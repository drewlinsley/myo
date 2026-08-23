"""View geometry + normalization for DINOv2 feature extraction.

Torch-free on purpose: the geometry is where the subtle bugs live (aliasing,
aspect distortion, tile coverage), and keeping it here means it can be unit
tested without a GPU or the timm weights.

Two framings, and the choice matters scientifically:

  whole   the entire 1600x1100 field in one view. Force is a TISSUE-level
          property, so a model that only ever sees a 256x256 crop (3.7% of the
          FOV) may simply never observe the thing that determines it.
  tiled   518x518 crops at native resolution. Preserves fine texture that the
          whole-FOV resize necessarily discards.

The whole-FOV path downsamples ~3x. A naive nearest/bilinear resize aliases at
that ratio, folding high-frequency texture into exactly the statistics DINOv2
keys on — which would look like "whole-FOV framing doesn't work" rather than
like a bug. `area_resize` below is a true box filter, so it is antialiased by
construction.
"""

import numpy as np


def tile_starts(dim, size, n):
    """n evenly spaced tile origins covering [0, dim) with `size`-px tiles.

    The last tile is flush with the far edge, so the union covers the full
    extent (no unsampled strip at the right/bottom).
    """
    if dim <= size or n <= 1:
        return [0]
    last = dim - size
    return sorted({int(round(v)) for v in np.linspace(0, last, n)})


def plan_views(h, w, framing, fov_size=518, tile_size=518, tile_grid=(4, 3),
               fov_fit="pad"):
    """Describe every 2D view of one slice.

    Returns a list of dicts: {"mode", "y", "x", "th", "tw"} where mode is
    "resize" (take the whole slice, fit it into fov_size) or "crop" (take a
    tile_size window at (y, x) at native resolution).
    """
    if framing == "whole":
        return [{"mode": "resize", "y": 0, "x": 0, "th": h, "tw": w,
                 "out": fov_size, "fit": fov_fit}]
    if framing == "tiled":
        gx, gy = tile_grid
        ys = tile_starts(h, tile_size, gy)
        xs = tile_starts(w, tile_size, gx)
        return [{"mode": "crop", "y": int(y), "x": int(x),
                 "th": tile_size, "tw": tile_size, "out": tile_size}
                for y in ys for x in xs]
    raise ValueError(f"framing must be 'whole' or 'tiled', got {framing!r}")


def _integral(a):
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    return ii


def _ii_at(ii, y, x):
    """Bilinear sample of an integral image at fractional (y, x)."""
    H, W = ii.shape
    y = np.clip(y, 0, H - 1)
    x = np.clip(x, 0, W - 1)
    y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    a = ii[np.ix_(y0, x0)]; b = ii[np.ix_(y0, x1)]
    c = ii[np.ix_(y1, x0)]; d = ii[np.ix_(y1, x1)]
    return (a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx
            + c * fy * (1 - fx) + d * fy * fx)


def area_resize(img, out_h, out_w):
    """Exact box-filter ('area') resize — antialiased for any downsample ratio.

    Uses a bilinearly sampled integral image, so each output pixel is the true
    mean of its source rectangle rather than a point sample.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    if (h, w) == (out_h, out_w):
        return img.astype(np.float32)
    ii = _integral(img)
    ys = np.linspace(0, h, out_h + 1)
    xs = np.linspace(0, w, out_w + 1)
    y0, y1 = ys[:-1], ys[1:]
    x0, x1 = xs[:-1], xs[1:]
    tot = (_ii_at(ii, y1, x1) - _ii_at(ii, y0, x1)
           - _ii_at(ii, y1, x0) + _ii_at(ii, y0, x0))
    area = np.outer(y1 - y0, x1 - x0)
    return (tot / np.maximum(area, 1e-12)).astype(np.float32)


def fit_into_square(img, out, fit="pad", pad_value=0.0):
    """Resize a (H, W) array into an out x out square.

    fit="pad"    preserve aspect (long side -> out), pad the short side.
                 1600x1100 has aspect 1.45; squashing it to square distorts
                 myotube alignment, which is plausibly the force-relevant
                 morphology. Padding costs ~30% background tokens instead.
    fit="squash" fill the square, distorting aspect. Cheaper in tokens.
    """
    h, w = img.shape
    if fit == "squash":
        return area_resize(img, out, out)
    scale = out / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = area_resize(img, nh, nw)
    canvas = np.full((out, out), float(pad_value), dtype=np.float32)
    oy, ox = (out - nh) // 2, (out - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def make_view(slice2d, spec, p_low, p_high, mean_rgb, std_rgb, normalize_fn):
    """One raw (H, W) slice -> a (3, out, out) float32 array for DINOv2.

    normalize_fn is src.data.normalization.normalize, called with
    apply_timm=False: that helper's TIMM constants are single-channel
    (0.485/0.229), while DINOv2 wants the real per-channel ImageNet stats.
    Those are applied here instead, from timm's own resolved data config.
    """
    if spec["mode"] == "crop":
        y, x, th, tw = spec["y"], spec["x"], spec["th"], spec["tw"]
        sub = slice2d[y:y + th, x:x + tw]
        if sub.shape != (th, tw):          # edge tile on a small FOV
            pad = np.zeros((th, tw), dtype=sub.dtype)
            pad[:sub.shape[0], :sub.shape[1]] = sub
            sub = pad
        img = normalize_fn(sub, p_low, p_high, apply_timm=False)
        if img.shape[0] != spec["out"] or img.shape[1] != spec["out"]:
            img = area_resize(img, spec["out"], spec["out"])
    else:
        img = normalize_fn(slice2d, p_low, p_high, apply_timm=False)
        img = fit_into_square(img, spec["out"], spec.get("fit", "pad"))

    img = np.asarray(img, dtype=np.float32)
    rgb = np.repeat(img[None], 3, axis=0)
    m = np.asarray(mean_rgb, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std_rgb, dtype=np.float32).reshape(3, 1, 1)
    return (rgb - m) / s


def view_foreground(mask2d, spec):
    """Foreground fraction of a view, using the same geometry as make_view."""
    if mask2d is None:
        return None
    m = np.asarray(mask2d, dtype=np.float32)
    if spec["mode"] == "crop":
        y, x, th, tw = spec["y"], spec["x"], spec["th"], spec["tw"]
        sub = m[y:y + th, x:x + tw]
        if sub.size == 0:
            return 0.0
        return float(sub.mean())
    return float(m.mean())


def fg_window_scores(mask2d, tile, stride):
    """Foreground fraction of every `tile`-sized window on a `stride` grid.

    Integer-aligned, so it uses the integral image directly rather than the
    bilinear sampler `area_resize` needs.
    """
    m = np.asarray(mask2d, dtype=np.float64)
    h, w = m.shape
    if h < tile or w < tile:
        return [], [], np.zeros((0, 0))
    ii = _integral(m)
    ys = list(range(0, h - tile + 1, stride))
    xs = list(range(0, w - tile + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)          # keep the far edge reachable
    if xs[-1] != w - tile:
        xs.append(w - tile)
    ya, xa = np.asarray(ys), np.asarray(xs)
    tot = (ii[np.ix_(ya + tile, xa + tile)] - ii[np.ix_(ya, xa + tile)]
           - ii[np.ix_(ya + tile, xa)] + ii[np.ix_(ya, xa)])
    return ys, xs, tot / float(tile * tile)


def plan_views_fg(mask2d, n_tiles=12, tile_size=518, stride=64, min_sep=None,
                  min_fg=0.0):
    """Place tiles WHERE THE TISSUE IS, instead of on a fixed grid.

    The fixed 4x3 grid samples the field uniformly, so on a sparse field most
    tiles are mostly background and a strict foreground threshold can leave a
    volume with nothing. This searches the field for the `n_tiles` windows with
    the highest tissue coverage, greedily, with a minimum separation so they do
    not all collapse onto the single densest spot.

    Returns specs in the same shape `plan_views` returns, so downstream code
    (make_view, view_foreground) is unchanged.

    min_sep defaults to tile_size // 2, i.e. at most 50% overlap between kept
    tiles -- enough to follow tissue, not so much that one region is counted
    many times and dominates the volume average.

    min_fg is a floor on a window's tissue coverage. It matters more than it
    looks: WITHOUT it, greedy selection keeps going until it has n_tiles, so
    once the tissue-rich windows are used up it pads the list with background.
    Measured on a diagonal-band phantom, forcing 12 tiles gave coverages of
    0.70 0.70 0.70 0.69 0.31 0.28 0.28 0.25 0.01 -- the four good tiles, then
    junk that dragged the volume average BELOW the fixed grid it was meant to
    beat. Returning fewer tiles is the correct answer; callers must handle an
    empty list rather than assume a fixed count.
    """
    h, w = np.asarray(mask2d).shape
    if h < tile_size or w < tile_size:
        return [{"mode": "crop", "y": 0, "x": 0, "th": tile_size,
                 "tw": tile_size, "out": tile_size}]
    if min_sep is None:
        min_sep = tile_size // 2
    ys, xs, sc = fg_window_scores(mask2d, tile_size, stride)
    cand = [(float(sc[i, j]), ys[i], xs[j])
            for i in range(len(ys)) for j in range(len(xs))]
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))   # deterministic ties
    picked = []
    for s, y, x in cand:
        if len(picked) >= n_tiles:
            break
        if s < min_fg:
            break                    # sorted desc: nothing later qualifies
        if all(abs(y - py) >= min_sep or abs(x - px) >= min_sep
               for _, py, px in picked):
            picked.append((s, y, x))
    return [{"mode": "crop", "y": int(y), "x": int(x), "th": tile_size,
             "tw": tile_size, "out": tile_size} for _, y, x in picked]
