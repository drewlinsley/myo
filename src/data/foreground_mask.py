"""Foreground masking for raw BF volumes — for post-hoc artifact removal.

Mirrors the threshold + cleanup logic used in datasets.py / explore_masks.py
so prediction-time masking matches training-time masking semantics.
"""

import numpy as np


def _threshold(bf_raw, method):
    from skimage.filters import (
        threshold_otsu, threshold_li, threshold_triangle, threshold_minimum)
    fn = {"minimum": threshold_minimum,
          "otsu": threshold_otsu,
          "li": threshold_li,
          "triangle": threshold_triangle}[method]
    return float(fn(bf_raw.ravel().astype(np.float64)))


def _cleanup_per_slice(mask3d, dilate, min_frac):
    """Per-Z-slice 2D dilation + small-component removal."""
    from scipy.ndimage import (
        binary_dilation, generate_binary_structure, label as ndimage_label)
    struct = generate_binary_structure(2, 1)
    out = np.zeros_like(mask3d, dtype=bool)
    for z in range(mask3d.shape[0]):
        m = mask3d[z]
        if dilate:
            m = binary_dilation(m, structure=struct, iterations=dilate)
        if min_frac and min_frac > 0:
            labeled, n_comp = ndimage_label(m, structure=struct)
            if n_comp:
                fg_total = int(m.sum())
                min_size = int(min_frac * fg_total)
                sizes = np.bincount(labeled.ravel())
                keep_ids = np.where(sizes >= min_size)[0]
                keep_ids = keep_ids[keep_ids != 0]
                m = np.isin(labeled, keep_ids)
        out[z] = m
    return out


def compute_bf_foreground_mask(bf_raw, method="minimum",
                               dilate=0, min_component_frac=0.0):
    """Threshold raw BF to a (Z, H, W) bool mask, with optional cleanup.

    Args:
        bf_raw: numpy array, (Z, H, W) — raw (un-normalized) BF.
        method: "minimum" | "otsu" | "li" | "triangle".
        dilate: number of 2D dilation iterations (per Z-slice).
        min_component_frac: drop connected components smaller than this
            fraction of the foreground pixels (per Z-slice). 0 disables.

    Returns:
        (Z, H, W) bool mask. True = foreground (keep predictions).
    """
    thresh = _threshold(bf_raw, method)
    mask = bf_raw > thresh
    if dilate or min_component_frac > 0:
        mask = _cleanup_per_slice(mask, dilate, min_component_frac)
    return mask
