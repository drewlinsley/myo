"""Invertible normalization for BF and GFP channels.

Pipeline:
  1. Per-volume percentile clip: np.clip(x, p_low, p_high)
  2. Scale to [0, 1]: x = (x - p_low) / (p_high - p_low)
  3. If using pretrained weights: apply TIMM stats: x = (x - mean) / std
  4. If random init: skip step 3 (stay in [0, 1])

GFP target always stays in [0, 1] (no TIMM stats).
"""

import numpy as np

TIMM_MEAN = 0.485
TIMM_STD = 0.229


def normalize(data, p_low, p_high, apply_timm=False):
    """Normalize raw uint16/float data to [0,1] (optionally + TIMM stats).

    Args:
        data: numpy array (any shape)
        p_low: lower percentile clip value
        p_high: upper percentile clip value
        apply_timm: if True, further normalize with ImageNet channel stats

    Returns:
        Normalized float32 array
    """
    data = data.astype(np.float32)
    data = np.clip(data, p_low, p_high)

    denom = p_high - p_low
    if denom > 0:
        data = (data - p_low) / denom
    else:
        data = np.zeros_like(data)

    if apply_timm:
        data = (data - TIMM_MEAN) / TIMM_STD

    return data


def normalize_auto(data, percentile_clip=(0.5, 99.5)):
    """Compute percentiles on-the-fly and normalize to [0,1].

    Args:
        data: numpy array (any shape)
        percentile_clip: (low_pct, high_pct) percentile bounds

    Returns:
        (normalized_data, p_low, p_high) — p_low/p_high are the computed
        percentile values so callers can detect flat/empty data.
    """
    data = data.astype(np.float32)
    p_low = float(np.percentile(data, percentile_clip[0]))
    p_high = float(np.percentile(data, percentile_clip[1]))
    return normalize(data, p_low, p_high, apply_timm=False), p_low, p_high


def denormalize(data, p_low, p_high, applied_timm=False):
    """Exact inverse of normalize: recover original scale.

    Args:
        data: normalized array
        p_low: lower percentile clip value used during normalization
        p_high: upper percentile clip value used during normalization
        applied_timm: whether TIMM stats were applied

    Returns:
        Array in original value range (clipped to [p_low, p_high])
    """
    data = data.copy() if isinstance(data, np.ndarray) else data.clone()

    if applied_timm:
        data = data * TIMM_STD + TIMM_MEAN

    data = data * (p_high - p_low) + p_low
    return data


def global_percentiles(stats_dir, modality, stems=None):
    """Dataset-wide (p_low, p_high) pooled across every per-volume stats JSON.

    Why this exists
    ---------------
    The default per-volume normalization clips each volume to ITS OWN
    percentiles and rescales to [0, 1], so a dim sparse tissue and a bright
    dense one both end up spanning the same range. That erases absolute
    intensity — which, for GFP/phalloidin, is the most plausible proxy for
    myotube density and therefore for contraction force. Normalizing every
    volume by one shared pair keeps relative brightness between tissues.

    p_low is the min over volumes and p_high the max, so no volume is clipped
    harder than it was before; the trade is a narrower effective dynamic range
    for dim volumes, which is exactly the signal we want to preserve.

    Args:
        stats_dir: directory of <stem>.json files written by compute_stats.py
        modality: "bf" or "gfp"
        stems: optional list of stems to restrict to (default: every JSON).
            Pass the TRAINING stems only if you need the statistic to be
            leak-free with respect to a held-out split.

    Returns:
        (p_low, p_high) floats.
    """
    import glob as _glob
    import json as _json
    import os as _os

    if stems is None:
        paths = sorted(_glob.glob(_os.path.join(stats_dir, "*.json")))
    else:
        paths = [_os.path.join(stats_dir, f"{s}.json") for s in stems]

    lows, highs = [], []
    for p in paths:
        if not _os.path.exists(p):
            continue
        with open(p) as f:
            st = _json.load(f)
        if modality not in st:
            continue
        lows.append(float(st[modality]["p_low"]))
        highs.append(float(st[modality]["p_high"]))

    if not lows:
        raise ValueError(
            f"global_percentiles: no stats JSON "
            + (f"for the {len(paths)} requested stem(s) " if stems is not None
               else f"in {stats_dir} ")
            + f"carried a '{modality}' entry.")
    if stems is not None and len(lows) < len(stems):
        raise ValueError(
            f"global_percentiles: only {len(lows)} of {len(stems)} requested "
            f"stems had a '{modality}' stats entry. A silent subset would "
            f"narrow the 'global' statistic to whatever happened to exist — "
            f"check for a stem-naming mismatch.")
    return float(min(lows)), float(max(highs))
