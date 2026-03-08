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
