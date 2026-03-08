"""Tests for normalization round-trip and edge cases."""

import numpy as np
import pytest
from src.data.normalization import normalize, denormalize


def test_roundtrip_no_timm():
    """normalize -> denormalize should recover original (within clipped range)."""
    data = np.array([100, 500, 1000, 5000, 9000, 10000], dtype=np.float32)
    p_low, p_high = 200.0, 9500.0

    normed = normalize(data, p_low, p_high, apply_timm=False)
    recovered = denormalize(normed, p_low, p_high, applied_timm=False)

    # Values within [p_low, p_high] should be recovered exactly
    clipped = np.clip(data, p_low, p_high)
    np.testing.assert_allclose(recovered, clipped, atol=1e-4)


def test_roundtrip_with_timm():
    """normalize -> denormalize with TIMM stats should round-trip."""
    data = np.random.RandomState(42).randint(500, 8000, size=(4, 4)).astype(np.float32)
    p_low, p_high = 200.0, 9000.0

    normed = normalize(data, p_low, p_high, apply_timm=True)
    recovered = denormalize(normed, p_low, p_high, applied_timm=True)

    clipped = np.clip(data, p_low, p_high)
    np.testing.assert_allclose(recovered, clipped, atol=1e-3)


def test_output_range_no_timm():
    """Without TIMM, output should be in [0, 1]."""
    data = np.array([0, 500, 1000], dtype=np.float32)
    normed = normalize(data, 0.0, 1000.0, apply_timm=False)
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_zero_range():
    """If p_low == p_high, output should be all zeros."""
    data = np.array([5.0, 5.0, 5.0])
    normed = normalize(data, 5.0, 5.0, apply_timm=False)
    np.testing.assert_array_equal(normed, 0.0)


def test_percentile_bounds():
    """Values outside [p_low, p_high] should be clipped before scaling."""
    data = np.array([0.0, 500.0, 10000.0])
    normed = normalize(data, 500.0, 500.0, apply_timm=False)
    # All clipped to same value -> all zeros
    np.testing.assert_array_equal(normed, 0.0)
