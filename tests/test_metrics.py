"""Tests for PSNR, SSIM, Pearson, MAE metrics."""

import numpy as np
import torch
import pytest
from src.metrics import psnr, ssim, pearson_corr, mae


def test_psnr_identical():
    """PSNR of identical arrays should be inf."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    result = psnr(x, x)
    assert result == float("inf")


def test_psnr_torch_identical():
    """PSNR of identical tensors should be inf."""
    x = torch.randn(1, 1, 32, 32)
    result = psnr(x, x)
    assert result == float("inf")


def test_psnr_positive():
    """PSNR should be positive for similar but different signals."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    y = x + np.random.RandomState(43).randn(32, 32).astype(np.float32) * 0.01
    result = psnr(x, y)
    assert result > 0


def test_ssim_identical():
    """SSIM of identical arrays should be 1.0."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    result = ssim(x, x, win_size=7)
    assert abs(result - 1.0) < 1e-5


def test_ssim_range():
    """SSIM should be in [-1, 1]."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    y = np.random.RandomState(43).rand(32, 32).astype(np.float32)
    result = ssim(x, y, win_size=7)
    assert -1.0 <= result <= 1.0


def test_pearson_identical():
    """Pearson of identical arrays should be 1.0."""
    x = np.random.RandomState(42).rand(100).astype(np.float32)
    result = pearson_corr(x, x)
    assert abs(result - 1.0) < 1e-5


def test_pearson_uncorrelated():
    """Pearson of uncorrelated arrays should be near 0."""
    x = np.random.RandomState(42).rand(10000).astype(np.float32)
    y = np.random.RandomState(43).rand(10000).astype(np.float32)
    result = pearson_corr(x, y)
    assert abs(result) < 0.1


def test_mae_identical():
    """MAE of identical arrays should be 0."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    result = mae(x, x)
    assert result == 0.0


def test_mae_positive():
    """MAE should be positive for different arrays."""
    x = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    y = np.random.RandomState(43).rand(32, 32).astype(np.float32)
    result = mae(x, y)
    assert result > 0
