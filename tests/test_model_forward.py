"""Tests for model forward pass shapes (2D and 3D)."""

import torch
import pytest


def test_2d_model_forward():
    """2D U-Net: (1, 1, 256, 256) -> (1, 1, 256, 256)."""
    import segmentation_models_pytorch as smp_2d

    model = smp_2d.Unet(
        encoder_name="resnext50_32x4d",
        encoder_weights=None,  # random init for speed
        in_channels=1,
        classes=1,
        activation=None,
    )
    model.eval()

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 256, 256)


def test_2d_model_forward_small():
    """2D U-Net with smaller input: (1, 1, 32, 32)."""
    import segmentation_models_pytorch as smp_2d

    model = smp_2d.Unet(
        encoder_name="resnext50_32x4d",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 32)


def test_3d_model_forward():
    """3D U-Net: (1, 1, 16, 16, 8) -> (1, 1, 16, 16, 8) with time_kernel=[1,1,1]."""
    import segmentation_models_pytorch_3d as smp_3d

    model = smp_3d.create_model(
        "unet",
        encoder_name="resnext50_32x4d",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
        time_kernel=[1, 1, 1],
    )
    model.eval()

    # BCHWD format
    x = torch.randn(1, 1, 16, 16, 8)
    with torch.no_grad():
        y = model(x)

    assert y.shape[0] == 1
    assert y.shape[1] == 1


def test_model_factory_2d(config_2d):
    """build_model with 2D config."""
    from src.models import build_model

    # Use random weights to avoid downloading
    config_2d["model"]["encoder_weights"] = None
    model = build_model(config_2d)

    x = torch.randn(1, 1, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 32)


def test_model_factory_3d(config_3d):
    """build_model with 3D config."""
    from src.models import build_model

    model = build_model(config_3d)

    x = torch.randn(1, 1, 16, 16, 8)
    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.shape[0] == 1
    assert y.shape[1] == 1
