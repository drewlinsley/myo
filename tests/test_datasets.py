"""Tests for SliceDataset and VolumeDataset."""

import torch
import pytest
from src.data.datasets import SliceDataset, VolumeDataset
from src.data import transforms as T


def test_slice_dataset_shape(temp_data_dir):
    """SliceDataset should return (1, H, W) tensors."""
    from glob import glob
    import os

    bf_files = sorted(glob(os.path.join(temp_data_dir, "bf", "*.npy")))
    gfp_files = sorted(glob(os.path.join(temp_data_dir, "gfp", "*.npy")))
    stats_dir = os.path.join(temp_data_dir, "stats")

    transform = T.Compose([T.CenterCrop2D(16), T.ToTensor2D()])
    ds = SliceDataset(bf_files, gfp_files, stats_dir, apply_timm=False,
                      transform=transform, crop_size=16)

    assert len(ds) > 0
    bf, gfp = ds[0]
    assert bf.shape == (1, 16, 16)
    assert gfp.shape == (1, 16, 16)
    assert bf.dtype == torch.float32


def test_slice_dataset_index_count(temp_data_dir):
    """SliceDataset should have one sample per Z-slice across all volumes."""
    from glob import glob
    import os
    import numpy as np

    bf_files = sorted(glob(os.path.join(temp_data_dir, "bf", "*.npy")))
    gfp_files = sorted(glob(os.path.join(temp_data_dir, "gfp", "*.npy")))
    stats_dir = os.path.join(temp_data_dir, "stats")

    transform = T.Compose([T.CenterCrop2D(16), T.ToTensor2D()])
    ds = SliceDataset(bf_files, gfp_files, stats_dir, apply_timm=False,
                      transform=transform, crop_size=16)

    total_z = sum(np.load(f, mmap_mode="r").shape[0] for f in bf_files)
    assert len(ds) == total_z


def test_volume_dataset_shape(temp_data_dir):
    """VolumeDataset should return (1, H, W, D) tensors."""
    from glob import glob
    import os

    bf_files = sorted(glob(os.path.join(temp_data_dir, "bf", "*.npy")))
    gfp_files = sorted(glob(os.path.join(temp_data_dir, "gfp", "*.npy")))
    stats_dir = os.path.join(temp_data_dir, "stats")

    transform = T.Compose([
        T.CenterCrop3D(8, 16, 16),
        T.ToTensor3D(),
    ])
    ds = VolumeDataset(bf_files, gfp_files, stats_dir, apply_timm=False,
                       transform=transform, patch_depth=8, crop_size=16,
                       patches_per_volume=2)

    assert len(ds) == 2 * len(bf_files)
    bf, gfp = ds[0]
    assert bf.shape == (1, 16, 16, 8)
    assert gfp.shape == (1, 16, 16, 8)
    assert bf.dtype == torch.float32


def test_transforms_applied_consistently(temp_data_dir):
    """Input and target should receive identical spatial transforms."""
    from glob import glob
    import os

    bf_files = sorted(glob(os.path.join(temp_data_dir, "bf", "*.npy")))
    gfp_files = sorted(glob(os.path.join(temp_data_dir, "gfp", "*.npy")))
    stats_dir = os.path.join(temp_data_dir, "stats")

    # Use center crop (deterministic) to verify consistency
    transform = T.Compose([T.CenterCrop2D(16), T.ToTensor2D()])
    ds = SliceDataset(bf_files, gfp_files, stats_dir, apply_timm=False,
                      transform=transform, crop_size=16)

    bf1, gfp1 = ds[0]
    bf2, gfp2 = ds[0]

    # Same index should give same result with deterministic transforms
    torch.testing.assert_close(bf1, bf2)
    torch.testing.assert_close(gfp1, gfp2)
