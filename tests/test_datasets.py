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
    bf, gfp, _mask = ds[0]
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
    bf, gfp, _mask = ds[0]
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

    bf1, gfp1, _m1 = ds[0]
    bf2, gfp2, _m2 = ds[0]

    # Same index should give same result with deterministic transforms
    torch.testing.assert_close(bf1, bf2)
    torch.testing.assert_close(gfp1, gfp2)


def test_volume_dataset_len_is_built_in_init(temp_data_dir):
    """Regression guard: index_map must be built in __init__.

    It once sat below a `return` inside _fast_ok, making it unreachable, so
    __len__ raised AttributeError and 3D BF->GFP training could not start.
    """
    from glob import glob
    import os

    bf_files = sorted(glob(os.path.join(temp_data_dir, "bf", "*.npy")))
    gfp_files = sorted(glob(os.path.join(temp_data_dir, "gfp", "*.npy")))
    stats_dir = os.path.join(temp_data_dir, "stats")

    ds = VolumeDataset(bf_files, gfp_files, stats_dir, apply_timm=False,
                       transform=None, patch_depth=8, crop_size=16,
                       patches_per_volume=3)
    assert hasattr(ds, "index_map"), "index_map must exist after __init__"
    assert len(ds) == 3 * len(bf_files)


def test_global_percentiles_pools_across_volumes(temp_data_dir):
    """global_percentiles must span every volume, and refuse a silent subset."""
    import os
    import pytest
    from src.data.normalization import global_percentiles

    stats_dir = os.path.join(temp_data_dir, "stats")
    lo, hi = global_percentiles(stats_dir, "gfp")
    assert lo <= hi
    # a stem with no stats JSON must raise, not silently narrow the statistic
    with pytest.raises(ValueError):
        global_percentiles(stats_dir, "gfp", stems=["vol_001", "does_not_exist"])


def test_regression_dataset_global_scope_requires_explicit_pct(temp_data_dir):
    """norm_scope='global' without global_pct would pool over held-out volumes."""
    import os
    import pytest
    from src.data.regression_dataset import VolumeRegressionDataset

    files = [os.path.join(temp_data_dir, "gfp", "vol_001.npy")]
    stats_dir = os.path.join(temp_data_dir, "stats")
    with pytest.raises(ValueError):
        VolumeRegressionDataset(files, stats_dir, {"vol_001": 1.0},
                                norm_scope="global", modality="gfp")
