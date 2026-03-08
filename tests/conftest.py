"""Shared test fixtures: synthetic volumes, temp files, minimal configs."""

import os
import json
import tempfile
import numpy as np
import pytest


@pytest.fixture
def synthetic_volume():
    """Small synthetic volume (8, 32, 32) uint16 with realistic range."""
    rng = np.random.RandomState(42)
    vol = rng.randint(100, 10000, size=(32, 32, 32), dtype=np.uint16)
    return vol


@pytest.fixture
def synthetic_stats():
    """Stats dict matching synthetic_volume."""
    return {
        "bf": {"p_low": 150.0, "p_high": 9500.0, "mean": 5000.0, "std": 2800.0,
               "min": 100.0, "max": 9999.0},
        "gfp": {"p_low": 120.0, "p_high": 9200.0, "mean": 4500.0, "std": 2600.0,
                "min": 100.0, "max": 9999.0},
    }


@pytest.fixture
def temp_data_dir(synthetic_volume, synthetic_stats):
    """Create a temporary data directory with .npy files and stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bf_dir = os.path.join(tmpdir, "bf")
        gfp_dir = os.path.join(tmpdir, "gfp")
        stats_dir = os.path.join(tmpdir, "stats")
        os.makedirs(bf_dir)
        os.makedirs(gfp_dir)
        os.makedirs(stats_dir)

        # Create two volumes
        for name in ["vol_001", "vol_002"]:
            rng = np.random.RandomState(hash(name) % 2**31)
            bf = rng.randint(100, 10000, size=(32, 32, 32), dtype=np.uint16)
            gfp = rng.randint(100, 10000, size=(32, 32, 32), dtype=np.uint16)
            np.save(os.path.join(bf_dir, f"{name}.npy"), bf)
            np.save(os.path.join(gfp_dir, f"{name}.npy"), gfp)
            with open(os.path.join(stats_dir, f"{name}.json"), "w") as f:
                json.dump(synthetic_stats, f)

        yield tmpdir


@pytest.fixture
def config_2d(temp_data_dir):
    """Minimal 2D config dict."""
    return {
        "seed": 42,
        "experiment_name": "test_2d",
        "data": {
            "data_dir": temp_data_dir,
            "val_fraction": 0.5,
            "crop_size": 16,
            "cache_volumes": False,
        },
        "model": {
            "dims": "2d",
            "arch": "unet",
            "encoder": "resnext50_32x4d",
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "out_channels": 1,
            "time_kernel": None,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "grad_accumulation_steps": 1,
            "mixed_precision": False,
            "losses": {"mse": 1.0, "l1": 0.1},
            "num_workers": 0,
            "checkpoint_dir": os.path.join(temp_data_dir, "ckpts"),
            "save_every": 1,
        },
    }


@pytest.fixture
def config_3d(temp_data_dir):
    """Minimal 3D config dict."""
    return {
        "seed": 42,
        "experiment_name": "test_3d",
        "data": {
            "data_dir": temp_data_dir,
            "val_fraction": 0.5,
            "crop_size": 32,
            "patch_depth": 32,
            "patches_per_volume": 2,
            "cache_volumes": False,
        },
        "model": {
            "dims": "3d",
            "arch": "unet",
            "encoder": "resnext50_32x4d",
            "encoder_weights": None,
            "in_channels": 1,
            "out_channels": 1,
            "time_kernel": [1, 1, 1],
        },
        "training": {
            "epochs": 2,
            "batch_size": 1,
            "lr": 0.001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "grad_accumulation_steps": 1,
            "mixed_precision": False,
            "losses": {"mse": 1.0},
            "num_workers": 0,
            "checkpoint_dir": os.path.join(temp_data_dir, "ckpts"),
            "save_every": 1,
        },
    }
