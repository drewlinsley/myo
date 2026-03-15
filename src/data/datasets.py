"""Dataset classes for 2D (per-slice) and 3D (volumetric) experiments.

BaseDataset handles file loading, stats loading, and normalization.
SliceDataset indexes by (file_idx, z_idx) for 2D U-Net experiments.
VolumeDataset indexes by (file_idx, patch_idx) for 3D U-Net experiments.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize


class BaseDataset(Dataset):
    """Base class with shared loading and normalization logic.

    Args:
        bf_files: list of brightfield .npy paths
        gfp_files: list of GFP .npy paths
        stats_dir: directory with per-volume JSON stats
        apply_timm: whether to apply TIMM ImageNet normalization to BF input
        transform: augmentation pipeline (applied to concatenated BF+GFP)
        cache_volumes: if True, keep full volumes in RAM
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False, z_range=None):
        assert len(bf_files) == len(gfp_files)
        self.bf_files = bf_files
        self.gfp_files = gfp_files
        self.stats_dir = stats_dir
        self.apply_timm = apply_timm
        self.transform = transform
        self.cache_volumes = cache_volumes
        self.z_range = z_range  # e.g. [70, 105] means Z slices 70..104
        self._cache = {}

        # Load stats for all volumes
        self.stats = []
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                self.stats.append(json.load(f))

    def _load_volume(self, idx):
        """Load and normalize a BF+GFP volume pair."""
        if self.cache_volumes and idx in self._cache:
            return self._cache[idx]

        bf_raw = np.load(self.bf_files[idx], mmap_mode=None if self.cache_volumes else "r")
        gfp_raw = np.load(self.gfp_files[idx], mmap_mode=None if self.cache_volumes else "r")

        # Restrict Z range if specified
        if self.z_range is not None:
            z_lo = max(0, self.z_range[0])
            z_hi = min(bf_raw.shape[0], self.z_range[1])
            bf_raw = bf_raw[z_lo:z_hi]
            gfp_raw = gfp_raw[z_lo:z_hi]

        st = self.stats[idx]
        bf = normalize(bf_raw, st["bf"]["p_low"], st["bf"]["p_high"],
                       apply_timm=self.apply_timm)
        gfp = normalize(gfp_raw, st["gfp"]["p_low"], st["gfp"]["p_high"],
                        apply_timm=False)  # target always [0,1]

        if self.cache_volumes:
            self._cache[idx] = (bf, gfp)

        return bf, gfp


class SliceDataset(BaseDataset):
    """2D dataset: indexes individual Z-slices across all volumes.

    Each sample is a single Z-slice: BF (1, H, W) and GFP (1, H, W).
    Augmentations operate on (H, W, C) numpy arrays where C=2 (BF+GFP concat).
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False, crop_size=256, z_range=None):
        super().__init__(bf_files, gfp_files, stats_dir, apply_timm,
                         transform, cache_volumes, z_range=z_range)
        self.crop_size = crop_size

        # Build index: (file_idx, z_idx) for each sample
        # z_idx is relative to the (possibly z-clipped) volume
        self.index_map = []
        for i, bf_path in enumerate(bf_files):
            bf = np.load(bf_path, mmap_mode="r")
            n_z_total = bf.shape[0]
            if z_range is not None:
                n_z = min(n_z_total, z_range[1]) - max(0, z_range[0])
            else:
                n_z = n_z_total
            for z in range(n_z):
                self.index_map.append((i, z))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, z_idx = self.index_map[idx]
        bf, gfp = self._load_volume(file_idx)

        # Extract single slice: (H, W)
        bf_slice = bf[z_idx]
        gfp_slice = gfp[z_idx]

        # Stack as (H, W, 2) for joint augmentation
        combined = np.stack([bf_slice, gfp_slice], axis=-1)

        if self.transform:
            combined = self.transform(combined)
            # After ToTensor2D: (2, H, W)
            bf_out = combined[:1]   # (1, H, W)
            gfp_out = combined[1:2]  # (1, H, W)
        else:
            bf_out = torch.from_numpy(bf_slice[np.newaxis].copy()).float()
            gfp_out = torch.from_numpy(gfp_slice[np.newaxis].copy()).float()

        return bf_out, gfp_out


class VolumeDataset(BaseDataset):
    """3D dataset: random 3D patches from volumes.

    Each sample is a 3D patch: BF (1, H, W, D) and GFP (1, H, W, D).
    Augmentations operate on (D, H, W, C) numpy arrays where C=2 (BF+GFP concat).
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False,
                 patch_depth=32, crop_size=256, patches_per_volume=32, z_range=None):
        super().__init__(bf_files, gfp_files, stats_dir, apply_timm,
                         transform, cache_volumes, z_range=z_range)
        self.patch_depth = patch_depth
        self.crop_size = crop_size
        self.patches_per_volume = patches_per_volume

        # Build index: (file_idx, patch_idx)
        self.index_map = []
        for i in range(len(bf_files)):
            for p in range(patches_per_volume):
                self.index_map.append((i, p))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, _ = self.index_map[idx]
        bf, gfp = self._load_volume(file_idx)

        # bf, gfp are (Z, H, W) — add channel dim for concat
        # Stack as (Z, H, W, 2) for joint augmentation
        combined = np.stack([bf, gfp], axis=-1)  # (Z, H, W, 2)

        # Pad if volume is smaller than patch size
        z, h, w, c = combined.shape
        pad_z = max(0, self.patch_depth - z)
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_z > 0 or pad_h > 0 or pad_w > 0:
            combined = np.pad(combined, ((0, pad_z), (0, pad_h), (0, pad_w), (0, 0)),
                              mode="reflect")

        if self.transform:
            combined = self.transform(combined)
            # After ToTensor3D: (2, H, W, D) — CHWD format
            bf_out = combined[:1]    # (1, H, W, D)
            gfp_out = combined[1:2]  # (1, H, W, D)
        else:
            # Manual crop and convert
            z, h, w, c = combined.shape
            zd = np.random.randint(0, z - self.patch_depth + 1)
            yh = np.random.randint(0, h - self.crop_size + 1)
            xw = np.random.randint(0, w - self.crop_size + 1)
            patch = combined[zd:zd+self.patch_depth, yh:yh+self.crop_size, xw:xw+self.crop_size]
            # (D, H, W, C) -> (C, H, W, D)
            patch = patch.transpose(3, 1, 2, 0)
            bf_out = torch.from_numpy(patch[:1].copy()).float()
            gfp_out = torch.from_numpy(patch[1:2].copy()).float()

        return bf_out, gfp_out
