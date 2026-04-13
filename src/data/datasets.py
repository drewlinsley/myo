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

from src.data.normalization import normalize, normalize_auto


class BaseDataset(Dataset):
    """Base class with shared loading and normalization logic.

    Args:
        bf_files: list of brightfield .npy paths
        gfp_files: list of GFP .npy paths
        stats_dir: directory with per-volume JSON stats
        apply_timm: whether to apply TIMM ImageNet normalization to BF input
        transform: augmentation pipeline (applied to concatenated BF+GFP)
        cache_volumes: if True, keep full volumes in RAM
        gfp_norm_mode: 'volume' | 'per_z' | 'per_patch' — when to normalize GFP
        filter_empty_gfp: if True, skip samples where mean GFP < threshold
        empty_gfp_threshold: mean-GFP threshold for filtering
        percentile_clip: (low, high) percentile bounds for normalize_auto
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False, z_range=None,
                 gfp_norm_mode="volume", filter_empty_gfp=False,
                 empty_gfp_threshold=0.01, percentile_clip=(0.5, 99.5),
                 mask_threshold_method=None):
        assert len(bf_files) == len(gfp_files)
        self.bf_files = bf_files
        self.gfp_files = gfp_files
        self.stats_dir = stats_dir
        self.apply_timm = apply_timm
        self.transform = transform
        self.cache_volumes = cache_volumes
        self.z_range = z_range  # e.g. [70, 105] means Z slices 70..104
        self.gfp_norm_mode = gfp_norm_mode
        self.filter_empty_gfp = filter_empty_gfp
        self.empty_gfp_threshold = empty_gfp_threshold
        self.percentile_clip = tuple(percentile_clip)
        self.mask_threshold_method = mask_threshold_method  # None | "minimum" | "otsu" | "li" | "triangle"
        self._cache = {}

        # Load stats for all volumes
        self.stats = []
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]
            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                self.stats.append(json.load(f))

    def _load_volume(self, idx):
        """Load and normalize a BF+GFP volume pair, optionally with a mask.

        GFP normalization depends on gfp_norm_mode:
          - 'volume': normalize with pre-computed volume stats (current default)
          - 'per_z' / 'per_patch': store raw float32 GFP, normalize later
        BF is always normalized with volume stats.

        Returns:
            (bf, gfp, mask) — mask is None when mask_threshold_method is None.
        """
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

        # Compute foreground mask from raw BF before normalization
        mask = None
        if self.mask_threshold_method is not None:
            from skimage.filters import (
                threshold_otsu, threshold_li, threshold_triangle,
                threshold_minimum,
            )
            flat = bf_raw.ravel().astype(np.float64)
            _algo = {"minimum": threshold_minimum,
                     "otsu": threshold_otsu,
                     "li": threshold_li,
                     "triangle": threshold_triangle}
            thresh = float(_algo[self.mask_threshold_method](flat))
            mask = (bf_raw > thresh)  # (Z, H, W) bool

        st = self.stats[idx]
        bf = normalize(bf_raw, st["bf"]["p_low"], st["bf"]["p_high"],
                       apply_timm=self.apply_timm)

        if self.gfp_norm_mode == "volume":
            gfp = normalize(gfp_raw, st["gfp"]["p_low"], st["gfp"]["p_high"],
                            apply_timm=False)
        else:
            # per_z / per_patch: keep raw float32, normalize in __getitem__
            gfp = gfp_raw.astype(np.float32)

        if self.cache_volumes:
            self._cache[idx] = (bf, gfp, mask)

        return bf, gfp, mask

    def _normalize_patch_gfp(self, gfp_tensor):
        """Percentile-normalize a cropped GFP tensor to [0,1] using its own stats.

        Args:
            gfp_tensor: torch.Tensor, e.g. (1, H, W) or (1, H, W, D)

        Returns:
            Normalized tensor in [0, 1]
        """
        arr = gfp_tensor.numpy()
        normed, _, _ = normalize_auto(arr, self.percentile_clip)
        return torch.from_numpy(normed)


class SliceDataset(BaseDataset):
    """2D dataset: indexes individual Z-slices across all volumes.

    Each sample is a single Z-slice: BF (1, H, W) and GFP (1, H, W).
    Augmentations operate on (H, W, C) numpy arrays where C=2 (BF+GFP concat).
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False, crop_size=256, z_range=None,
                 gfp_norm_mode="volume", filter_empty_gfp=False,
                 empty_gfp_threshold=0.01, percentile_clip=(0.5, 99.5),
                 mask_threshold_method=None):
        super().__init__(bf_files, gfp_files, stats_dir, apply_timm,
                         transform, cache_volumes, z_range=z_range,
                         gfp_norm_mode=gfp_norm_mode,
                         filter_empty_gfp=filter_empty_gfp,
                         empty_gfp_threshold=empty_gfp_threshold,
                         percentile_clip=percentile_clip,
                         mask_threshold_method=mask_threshold_method)
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

        # Filter out slices where normalized GFP is mostly empty
        if filter_empty_gfp:
            n_before = len(self.index_map)
            filtered = []
            for file_idx, z_idx in self.index_map:
                _, gfp, _ = self._load_volume(file_idx)
                gfp_slice = gfp[z_idx]
                # For volume mode, GFP is already [0,1]; for per_z/per_patch, auto-normalize
                if self.gfp_norm_mode != "volume":
                    gfp_slice, _, _ = normalize_auto(gfp_slice, self.percentile_clip)
                if gfp_slice.mean() >= empty_gfp_threshold:
                    filtered.append((file_idx, z_idx))
            self.index_map = filtered
            self.n_filtered = n_before - len(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, z_idx = self.index_map[idx]
        bf, gfp, mask = self._load_volume(file_idx)

        # Extract single slice: (H, W)
        bf_slice = bf[z_idx]
        gfp_slice = gfp[z_idx]

        # per_z: normalize this slice's GFP before combining
        if self.gfp_norm_mode == "per_z":
            gfp_slice, _, _ = normalize_auto(gfp_slice, self.percentile_clip)

        if mask is not None:
            mask_slice = mask[z_idx].astype(np.float32)
            # Stack as (H, W, 3) for joint augmentation: BF, GFP, mask
            combined = np.stack([bf_slice, gfp_slice, mask_slice], axis=-1)
        else:
            # Stack as (H, W, 2) for joint augmentation
            combined = np.stack([bf_slice, gfp_slice], axis=-1)

        if self.transform:
            combined = self.transform(combined)
            # After ToTensor2D: (C, H, W) where C=2 or 3
            bf_out = combined[:1]   # (1, H, W)
            gfp_out = combined[1:2]  # (1, H, W)
            if mask is not None:
                mask_out = (combined[2:3] > 0.5).float()  # (1, H, W)
            else:
                mask_out = torch.ones_like(gfp_out)
        else:
            bf_out = torch.from_numpy(bf_slice[np.newaxis].copy()).float()
            gfp_out = torch.from_numpy(gfp_slice[np.newaxis].copy()).float()
            if mask is not None:
                mask_out = torch.from_numpy(
                    mask_slice[np.newaxis].copy()).float()
                mask_out = (mask_out > 0.5).float()
            else:
                mask_out = torch.ones_like(gfp_out)

        # per_patch: normalize GFP after crop/transform using patch stats
        if self.gfp_norm_mode == "per_patch":
            gfp_out = self._normalize_patch_gfp(gfp_out)

        return bf_out, gfp_out, mask_out


class VolumeDataset(BaseDataset):
    """3D dataset: random 3D patches from volumes.

    Each sample is a 3D patch: BF (1, H, W, D) and GFP (1, H, W, D).
    Augmentations operate on (D, H, W, C) numpy arrays where C=2 (BF+GFP concat).
    """

    def __init__(self, bf_files, gfp_files, stats_dir, apply_timm=True,
                 transform=None, cache_volumes=False,
                 patch_depth=32, crop_size=256, patches_per_volume=32, z_range=None,
                 gfp_norm_mode="volume", filter_empty_gfp=False,
                 empty_gfp_threshold=0.01, percentile_clip=(0.5, 99.5),
                 mask_threshold_method=None):
        super().__init__(bf_files, gfp_files, stats_dir, apply_timm,
                         transform, cache_volumes, z_range=z_range,
                         gfp_norm_mode=gfp_norm_mode,
                         filter_empty_gfp=filter_empty_gfp,
                         empty_gfp_threshold=empty_gfp_threshold,
                         percentile_clip=percentile_clip,
                         mask_threshold_method=mask_threshold_method)
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

    def _extract_patch(self, file_idx):
        """Extract a single BF/GFP/mask patch triple from a volume.

        Returns (bf_out, gfp_out, mask_out) tensors. mask_out is all-ones
        when masking is disabled.
        """
        bf, gfp, mask = self._load_volume(file_idx)

        # per_z: normalize each Z-slice of raw GFP independently
        if self.gfp_norm_mode == "per_z":
            gfp_normed = np.empty_like(gfp)
            for zi in range(gfp.shape[0]):
                gfp_normed[zi], _, _ = normalize_auto(gfp[zi], self.percentile_clip)
            gfp = gfp_normed

        if mask is not None:
            mask_float = mask.astype(np.float32)
            # Stack as (Z, H, W, 3) for joint augmentation: BF, GFP, mask
            combined = np.stack([bf, gfp, mask_float], axis=-1)
        else:
            # Stack as (Z, H, W, 2)
            combined = np.stack([bf, gfp], axis=-1)

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
            # After ToTensor3D: (C, H, W, D) where C=2 or 3
            bf_out = combined[:1]    # (1, H, W, D)
            gfp_out = combined[1:2]  # (1, H, W, D)
            if mask is not None:
                mask_out = (combined[2:3] > 0.5).float()  # (1, H, W, D)
            else:
                mask_out = torch.ones_like(gfp_out)
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
            if mask is not None:
                mask_out = torch.from_numpy(patch[2:3].copy()).float()
                mask_out = (mask_out > 0.5).float()
            else:
                mask_out = torch.ones_like(gfp_out)

        # per_patch: normalize GFP after crop using patch stats
        if self.gfp_norm_mode == "per_patch":
            gfp_out = self._normalize_patch_gfp(gfp_out)

        return bf_out, gfp_out, mask_out

    def __getitem__(self, idx):
        file_idx, _ = self.index_map[idx]

        if self.filter_empty_gfp:
            # Retry loop: resample if GFP patch is empty
            max_attempts = 50
            for attempt in range(max_attempts):
                bf_out, gfp_out, mask_out = self._extract_patch(file_idx)
                if gfp_out.mean().item() >= self.empty_gfp_threshold:
                    return bf_out, gfp_out, mask_out
                # After several failures, try a different volume
                if attempt == max_attempts // 2:
                    file_idx = np.random.randint(0, len(self.bf_files))
            # Exhausted attempts — return last patch anyway
            return bf_out, gfp_out, mask_out
        else:
            return self._extract_patch(file_idx)
