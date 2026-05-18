"""Per-slice / per-patch dataset for scalar regression on volume metadata.

Yields (img, target_scalar). Mirrors the structure of
GFPClassificationDataset but with a single float target per volume.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize


class VolumeRegressionDataset(Dataset):
    def __init__(self, files, stats_dir, targets,
                 transform=None, z_range=None, apply_timm=True,
                 percentile_clip=(0.5, 99.5),
                 mode="2d", patch_depth=32, patches_per_volume=32,
                 crop_size=256, modality="bf"):
        self.files = files
        self.stats_dir = stats_dir
        self.transform = transform
        self.z_range = z_range
        self.apply_timm = apply_timm
        self.percentile_clip = tuple(percentile_clip)
        self.mode = mode
        self.patch_depth = patch_depth
        self.patches_per_volume = patches_per_volume
        self.crop_size = crop_size
        self.modality = modality

        self.stats = []
        self.target_vals = []
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                self.stats.append(json.load(f))
            self.target_vals.append(float(targets[stem]))

        self.index_map = []
        self.file_idx_map = []  # parallel: file index for each entry
        if mode == "2d":
            for i, path in enumerate(files):
                vol = np.load(path, mmap_mode="r")
                n_z = vol.shape[0]
                if z_range is not None:
                    n_z = min(n_z, z_range[1]) - max(0, z_range[0])
                for z in range(n_z):
                    self.index_map.append((i, z))
                    self.file_idx_map.append(i)
        else:
            for i in range(len(files)):
                for p in range(patches_per_volume):
                    self.index_map.append((i, p))
                    self.file_idx_map.append(i)
        self._cache = {}

    def __len__(self):
        return len(self.index_map)

    def _load(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]
        raw = np.load(self.files[file_idx])
        if self.z_range is not None:
            z_lo = max(0, self.z_range[0])
            z_hi = min(raw.shape[0], self.z_range[1])
            raw = raw[z_lo:z_hi]
        st = self.stats[file_idx]
        img = normalize(raw, st[self.modality]["p_low"],
                        st[self.modality]["p_high"],
                        apply_timm=self.apply_timm)
        self._cache[file_idx] = img
        return img

    def __getitem__(self, idx):
        file_idx, slot = self.index_map[idx]
        img = self._load(file_idx)
        if self.mode == "2d":
            slc = img[slot]
            if self.transform:
                t = self.transform(slc[..., None])
            else:
                t = torch.from_numpy(slc[np.newaxis].copy()).float()
        else:
            z, h, w = img.shape
            pd, cs = self.patch_depth, self.crop_size
            pad_z = max(0, pd - z)
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            vol = img
            if pad_z or pad_h or pad_w:
                vol = np.pad(vol, ((0, pad_z), (0, pad_h), (0, pad_w)),
                             mode="reflect")
            z, h, w = vol.shape
            zd = np.random.randint(0, z - pd + 1)
            yh = np.random.randint(0, h - cs + 1)
            xw = np.random.randint(0, w - cs + 1)
            patch = vol[zd:zd + pd, yh:yh + cs, xw:xw + cs]
            if self.transform:
                t = self.transform(patch[..., None])
            else:
                t = torch.from_numpy(
                    patch.transpose(1, 2, 0)[np.newaxis].copy()).float()
        return t, float(self.target_vals[file_idx]), int(file_idx)
