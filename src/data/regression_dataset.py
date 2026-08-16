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
            for i, path in enumerate(files):
                n_z = np.load(path, mmap_mode="r").shape[0]
                if z_range is not None:
                    n_z = min(n_z, z_range[1]) - max(0, z_range[0])
                if n_z < 1:
                    # 0 z-planes after the z_range crop — skip this file (mirror
                    # the 2D path) so the downstream count==0 exclusion engages
                    # instead of np.pad crashing on a size-0 axis in __getitem__.
                    continue
                for p in range(patches_per_volume):
                    self.index_map.append((i, p))
                    self.file_idx_map.append(i)
        self._cache = {}

    def __len__(self):
        return len(self.index_map)

    def _load_raw(self, file_idx):
        """Raw (z-cropped) volume, mmap-backed. Only the mmap handle is cached:
        slicing a crop/slice out of it materializes just those pages, and the
        per-volume-stats normalization is element-wise, so normalizing the crop
        afterwards is identical to the old normalize-whole-volume-then-crop —
        without the full-volume read + float32 copy per sample (which also blew
        up worker RAM via the old unbounded normalized-volume cache)."""
        if file_idx in self._cache:
            return self._cache[file_idx]
        raw = np.load(self.files[file_idx], mmap_mode="r")
        if self.z_range is not None:
            z_lo = max(0, self.z_range[0])
            z_hi = min(raw.shape[0], self.z_range[1])
            raw = raw[z_lo:z_hi]
        self._cache[file_idx] = raw
        return raw

    def _normalize(self, patch, file_idx):
        st = self.stats[file_idx]
        return normalize(patch, st[self.modality]["p_low"],
                         st[self.modality]["p_high"],
                         apply_timm=self.apply_timm)

    def __getitem__(self, idx):
        file_idx, slot = self.index_map[idx]
        raw = self._load_raw(file_idx)
        if self.mode == "2d":
            slc = self._normalize(np.asarray(raw[slot]), file_idx)
            # Pad sub-crop FOVs up to crop_size (mirror the 3D branch) so the
            # 2D RandomCrop/CenterCrop don't get a negative crop range.
            cs = self.crop_size
            ph = max(0, cs - slc.shape[0])
            pw = max(0, cs - slc.shape[1])
            if ph or pw:
                slc = np.pad(slc, ((0, ph), (0, pw)), mode="reflect")
            if self.transform:
                t = self.transform(slc[..., None])
            else:
                t = torch.from_numpy(slc[np.newaxis].copy()).float()
        else:
            z, h, w = raw.shape
            pd, cs = self.patch_depth, self.crop_size
            # Same np.random draw order/ranges as the old padded-volume version
            # (max(z, pd) - pd + 1 == max(z - pd, 0) + 1), so seeded eval
            # patches (_eval_det) are bitwise-identical.
            zd = np.random.randint(0, max(z - pd, 0) + 1)
            yh = np.random.randint(0, max(h - cs, 0) + 1)
            xw = np.random.randint(0, max(w - cs, 0) + 1)
            patch = np.asarray(raw[zd:zd + pd, yh:yh + cs, xw:xw + cs])
            pad = ((0, pd - patch.shape[0]), (0, cs - patch.shape[1]),
                   (0, cs - patch.shape[2]))
            if any(p[1] > 0 for p in pad):
                patch = np.pad(patch, pad, mode="reflect")
            patch = self._normalize(patch, file_idx)
            if self.transform:
                t = self.transform(patch[..., None])
            else:
                t = torch.from_numpy(
                    patch.transpose(1, 2, 0)[np.newaxis].copy()).float()
        return t, float(self.target_vals[file_idx]), int(file_idx)
