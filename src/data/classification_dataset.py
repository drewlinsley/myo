"""Per-slice GFP classification dataset with two label heads."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize


CONTROL_KEYWORDS = {"control", "ctrl", "no", "none", "untreated", "vehicle",
                    "unstimulated", "baseline", "wt", "wild type", "dmso"}


def binarize(label):
    """Map raw label string → 'Control' or 'Perturbed'. None stays None."""
    if label is None:
        return None
    words = set(str(label).strip().lower().split())
    return "Control" if (words & CONTROL_KEYWORDS) else "Perturbed"


class GFPClassificationDataset(Dataset):
    """Per-slice GFP dataset yielding (gfp, exercise_idx, perturbation_idx).

    Missing labels are -1 (ignored by CrossEntropyLoss when ignore_index=-1).

    Args:
        bf_files: list of BF .npy paths (used only for stats lookup — unused at runtime)
        gfp_files: list of GFP .npy paths
        stats_dir: dir with per-volume JSON stats
        metadata: dict {stem: {"Exercise": str|None, "Perturbation": str|None, ...}}
        label_vocab: {"exercise": [classes], "perturbation": [classes]}
        transform: augmentation pipeline
        z_range: optional [lo, hi] z clipping
        apply_timm: apply TIMM ImageNet normalization to input
    """

    def __init__(self, gfp_files, stats_dir, metadata, label_vocab,
                 transform=None, z_range=None, apply_timm=True,
                 percentile_clip=(0.5, 99.5), use_raw_labels=False,
                 mode="2d", patch_depth=32, patches_per_volume=32,
                 crop_size=256):
        self.gfp_files = gfp_files
        self.stats_dir = stats_dir
        self.metadata = metadata
        self.label_vocab = label_vocab
        self.transform = transform
        self.z_range = z_range
        self.apply_timm = apply_timm
        self.percentile_clip = tuple(percentile_clip)
        self.mode = mode
        self.patch_depth = patch_depth
        self.patches_per_volume = patches_per_volume
        self.crop_size = crop_size

        # Load stats and compute per-volume labels
        self.stats = []
        self.ex_idx = []
        self.pt_idx = []
        for gfp_path in gfp_files:
            stem = os.path.splitext(os.path.basename(gfp_path))[0]
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                self.stats.append(json.load(f))
            meta = metadata.get(stem, {})
            if use_raw_labels:
                ex = meta.get("Exercise") or None
                pt = meta.get("Perturbation") or None
            else:
                ex = binarize(meta.get("Exercise"))
                pt = binarize(meta.get("Perturbation"))
            self.ex_idx.append(
                label_vocab["exercise"].index(ex) if ex in label_vocab["exercise"] else -1)
            self.pt_idx.append(
                label_vocab["perturbation"].index(pt) if pt in label_vocab["perturbation"] else -1)

        # Build index map
        self.index_map = []
        if mode == "2d":
            for i, gfp_path in enumerate(gfp_files):
                vol = np.load(gfp_path, mmap_mode="r")
                n_z_total = vol.shape[0]
                if z_range is not None:
                    n_z = min(n_z_total, z_range[1]) - max(0, z_range[0])
                else:
                    n_z = n_z_total
                for z in range(n_z):
                    self.index_map.append((i, z))
        else:  # 3d: random patches per volume
            for i in range(len(gfp_files)):
                for p in range(patches_per_volume):
                    self.index_map.append((i, p))

        self._cache = {}

    def _load(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]
        gfp_raw = np.load(self.gfp_files[file_idx])
        if self.z_range is not None:
            z_lo = max(0, self.z_range[0])
            z_hi = min(gfp_raw.shape[0], self.z_range[1])
            gfp_raw = gfp_raw[z_lo:z_hi]
        st = self.stats[file_idx]
        gfp = normalize(gfp_raw, st["gfp"]["p_low"], st["gfp"]["p_high"],
                        apply_timm=self.apply_timm)
        self._cache[file_idx] = gfp
        return gfp

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, slot = self.index_map[idx]
        gfp = self._load(file_idx)

        if self.mode == "2d":
            slc = gfp[slot]  # (H, W)
            if self.transform:
                img = self.transform(slc[..., None])  # -> (1, H, W)
            else:
                img = torch.from_numpy(slc[np.newaxis].copy()).float()
        else:
            # 3D: random (D, H, W) patch, transformed to (1, H, W, D) tensor
            z, h, w = gfp.shape
            pd, cs = self.patch_depth, self.crop_size
            # Reflect-pad if volume is smaller than patch size
            pad_z = max(0, pd - z)
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            vol = gfp
            if pad_z or pad_h or pad_w:
                vol = np.pad(vol, ((0, pad_z), (0, pad_h), (0, pad_w)),
                             mode="reflect")
            z, h, w = vol.shape
            zd = np.random.randint(0, z - pd + 1)
            yh = np.random.randint(0, h - cs + 1)
            xw = np.random.randint(0, w - cs + 1)
            patch = vol[zd:zd + pd, yh:yh + cs, xw:xw + cs]  # (D, H, W)
            if self.transform:
                # transforms expect (D, H, W, C)
                img = self.transform(patch[..., None])  # -> (1, H, W, D)
            else:
                # (D, H, W) -> (1, H, W, D)
                img = torch.from_numpy(
                    patch.transpose(1, 2, 0)[np.newaxis].copy()).float()

        return img, int(self.ex_idx[file_idx]), int(self.pt_idx[file_idx])


def build_label_vocab(metadata, stems):
    """Build class lists for Exercise and Perturbation from binarized labels."""
    ex_set, pt_set = set(), set()
    for s in stems:
        meta = metadata.get(s, {})
        ex = binarize(meta.get("Exercise"))
        pt = binarize(meta.get("Perturbation"))
        if ex is not None:
            ex_set.add(ex)
        if pt is not None:
            pt_set.add(pt)
    # Stable ordering: Control first when present
    def order(s):
        xs = sorted(s)
        if "Control" in xs:
            xs.remove("Control")
            xs = ["Control"] + xs
        return xs
    return {"exercise": order(ex_set), "perturbation": order(pt_set)}
