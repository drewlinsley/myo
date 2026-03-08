"""Dataset for pix2pix-turbo: per-slice 2D pairs with tokenized prompt.

Indexes by (file_idx, z_idx), same as SliceDataset, but:
  - Outputs (3, 512, 512) images (grayscale replicated to RGB)
  - BF in [0, 1], GFP in [-1, 1] (VAE convention)
  - No TIMM normalization (SD-Turbo VAE has its own)
  - Returns dict with conditioning_pixel_values, output_pixel_values, input_ids
"""

import random
import numpy as np
import torch

from src.data.datasets import BaseDataset


class SliceDatasetPix2Pix(BaseDataset):
    """2D dataset for pix2pix-turbo training.

    Args:
        bf_files: list of brightfield .npy paths
        gfp_files: list of GFP .npy paths
        stats_dir: directory with per-volume JSON stats
        tokenizer: HuggingFace tokenizer for text prompt
        prompt: text prompt describing the translation task
        crop_size: spatial crop size (default 512)
        train: if True, apply random augmentations; otherwise center crop
        cache_volumes: if True, keep volumes in RAM
    """

    def __init__(
        self,
        bf_files,
        gfp_files,
        stats_dir,
        tokenizer,
        prompt="brightfield to GFP fluorescence",
        crop_size=512,
        train=True,
        cache_volumes=False,
    ):
        super().__init__(
            bf_files,
            gfp_files,
            stats_dir,
            apply_timm=False,  # no TIMM stats for SD-Turbo
            transform=None,
            cache_volumes=cache_volumes,
        )
        self.crop_size = crop_size
        self.train = train

        # Pre-tokenize the prompt (same for all samples)
        self.input_ids = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]  # (seq_len,)

        # Build index: (file_idx, z_idx)
        self.index_map = []
        for i, bf_path in enumerate(bf_files):
            bf = np.load(bf_path, mmap_mode="r")
            n_z = bf.shape[0]
            for z in range(n_z):
                self.index_map.append((i, z))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, z_idx = self.index_map[idx]
        bf, gfp = self._load_volume(file_idx)

        bf_slice = bf[z_idx].copy()   # (H, W) in [0, 1]
        gfp_slice = gfp[z_idx].copy()  # (H, W) in [0, 1]

        h, w = bf_slice.shape

        # Pad if smaller than crop_size
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h > 0 or pad_w > 0:
            bf_slice = np.pad(bf_slice, ((0, pad_h), (0, pad_w)), mode="reflect")
            gfp_slice = np.pad(gfp_slice, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = bf_slice.shape

        # Crop
        if self.train:
            y = random.randint(0, h - self.crop_size)
            x = random.randint(0, w - self.crop_size)
        else:
            y = (h - self.crop_size) // 2
            x = (w - self.crop_size) // 2
        bf_slice = bf_slice[y : y + self.crop_size, x : x + self.crop_size]
        gfp_slice = gfp_slice[y : y + self.crop_size, x : x + self.crop_size]

        # Augmentation (train only)
        if self.train:
            if random.random() > 0.5:
                bf_slice = np.ascontiguousarray(bf_slice[:, ::-1])
                gfp_slice = np.ascontiguousarray(gfp_slice[:, ::-1])
            if random.random() > 0.5:
                bf_slice = np.ascontiguousarray(bf_slice[::-1])
                gfp_slice = np.ascontiguousarray(gfp_slice[::-1])
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                bf_slice = np.ascontiguousarray(np.rot90(bf_slice, k))
                gfp_slice = np.ascontiguousarray(np.rot90(gfp_slice, k))

        # Replicate grayscale to 3 channels: (3, H, W)
        bf_3ch = np.stack([bf_slice, bf_slice, bf_slice], axis=0)  # [0, 1]
        gfp_3ch = np.stack([gfp_slice, gfp_slice, gfp_slice], axis=0)  # [0, 1]

        # BF input: scale to [-1, 1] for SD-Turbo VAE
        conditioning = torch.from_numpy(bf_3ch).float() * 2.0 - 1.0  # [-1, 1]
        # GFP target: scale to [-1, 1] for VAE convention
        output = torch.from_numpy(gfp_3ch).float() * 2.0 - 1.0  # [-1, 1]

        return {
            "conditioning_pixel_values": conditioning,
            "output_pixel_values": output,
            "input_ids": self.input_ids,
        }
