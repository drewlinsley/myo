"""2D and 3D augmentations for microscopy data.

2D transforms operate on (H, W, C) numpy arrays.
3D transforms operate on (D, H, W, C) numpy arrays.
All transforms apply identically to input and target (concatenated along C).
"""

import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 2D Transforms (for SliceDataset)
# ---------------------------------------------------------------------------

class RandomCrop2D:
    """Random spatial crop of (H, W, C) array."""
    def __init__(self, size):
        self.h = size if isinstance(size, int) else size[0]
        self.w = size if isinstance(size, int) else size[1]

    def __call__(self, img):
        h, w = img.shape[:2]
        y = random.randint(0, h - self.h)
        x = random.randint(0, w - self.w)
        return img[y:y+self.h, x:x+self.w]


class CenterCrop2D:
    """Center crop of (H, W, C) array."""
    def __init__(self, size):
        self.h = size if isinstance(size, int) else size[0]
        self.w = size if isinstance(size, int) else size[1]

    def __call__(self, img):
        h, w = img.shape[:2]
        y = (h - self.h) // 2
        x = (w - self.w) // 2
        return img[y:y+self.h, x:x+self.w]


class RandomHFlip2D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.ascontiguousarray(img[:, ::-1])
        return img


class RandomVFlip2D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.ascontiguousarray(img[::-1])
        return img


class RandomRot90_2D:
    """Random 90-degree rotation in the HW plane."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
        return img


class IntensityJitter2D:
    """Small multiplicative + additive intensity jitter (applied to first N channels only)."""
    def __init__(self, n_input_channels=1, brightness=0.05, contrast=0.05):
        self.n = n_input_channels
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        img = img.copy()
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness)
        img[..., :self.n] = img[..., :self.n] * alpha + beta
        return img


class ToTensor2D:
    """Convert (H, W, C) numpy array to (C, H, W) float32 tensor."""
    def __call__(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1).copy()).float()


class Compose:
    """Chain multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# 3D Transforms (for VolumeDataset)
# ---------------------------------------------------------------------------

class RandomCrop3D:
    """Random 3D crop of (D, H, W, C) array."""
    def __init__(self, depth, height, width):
        self.d = depth
        self.h = height
        self.w = width

    def __call__(self, vol):
        d, h, w = vol.shape[:3]
        zd = random.randint(0, d - self.d)
        yh = random.randint(0, h - self.h)
        xw = random.randint(0, w - self.w)
        return vol[zd:zd+self.d, yh:yh+self.h, xw:xw+self.w]


class CenterCrop3D:
    """Center 3D crop of (D, H, W, C) array."""
    def __init__(self, depth, height, width):
        self.d = depth
        self.h = height
        self.w = width

    def __call__(self, vol):
        d, h, w = vol.shape[:3]
        zd = (d - self.d) // 2
        yh = (h - self.h) // 2
        xw = (w - self.w) // 2
        return vol[zd:zd+self.d, yh:yh+self.h, xw:xw+self.w]


class RandomHFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol):
        if random.random() < self.p:
            return np.ascontiguousarray(vol[:, :, ::-1])
        return vol


class RandomVFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol):
        if random.random() < self.p:
            return np.ascontiguousarray(vol[:, ::-1])
        return vol


class RandomZFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol):
        if random.random() < self.p:
            return np.ascontiguousarray(vol[::-1])
        return vol


class RandomRot90_3D:
    """Random 90-degree rotation in the XY plane."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            return np.ascontiguousarray(np.rot90(vol, k, axes=(1, 2)))
        return vol


class IntensityJitter3D:
    """Small multiplicative + additive intensity jitter for 3D volumes."""
    def __init__(self, n_input_channels=1, brightness=0.05, contrast=0.05):
        self.n = n_input_channels
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, vol):
        vol = vol.copy()
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness)
        vol[..., :self.n] = vol[..., :self.n] * alpha + beta
        return vol


class ToTensor3D:
    """Convert (D, H, W, C) numpy array to (C, H, W, D) float32 tensor.

    The model expects (B, C, H, W, D) — BCHWD format.
    """
    def __call__(self, vol):
        # (D, H, W, C) -> (C, H, W, D)
        return torch.from_numpy(vol.transpose(3, 1, 2, 0).copy()).float()
