import numpy as np
import torch
from torch.utils.data import Dataset
from src import video_functional as VF


class BrightfieldFluorescence(Dataset):
    """Dataset for paired brightfield -> fluorescence prediction.

    Expects paired .npy files where each file contains a video stack
    of shape (T, H, W) or (T, H, W, C).

    Args:
        bf_files: list of paths to brightfield .npy stacks
        fl_files: list of paths to corresponding fluorescence .npy stacks
        mu: TIMM channel mean (scalar)
        std: TIMM channel std (scalar)
        transform: video transform (operates on list of frames)
        time: number of frames per sample
        pre_hw: spatial size to resize to before cropping (for augmentation headroom)
        dims: dimension ordering, default "BCDHW"
    """

    def __init__(self, bf_files, fl_files, mu, std, transform=None, time=32, pre_hw=None, dims="BCDHW"):
        assert len(bf_files) == len(fl_files), \
            f"Mismatched file counts: {len(bf_files)} brightfield vs {len(fl_files)} fluorescence"
        self.bf_files = bf_files
        self.fl_files = fl_files
        self.mu = np.float32(mu)
        self.std = np.float32(std)
        self.transform = transform
        self.time = time
        self.pre_hw = pre_hw
        self.dims = dims

    def __len__(self):
        return len(self.bf_files)

    def _load_and_normalize(self, path):
        """Load a .npy stack, ensure 4D (T, H, W, 1), normalize to [0,1] then TIMM stats."""
        data = np.load(path).astype(np.float32)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        elif data.ndim == 4:
            data = data[..., :1]  # keep first channel only

        # Min-max to [0, 1]
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 0:
            data = (data - dmin) / (dmax - dmin)

        # Normalize with TIMM pretrained stats
        data = (data - self.mu) / self.std
        return data

    def __getitem__(self, idx):
        bf = self._load_and_normalize(self.bf_files[idx])
        fl = self._load_and_normalize(self.fl_files[idx])

        # Temporal sampling: pick a random window of `self.time` frames
        min_len = min(len(bf), len(fl))
        if min_len < self.time:
            # Pad temporally if needed
            pad_t = self.time - min_len
            bf = np.pad(bf, ((0, pad_t), (0, 0), (0, 0), (0, 0)))
            fl = np.pad(fl, ((0, pad_t), (0, 0), (0, 0), (0, 0)))
            start = 0
        else:
            start = np.random.randint(0, min_len - self.time + 1)

        bf = bf[start:start + self.time]
        fl = fl[start:start + self.time]

        # Spatial resize if pre_hw is set
        if self.pre_hw is not None:
            bf = np.asarray(VF.resize_clip(bf, self.pre_hw))[..., np.newaxis]
            fl = np.asarray(VF.resize_clip(fl, self.pre_hw))[..., np.newaxis]

        # Concatenate along channel dim so transforms apply identically to both
        # Shape: (T, H, W, 2)
        combined = np.concatenate([bf, fl], axis=-1)

        if self.transform:
            combined = self.transform(combined)
            if self.dims == "BCDHW":
                combined = combined.permute(3, 0, 1, 2)  # (C, T, H, W)
        else:
            combined = torch.from_numpy(np.stack(combined) if isinstance(combined, list) else combined)
            if self.dims == "BCDHW":
                combined = combined.permute(3, 0, 1, 2)

        bf_out = combined[:1].float()   # (1, T, H, W)
        fl_out = combined[1:2].float()  # (1, T, H, W)
        return bf_out, fl_out


class IntraData_train(Dataset):
    """Dataset for 3D segmentation training from a single video + label mask."""

    def __init__(self, bright, label, mu, std, dims, transform=None, time=32):
        label = label[None, ..., None].repeat(len(bright), 0)
        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())

        bright = bright[..., None]
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std

        self.all = np.concatenate((bright, label), -1).astype(np.float32)
        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.all) - self.time - 1

    def __getitem__(self, idx):
        index = torch.arange(idx, idx + self.time)
        sample = self.all[index]
        if self.transform:
            sample = self.transform(sample)
            if self.dims == "BCDHW":
                sample = sample.permute(3, 0, 1, 2)
        m = sample[[0]]
        l = sample[1].long()
        l = l[[-1]]
        return m, l


class IntraData_test(Dataset):
    """Dataset for 3D segmentation inference from a single video."""

    def __init__(self, bright, mu, std, dims, transform=None, time=32):
        self.original = bright.copy()
        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())

        bright = bright[..., None].astype(np.float32)
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std

        self.all = bright
        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.all) - self.time - 1

    def __getitem__(self, idx):
        index = torch.arange(idx, idx + self.time)
        sample = self.all[index]
        o = self.original[index]
        if self.transform:
            sample = self.transform(sample)
            if self.dims == "BCDHW":
                sample = sample.permute(3, 0, 1, 2)
        m = sample[[0]].float()
        return m, o
