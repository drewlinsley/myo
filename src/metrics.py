"""Evaluation metrics for per-pixel regression."""

import numpy as np
import torch


def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio.

    Works with both numpy arrays and torch tensors.
    """
    if isinstance(pred, torch.Tensor):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"))
        return 10 * torch.log10(max_val ** 2 / mse)
    else:
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(max_val ** 2 / mse)


def ssim(pred, target, win_size=7, data_range=1.0):
    """Structural Similarity Index (simplified, per-image).

    Uses skimage implementation for numpy arrays.
    For torch tensors, converts to numpy first.
    """
    from skimage.metrics import structural_similarity

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.squeeze()
    target = target.squeeze()

    if pred.ndim == 2:
        return structural_similarity(pred, target, data_range=data_range,
                                     win_size=win_size)
    elif pred.ndim == 3:
        # Average SSIM over slices (first dim)
        vals = []
        for i in range(pred.shape[0]):
            s = structural_similarity(pred[i], target[i], data_range=data_range,
                                      win_size=win_size)
            vals.append(s)
        return np.mean(vals)
    else:
        raise ValueError(f"Expected 2D or 3D arrays, got {pred.ndim}D")


def pearson_corr(pred, target):
    """Pearson correlation coefficient."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.ravel()
    target = target.ravel()

    if pred.std() == 0 or target.std() == 0:
        return 0.0

    return float(np.corrcoef(pred, target)[0, 1])


def mae(pred, target):
    """Mean Absolute Error."""
    if isinstance(pred, torch.Tensor):
        return torch.mean(torch.abs(pred - target))
    return float(np.mean(np.abs(pred - target)))
