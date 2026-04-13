"""Config-driven loss composition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_corr_loss(pred, target, mask=None):
    """1 - Pearson correlation, computed on flattened tensors. Differentiable.

    If mask is provided, only masked pixels contribute.
    """
    if mask is not None:
        sel = mask.bool().reshape(-1)
        pred_flat = pred.reshape(-1)[sel]
        target_flat = target.reshape(-1)[sel]
    else:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    cov = (pred_centered * target_centered).sum()
    pred_std = pred_centered.pow(2).sum().sqrt()
    target_std = target_centered.pow(2).sum().sqrt()
    corr = cov / (pred_std * target_std + 1e-8)
    return 1.0 - corr


class CombinedLoss(nn.Module):
    """Weighted combination of loss functions."""

    def __init__(self, losses_dict):
        """
        Args:
            losses_dict: dict mapping loss name -> weight, e.g. {"mse": 1.0, "l1": 0.1}
        """
        super().__init__()
        self.losses = {}
        for name, weight in losses_dict.items():
            self.losses[name] = weight

    def forward(self, pred, target, mask=None):
        """Compute weighted loss, optionally masked.

        Args:
            pred: model predictions
            target: ground truth
            mask: optional (same spatial shape as pred). 1=foreground, 0=background.
                  When provided, only foreground pixels contribute to the loss.
                  If mask has <1% non-zero pixels, returns zero to avoid noisy gradients.
        """
        # Guard: if mask has <1% foreground, return zero (degenerate crop)
        if mask is not None and mask.sum() < 0.01 * mask.numel():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype,
                                requires_grad=True)

        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for name, weight in self.losses.items():
            if name == "mse":
                if mask is not None:
                    per_pixel = (pred - target).pow(2)
                    loss = (per_pixel * mask).sum() / mask.sum().clamp(min=1)
                else:
                    loss = F.mse_loss(pred, target)
                total = total + weight * loss
            elif name == "l1":
                if mask is not None:
                    per_pixel = (pred - target).abs()
                    loss = (per_pixel * mask).sum() / mask.sum().clamp(min=1)
                else:
                    loss = F.l1_loss(pred, target)
                total = total + weight * loss
            elif name == "smooth_l1":
                if mask is not None:
                    per_pixel = F.smooth_l1_loss(pred, target, reduction="none")
                    loss = (per_pixel * mask).sum() / mask.sum().clamp(min=1)
                else:
                    loss = F.smooth_l1_loss(pred, target)
                total = total + weight * loss
            elif name == "pearson":
                total = total + weight * pearson_corr_loss(pred, target, mask=mask)
            else:
                raise ValueError(f"Unknown loss: {name}")
        return total


def build_loss(cfg):
    """Build loss function from config.

    Config format:
        training:
          losses:
            mse: 1.0
            l1: 0.1
    """
    losses_dict = cfg["training"].get("losses", {"mse": 1.0, "l1": 0.1})
    return CombinedLoss(losses_dict)
