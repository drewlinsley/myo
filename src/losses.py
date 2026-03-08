"""Config-driven loss composition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, pred, target):
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for name, weight in self.losses.items():
            if name == "mse":
                total = total + weight * F.mse_loss(pred, target)
            elif name == "l1":
                total = total + weight * F.l1_loss(pred, target)
            elif name == "smooth_l1":
                total = total + weight * F.smooth_l1_loss(pred, target)
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
