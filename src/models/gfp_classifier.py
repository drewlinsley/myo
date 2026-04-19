"""GFP-based two-head classifier.

Single encoder (ResNeXt-50 ImageNet) with two linear classification heads:
  - Exercise  (k_ex classes)
  - Perturbation (k_pt classes)

Per-sample: exactly one of the two labels is valid (the other is -1 / ignored).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class GFPTwoHeadClassifier(nn.Module):
    def __init__(self, encoder_name="resnext50_32x4d", encoder_weights="imagenet",
                 n_exercise=2, n_perturbation=2, in_channels=1):
        super().__init__()
        # Reuse smp's encoder (same init path as the U-Net)
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        feat_dim = self.encoder.out_channels[-1]  # stage 5 feature dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_exercise = nn.Linear(feat_dim, n_exercise)
        self.head_perturbation = nn.Linear(feat_dim, n_perturbation)

    def forward(self, x):
        feats = self.encoder(x)[-1]          # deepest stage: (B, D, H', W')
        pooled = self.pool(feats).flatten(1)  # (B, D)
        return self.head_exercise(pooled), self.head_perturbation(pooled)


def build_gfp_classifier(cfg, n_exercise, n_perturbation):
    mcfg = cfg["model"]
    return GFPTwoHeadClassifier(
        encoder_name=mcfg.get("encoder", "resnext50_32x4d"),
        encoder_weights=mcfg.get("encoder_weights", "imagenet"),
        n_exercise=n_exercise,
        n_perturbation=n_perturbation,
        in_channels=mcfg.get("in_channels", 1),
    )
