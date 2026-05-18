"""GFP-encoder + linear regression head (single scalar output)."""

import torch.nn as nn
import segmentation_models_pytorch as smp


class GFPRegressor(nn.Module):
    def __init__(self, encoder_name="resnext50_32x4d",
                 encoder_weights="imagenet",
                 in_channels=1, dims="2d", strides=None):
        super().__init__()
        self.dims = dims
        if dims == "2d":
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name, in_channels=in_channels, depth=5,
                weights=encoder_weights)
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif dims == "3d":
            from segmentation_models_pytorch_3d.encoders import (
                get_encoder as get_encoder_3d)
            kw = {}
            if strides is not None:
                kw["strides"] = tuple(tuple(s) for s in strides)
            self.encoder = get_encoder_3d(
                name=encoder_name, in_channels=in_channels, depth=5,
                weights=encoder_weights, **kw)
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError(f"dims must be '2d' or '3d', got {dims}")
        feat_dim = self.encoder.out_channels[-1]
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.encoder(x)[-1]
        pooled = self.pool(feats).flatten(1)
        return self.head(pooled).squeeze(-1)  # (B,)


def build_gfp_regressor(cfg):
    mcfg = cfg["model"]
    return GFPRegressor(
        encoder_name=mcfg.get("encoder", "resnext50_32x4d"),
        encoder_weights=mcfg.get("encoder_weights", "imagenet"),
        in_channels=mcfg.get("in_channels", 1),
        dims=mcfg.get("dims", "2d"),
        strides=mcfg.get("strides", None),
    )
