"""Model factory: build 2D or 3D U-Net based on config."""

import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """Create 2D or 3D U-Net based on config.

    Args:
        cfg: full config dict with model.dims, model.encoder, etc.

    Returns:
        torch.nn.Module
    """
    mcfg = cfg["model"]
    dims = mcfg["dims"]
    encoder_name = mcfg["encoder"]
    encoder_weights = mcfg.get("encoder_weights")  # None for random init
    in_channels = mcfg["in_channels"]
    out_channels = mcfg["out_channels"]

    if dims == "2d":
        import segmentation_models_pytorch as smp_2d
        model = smp_2d.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    elif dims == "3d":
        import segmentation_models_pytorch_3d as smp_3d
        time_kernel = mcfg.get("time_kernel", [1, 1, 1])
        strides = mcfg.get("strides",
                           ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)))
        # Convert strides to tuples if they're lists
        strides = tuple(tuple(s) for s in strides)
        model = smp_3d.create_model(
            "unet",
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
            time_kernel=time_kernel,
            strides=strides,
        )
    else:
        raise ValueError(f"model.dims must be '2d' or '3d', got '{dims}'")

    return model
