import os
import numpy as np
from functools import partial
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm as std_tqdm
from timm.data import resolve_data_config
import torch
import yaml


def read_config(cfg_file):
    """Read a YAML config file and return its contents."""
    assert cfg_file is not None, "No config file provided."
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def prepare_env(timm_model, seconds=5400):
    """Set up accelerator, device, tqdm, and TIMM preprocessing config."""
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=seconds))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    TIMM = resolve_data_config({}, model=timm_model)
    return accelerator, device, tqdm, TIMM


def normalize(x, mu, std, eps=1e-8):
    """Z-score normalize, then min-max to [0,1], then normalize to TIMM pretrained stats.

    Args:
        x: numpy array of image data
        mu: TIMM channel mean (scalar or array)
        std: TIMM channel std (scalar or array)
    Returns:
        Normalized array (float32)
    """
    x = x.astype(np.float32)
    x = (x - x.mean()) / (x.std() + eps)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    x = (x - mu) / std
    return x


def load_checkpoint(path, model, exclude_keys=None):
    """Load a checkpoint, handling DataParallel 'module.' prefix and optional key exclusion.

    Args:
        path: path to .pth file
        model: nn.Module to load weights into
        exclude_keys: list of substrings; keys containing any of these are excluded
    """
    weights = torch.load(path, map_location="cpu")
    first_key = next(iter(weights))
    if first_key.startswith("module."):
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
    if exclude_keys:
        weights = {k: v for k, v in weights.items()
                   if not any(ex in k for ex in exclude_keys)}
    model.load_state_dict(weights, strict=False)
    return model


def pad_image_3d(x, hw):
    """Pad 3D volume (T, H, W, C) to spatial size hw x hw."""
    xshape = x.shape
    hdiff, wdiff = hw - xshape[1], hw - xshape[2]
    if hdiff == 0 and wdiff == 0:
        return x
    pad = ((0, 0), (0, hdiff), (0, wdiff), (0, 0))
    return np.pad(x, pad, mode='constant')
