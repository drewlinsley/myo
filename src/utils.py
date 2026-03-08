"""Utility functions: seeding, environment setup, checkpoint save/load."""

import os
import random
import subprocess
import numpy as np
import torch
import yaml


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_config(cfg_file):
    """Read a YAML config file and return its contents."""
    assert cfg_file is not None, "No config file provided."
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_env(mixed_precision=False, seconds=5400):
    """Set up Accelerator with optional mixed precision.

    Returns:
        (accelerator, device, tqdm)
    """
    from functools import partial
    from accelerate import Accelerator, InitProcessGroupKwargs
    from datetime import timedelta
    from tqdm import tqdm as std_tqdm

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=seconds))
    mp = "fp16" if mixed_precision else "no"
    accelerator = Accelerator(
        kwargs_handlers=[process_group_kwargs],
        mixed_precision=mp,
    )
    device = accelerator.device
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    return accelerator, device, tqdm


def get_git_hash():
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_checkpoint(model, optimizer, epoch, val_loss, cfg, path, accelerator=None):
    """Save a training checkpoint."""
    if accelerator is not None:
        unwrapped = accelerator.unwrap_model(model)
    else:
        unwrapped = model

    state = {
        "epoch": epoch,
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
        "git_hash": get_git_hash(),
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint, handling DataParallel 'module.' prefix.

    Returns:
        dict with checkpoint metadata (epoch, val_loss, etc.)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Handle DataParallel prefix
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt


def make_train_val_split(file_stems, val_fraction=0.15, seed=42):
    """Deterministic train/val split using hash-based sorting (stable when adding files).

    Args:
        file_stems: list of filename stems (without extension)
        val_fraction: fraction to hold out for validation
        seed: random seed for split

    Returns:
        (train_stems, val_stems)
    """
    import hashlib

    def _hash(s):
        return hashlib.md5(f"{seed}_{s}".encode()).hexdigest()

    sorted_stems = sorted(file_stems, key=_hash)
    n_val = max(1, int(len(sorted_stems) * val_fraction))
    return sorted_stems[n_val:], sorted_stems[:n_val]
