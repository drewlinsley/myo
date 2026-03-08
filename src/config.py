"""Config loading: base + override YAML merge with validation."""

import os
import copy
import yaml


def deep_merge(base, override):
    """Recursively merge override dict into base dict (returns new dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path):
    """Load config YAML with optional base inheritance.

    If the config contains a 'base' key, load that file first and merge.
    The 'base' path is resolved relative to the config file's directory.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "base" in cfg:
        base_path = cfg.pop("base")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        base_path = os.path.join(config_dir, base_path)
        base_cfg = load_config(base_path)
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def validate_config(cfg):
    """Check for contradictions and required fields."""
    arch = cfg.get("model", {}).get("arch", "unet")

    # pix2pix_turbo has its own model structure; skip U-Net-specific checks
    if arch == "pix2pix_turbo":
        errors = []
        required = [
            "data.data_dir",
            "training.epochs",
            "training.batch_size",
            "training.lr",
            "training.checkpoint_dir",
        ]
        for path in required:
            parts = path.split(".")
            obj = cfg
            for part in parts:
                if not isinstance(obj, dict) or part not in obj:
                    errors.append(f"Missing required config key: {path}")
                    break
                obj = obj[part]
        if errors:
            raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
        return cfg

    errors = []

    dims = cfg.get("model", {}).get("dims")
    if dims not in ("2d", "3d"):
        errors.append(f"model.dims must be '2d' or '3d', got '{dims}'")

    time_kernel = cfg.get("model", {}).get("time_kernel")
    if dims == "2d" and time_kernel is not None:
        errors.append("model.time_kernel must be null for 2d mode")

    if dims == "3d" and time_kernel is None:
        errors.append("model.time_kernel is required for 3d mode")

    required = [
        "data.data_dir",
        "model.encoder",
        "model.in_channels",
        "model.out_channels",
        "training.epochs",
        "training.batch_size",
        "training.lr",
        "training.checkpoint_dir",
    ]
    for path in required:
        parts = path.split(".")
        obj = cfg
        for part in parts:
            if not isinstance(obj, dict) or part not in obj:
                errors.append(f"Missing required config key: {path}")
                break
            obj = obj[part]

    if errors:
        raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))

    return cfg
