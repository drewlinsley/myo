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


def tune_cudnn(enable=True):
    """Enable cuDNN autotuning for fixed-shape training (every batch here is the
    same (B, C, crop, crop[, D]) shape, the best case for benchmark mode). This
    trades bitwise run-to-run reproducibility (set_seed's deterministic flag)
    for a large 3D-conv speedup; pass enable=False (cfg training.cudnn_benchmark:
    false) to keep strict determinism instead."""
    if enable and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


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


def save_checkpoint(model, optimizer, epoch, val_loss, cfg, path,
                    accelerator=None, scheduler=None):
    """Save a training checkpoint.

    scheduler (optional): stored so --resume continues the LR schedule where it
    left off instead of restarting warmup + cosine from scratch.
    """
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
    if scheduler is not None:
        try:
            state["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
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

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


# ---------------------------------------------------------------------------
# Resumable training state
#
# save_checkpoint() stores what INFERENCE needs (weights + a little metadata).
# A crashed run needs more to continue honestly: optimizer moments, the epoch
# counter, the early-stopping bookkeeping, the LR scheduler position, and the
# RNG streams. These helpers write that as ONE atomic file per run, so killing
# the job at any moment leaves either the old state or the new one — never a
# half-written file.
# ---------------------------------------------------------------------------
def save_train_state(path, model, optimizer, epoch, fingerprint,
                     extra=None, scheduler=None, save_rng=True):
    """Atomically write a resumable training state.

    epoch: the number of epochs COMPLETED (training resumes at this index).
    fingerprint: dict of run-defining settings; load_train_state refuses to
        resume a state whose fingerprint disagrees, so a stale file from a
        different experiment can't silently continue into this one.
    extra: any JSON/pickle-able run bookkeeping (best metric, history, ...).
    """
    state = {
        "schema": "train_state_v1",
        "epoch": int(epoch),
        "fingerprint": dict(fingerprint),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra or {},
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if save_rng:
        state["rng"] = {
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_train_state(path, model, optimizer=None, scheduler=None,
                     fingerprint=None, restore_rng=True):
    """Restore a state written by save_train_state. Returns the full dict
    (['epoch'] = epochs already completed, ['extra'] = run bookkeeping)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    if fingerprint is not None:
        saved = st.get("fingerprint", {})
        diff = {k: (saved.get(k), v) for k, v in fingerprint.items()
                if saved.get(k) != v}
        if diff:
            lines = "\n".join(f"    {k}: saved={s!r} now={n!r}"
                              for k, (s, n) in diff.items())
            raise SystemExit(
                f"Refusing to resume {path}: it was written by a run with "
                f"different settings —\n{lines}\n"
                f"  Delete the state file to start fresh, or fix the flags.")
    model.load_state_dict(st["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in st:
        optimizer.load_state_dict(st["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in st:
        scheduler.load_state_dict(st["scheduler_state_dict"])
    if restore_rng and st.get("rng"):
        rng = st["rng"]
        torch.set_rng_state(rng["torch"].cpu()
                            if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            except Exception:
                pass          # different GPU count than the original run
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return st


def resolve_resume(flag, default_path):
    """Map a --resume value to a path to load, or None.

    flag is None (no resume), 'auto' (use default_path if it exists), or an
    explicit path (must exist). Returns the path to load, or None.
    """
    if flag is None:
        return None
    path = default_path if flag == "auto" else flag
    if os.path.exists(path):
        return path
    if flag != "auto":
        raise SystemExit(f"--resume {flag}: no such file")
    return None


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
