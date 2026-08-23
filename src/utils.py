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


def recalibrate_bn(model, loader, n_batches=32, device=None, input_fn=None):
    """Re-estimate BatchNorm running statistics on a new input distribution.

    Why this exists
    ---------------
    load_encoder_from_unet copies BN running_mean/running_var along with the
    weights, and those are BRIGHTFIELD statistics — the BF->GFP U-Net encoder
    only ever saw BF. A force probe run with --input gfp feeds that encoder a
    different modality, and --freeze_encoder holds encoder.eval() every epoch,
    so the stale BF statistics can never adapt. This runs forward passes in
    train() mode under no_grad so ONLY the running estimates move; no weight is
    touched and no gradient is taken.

    momentum=None makes each BN use a cumulative moving average, so the result
    is the true mean/var over the batches seen rather than an EMA that depends
    on batch order.

    Args:
        model: the model (or submodule) whose BN layers should be recalibrated.
        loader: yields batches; input_fn extracts the tensor to feed.
        n_batches: how many batches to average over.
        device: device to move inputs to (default: the model's first param).
        input_fn: batch -> input tensor. Default takes batch[0] for a
            sequence, else the batch itself.

    Returns:
        The number of BATCHES actually used (0 = nothing was done and the
        existing statistics were left untouched — check this, do not assume
        success).
    """
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return 0

    # Multi-GPU is NOT supported and fails silently rather than loudly, so
    # refuse it. Each rank would see only its own shard of the prepared loader
    # and estimate BN statistics from 1/N of the data with no all-reduce; DDP's
    # broadcast_buffers=True would then overwrite every rank with rank 0's
    # buffers at the first training forward. The result is "rank 0's few
    # batches" masquerading as a global estimate — wrong, and invisible.
    if (torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1):
        raise RuntimeError(
            f"recalibrate_bn does not support distributed training "
            f"(world_size={torch.distributed.get_world_size()}). Each rank "
            f"would estimate BatchNorm statistics from its own shard with no "
            f"all-reduce, and broadcast_buffers would then replace them all "
            f"with rank 0's — a silently wrong estimate.\n"
            f"  Either run single-GPU (CUDA_VISIBLE_DEVICES=0), or pass "
            f"--recalibrate_bn 0 to disable it.\n"
            f"  To add support: wrap the update in "
            f"torch.distributed.all_reduce over each BN's running_mean/"
            f"running_var, weighted by each rank's sample count.")

    if device is None:
        device = next(model.parameters()).device
    if input_fn is None:
        def input_fn(b):
            return b[0] if isinstance(b, (list, tuple)) else b

    # Probe the loader BEFORE destroying the existing statistics: resetting and
    # then finding nothing to estimate from would leave every BN at
    # mean=0/var=1, which is worse than the stale statistics we are replacing.
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= n_batches:
            break
    if not batches:
        return 0

    was_training = model.training
    saved = []
    for m in bns:
        saved.append(m.momentum)
        m.momentum = None          # cumulative average
        m.reset_running_stats()
        m.train()

    seen = 0
    with torch.no_grad():
        for batch in batches:
            model(input_fn(batch).to(device))
            seen += 1

    for m, mom in zip(bns, saved):
        m.momentum = mom
    # NOTE: model.train(was_training) below decides the final mode for every
    # submodule, BN included. Callers that want the freshly estimated running
    # stats used at inference must call model.eval() themselves.
    model.train(was_training)
    return seen


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
