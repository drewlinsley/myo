"""Per-volume adaptive Z-band selection ("z_range: auto").

Absolute z_range indices (e.g. [70, 105]) assume every stack puts the tissue at
the same depth. The 051826 drop has stacks from Z=70 to Z=183, so fixed indices
silently drop shallow stacks and pad others. Instead, compute_stats.py finds
each volume's best contiguous window (max mean GFP signal, falling back to BF
per-slice std) and stores it as "z_auto": [lo, hi] in the volume's stats JSON;
datasets then resolve `z_range: auto` per volume from those stats.

Torch-free (numpy + stdlib) so audit/compute_stats can use it anywhere.
"""

import numpy as np


def compute_z_auto(profile, z_window=35):
    """Best contiguous window [lo, hi) of length min(z_window, Z) over a per-Z
    signal profile (higher = more tissue). Returns [lo, hi] ints."""
    profile = np.asarray(profile, dtype=np.float64)
    z = len(profile)
    w = min(int(z_window), z)
    if w <= 0:
        return [0, z]
    c = np.concatenate([[0.0], np.cumsum(profile)])
    lo = int(np.argmax(c[w:] - c[:-w]))
    return [lo, lo + w]


def resolve_z_range(z_range, stats=None, n_z=None):
    """Resolve a config z_range into concrete (z_lo, z_hi) slice bounds.

    z_range may be:
      None        -> full stack (0, n_z)
      [lo, hi]    -> fixed absolute indices (legacy behavior)
      "auto"      -> this volume's stats["z_auto"] band (per-volume adaptive)

    Bounds are clamped to [0, n_z] when n_z is given. Raises if "auto" is
    requested but the stats JSON has no z_auto (rerun compute_stats.py — it
    fills z_auto into existing stats files without redoing percentiles).
    """
    if z_range is None:
        return 0, n_z
    if isinstance(z_range, str):
        if z_range != "auto":
            raise ValueError(
                f"z_range must be null, [lo, hi], or 'auto' (got {z_range!r})")
        band = (stats or {}).get("z_auto")
        if band is None:
            raise KeyError(
                "z_range: auto, but this volume's stats JSON has no 'z_auto' "
                "band — rerun compute_stats.py on this data_dir (it upgrades "
                "existing stats in place).")
        z_lo, z_hi = int(band[0]), int(band[1])
    else:
        z_lo, z_hi = int(z_range[0]), int(z_range[1])
    z_lo = max(0, z_lo)
    if n_z is not None:
        z_hi = min(int(n_z), z_hi)
    return z_lo, z_hi
