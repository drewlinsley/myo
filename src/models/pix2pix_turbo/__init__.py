import types
import torch

# Shim for older PyTorch that lacks torch.xpu (diffusers probes it on import)
if not hasattr(torch, "xpu"):
    _xpu = types.ModuleType("torch.xpu")
    _xpu.is_available = lambda: False
    torch.xpu = _xpu

from .pix2pix_turbo import Pix2Pix_Turbo

__all__ = ["Pix2Pix_Turbo"]
