import types
import torch

# Shim for older PyTorch that lacks full torch.xpu (diffusers probes it on import)
if not hasattr(torch, "xpu") or not hasattr(torch.xpu, "empty_cache"):
    _xpu = types.ModuleType("torch.xpu")
    _xpu.is_available = lambda: False
    _xpu.empty_cache = lambda: None
    _xpu.device_count = lambda: 0
    _xpu.current_device = lambda: 0
    _xpu.mem_get_info = lambda *a, **kw: (0, 0)
    torch.xpu = _xpu

from .pix2pix_turbo import Pix2Pix_Turbo

__all__ = ["Pix2Pix_Turbo"]
