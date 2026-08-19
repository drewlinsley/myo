#!/usr/bin/env python
"""Why isn't training on the GPU? — one-shot triage.

Every trainer here selects the device the same way (torch.cuda.is_available(),
or Accelerate which does the same underneath), so "it's running on CPU" is
always one of four things. This tells you which:

  1. CPU-only torch wheel       -> reinstall torch with a CUDA build
  2. driver / CUDA mismatch     -> nvidia-smi works but torch can't init
  3. GPUs masked off            -> CUDA_VISIBLE_DEVICES set to "" or a bad id
  4. Accelerate pinned to CPU   -> use_cpu: true in its default config

If all four are clean and the GPU still looks idle in nvidia-smi, the job IS on
the GPU and you're data-bound — run scripts/profile_pipeline.py instead.

Usage:  python scripts/check_gpu.py
"""

import os
import shutil
import subprocess
import sys

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def line(tag, msg):
    print(f"[{tag}] {msg}")


def main():
    problems = []

    print("=" * 66)
    print(" GPU triage")
    print("=" * 66)

    # ── interpreter ───────────────────────────────────────────────────
    line("info", f"python     {sys.version.split()[0]}  ({sys.executable})")
    line("info", f"conda env  {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")

    # ── 3. masking (check before torch, it changes what torch sees) ────
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        line(OK, "CUDA_VISIBLE_DEVICES unset (all GPUs visible)")
    elif cvd.strip() == "":
        line(BAD, "CUDA_VISIBLE_DEVICES is set to EMPTY -> every GPU is hidden")
        problems.append("unset CUDA_VISIBLE_DEVICES  (or set it to 0)")
    else:
        line(OK, f"CUDA_VISIBLE_DEVICES={cvd}")

    # ── driver ────────────────────────────────────────────────────────
    smi = shutil.which("nvidia-smi")
    driver_gpus = 0
    if not smi:
        line(BAD, "nvidia-smi not found -> no NVIDIA driver on this host")
        problems.append("this VM has no GPU/driver — check the instance type")
    else:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                stderr=subprocess.STDOUT, timeout=30).decode().strip()
            for row in out.splitlines():
                driver_gpus += 1
                line(OK, f"driver sees: {row}")
        except Exception as e:
            line(BAD, f"nvidia-smi failed: {e}")
            problems.append("driver is installed but broken — reboot or reinstall it")

    # ── 1/2. torch ────────────────────────────────────────────────────
    try:
        import torch
    except Exception as e:
        line(BAD, f"import torch failed: {e}")
        print("\nFix that first, then re-run.")
        return 1

    line("info", f"torch      {torch.__version__}")
    built = torch.version.cuda
    if built is None or "+cpu" in torch.__version__:
        line(BAD, "this is a CPU-ONLY torch build (no CUDA compiled in)")
        problems.append(
            "reinstall a CUDA wheel, e.g.\n"
            "        pip uninstall -y torch torchvision\n"
            "        pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124")
    else:
        line(OK, f"torch built against CUDA {built}")

    avail = torch.cuda.is_available()
    if avail:
        n = torch.cuda.device_count()
        line(OK, f"torch.cuda.is_available() = True  ({n} device(s))")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            line("info", f"  cuda:{i}  {p.name}  "
                         f"{p.total_memory / 1024**3:.1f} GiB  sm_{p.major}{p.minor}")
    else:
        line(BAD, "torch.cuda.is_available() = False  <- this is why it's on CPU")
        if driver_gpus and built:
            problems.append(
                "driver sees GPUs and torch has CUDA, but init failed — usually a "
                "driver/runtime version mismatch. Check `nvidia-smi` CUDA version "
                f"vs torch's {built}, and try:\n"
                "        python -c \"import torch; torch.zeros(1).cuda()\"\n"
                "      for the underlying error message.")

    # ── 4. accelerate default config ──────────────────────────────────
    try:
        from accelerate.utils import default_config_file  # noqa
        cfg_path = default_config_file
    except Exception:
        cfg_path = os.path.expanduser(
            "~/.cache/huggingface/accelerate/default_config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            acc = yaml.safe_load(open(cfg_path)) or {}
        except Exception:
            acc = {}
        if acc.get("use_cpu") or str(acc.get("distributed_type", "")).upper() == "CPU":
            line(BAD, f"accelerate config pins CPU: {cfg_path}")
            problems.append(
                f"the classifier trainers go through Accelerate — edit {cfg_path} "
                "(use_cpu: false) or just delete it to fall back to auto-detect")
        else:
            line(OK, f"accelerate config looks fine: {cfg_path}")
    else:
        line(OK, "no accelerate config (auto-detect — correct)")

    # ── proof: does a real op land on the GPU? ─────────────────────────
    if avail:
        try:
            a = torch.randn(2048, 2048, device="cuda")
            (a @ a).sum().item()
            torch.cuda.synchronize()
            used = torch.cuda.memory_allocated() / 1024**2
            line(OK, f"matmul ran on cuda:0 ({used:.0f} MiB allocated) — GPUs work")
        except Exception as e:
            line(BAD, f"GPU op failed: {e}")
            problems.append(f"CUDA reports available but ops fail: {e}")

    # ── verdict ───────────────────────────────────────────────────────
    print("=" * 66)
    if not problems:
        print(" No GPU problem found — torch can use the GPU from this env.")
        print("")
        print(" If nvidia-smi still shows low utilization during training, the")
        print(" job IS on the GPU and is starved by the data loader. Run:")
        print("     python scripts/profile_pipeline.py -c configs/gfp_classifier_3d.yaml")
        print("")
        print(" Also make sure you launch with THIS interpreter:")
        print(f"     {sys.executable}")
    else:
        print(f" {len(problems)} problem(s) to fix:")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
    print("=" * 66)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
