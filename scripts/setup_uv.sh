#!/bin/bash
# Create a uv-managed virtualenv for this repo and install all deps.
#
# uv is a fast drop-in for pip/venv. This makes a .venv/ in the repo root,
# pins a Python, and installs requirements.txt into it.
#
# Usage
# -----
#   bash scripts/setup_uv.sh                    # CPU/default torch wheel
#   TORCH_CUDA=cu121 bash scripts/setup_uv.sh   # GPU VM: CUDA 12.1 torch build
#   PYTHON_VERSION=3.11 bash scripts/setup_uv.sh
#
# Then, before running anything:
#   source .venv/bin/activate
#   bash scripts/force_all.sh
#
# Env knobs (defaults in parens):
#   PYTHON_VERSION (3.11)   Python for the venv (uv fetches it if missing)
#   VENV_DIR       (.venv)  where the env lives
#   TORCH_CUDA     (unset)  e.g. cu121 / cu124 -> install torch from that CUDA
#                           index FIRST so the GPU wheel wins over the PyPI one.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

# 1. uv present?
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with one of:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "    pip install uv"
  exit 1
fi
echo "uv: $(uv --version)"

# 2. create the venv (uv downloads the interpreter if needed)
echo "▶ creating $VENV_DIR (python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# 3. torch: on a GPU box install the CUDA build first so it isn't overwritten
#    by the default PyPI wheel that requirements.txt would pull.
if [ -n "${TORCH_CUDA:-}" ]; then
  echo "▶ installing CUDA torch ($TORCH_CUDA)"
  uv pip install --python "$VENV_DIR" torch torchvision \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

# 4. everything else
echo "▶ installing requirements.txt"
uv pip install --python "$VENV_DIR" -r requirements.txt

echo ""
echo "✅ done. activate with:  source $VENV_DIR/bin/activate"
echo "   verify torch/CUDA:    $VENV_DIR/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
