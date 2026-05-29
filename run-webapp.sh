#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8099}"
HOST="${HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    echo "Creating virtual environment in ${ROOT_DIR}/.venv"
    python3 -m venv "${ROOT_DIR}/.venv"
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
    echo "Installing Python dependencies"
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"
  fi
fi

CUDA_LIBRARY_DIRS="$("${PYTHON_BIN}" - <<'PY'
import site
from pathlib import Path

roots = [Path(item) for item in site.getsitepackages()]
user_site = site.getusersitepackages()
if user_site:
    roots.append(Path(user_site))

relative_dirs = (
    "nvidia/cublas/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cudnn/lib",
)

paths = []
for root in roots:
    for relative in relative_dirs:
        candidate = root / relative
        if candidate.exists():
            paths.append(str(candidate))

print(":".join(dict.fromkeys(paths)))
PY
)"
if [[ -n "${CUDA_LIBRARY_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIBRARY_DIRS}:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
echo "Starting Transcript Direct on http://${HOST}:${PORT}"
exec "${PYTHON_BIN}" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}"
