#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8099}"
HOST="${HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif [[ -x "${ROOT_DIR}/../PUDU_app/backend/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/../PUDU_app/backend/.venv/bin/python"
  else
    python3 -m venv "${ROOT_DIR}/.venv"
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"
  fi
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}"
