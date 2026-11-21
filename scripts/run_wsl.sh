#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
experiment_env="${CARRYRANK_ENV:-$HOME/.venvs/carryrank}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
exec "$experiment_env/bin/python" -m carryrank.experiment "$@"
