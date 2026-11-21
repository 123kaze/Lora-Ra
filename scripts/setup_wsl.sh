#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
experiment_env="${CARRYRANK_ENV:-$HOME/.venvs/carryrank}"
python3 -m venv "$experiment_env"
"$experiment_env/bin/python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$experiment_env/bin/python" -m pip install -r requirements.txt
"$experiment_env/bin/python" -m pip install --no-deps --no-build-isolation -e .
"$experiment_env/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
