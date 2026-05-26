#!/usr/bin/env bash
set -euo pipefail

GP_ROOT=${GP_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"}
cd "${GP_ROOT}/InternVL/internvl_chat"

export PYTHONPATH="${GP_ROOT}:$(pwd):${PYTHONPATH:-}"
export LAUNCHER="${LAUNCHER:-pytorch}"

"${PYTHON_BIN:-python}" internvl/train/internvl_chat_finetune.py "$@"
