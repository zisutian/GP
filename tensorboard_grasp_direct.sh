#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${GP_ROOT}/InternVL/internvl_chat"

LOGDIR="${LOGDIR:-work_dirs/internvl_chat_v2_5/internvl2_5_1b_grasp_direct_lmdb_lora/runs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

if command -v tensorboard >/dev/null 2>&1; then
  exec tensorboard \
    --logdir "${LOGDIR}" \
    --host "${HOST}" \
    --port "${PORT}"
fi

if python -c "import tensorboard" >/dev/null 2>&1; then
  exec python -m tensorboard.main \
    --logdir "${LOGDIR}" \
    --host "${HOST}" \
    --port "${PORT}"
fi

cat >&2 <<EOF
TensorBoard is not installed in the current Python environment.

Install it first:
  pip install tensorboard

Then run again:
  bash tensorboard_grasp_direct.sh
EOF
exit 1
