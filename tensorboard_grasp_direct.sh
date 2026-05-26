#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${GP_ROOT}/InternVL/internvl_chat"

DEFAULT_LOGDIR_SPEC="direct:work_dirs/internvl_chat_v2_5/grasp_direct_hparams,crop:work_dirs/internvl_chat_v2_5/grasp_crop_hparams,vcot:work_dirs/internvl_chat_v2_5/grasp_vcot_hparams"
LOGDIR="${LOGDIR:-}"
LOGDIR_SPEC="${LOGDIR_SPEC:-${DEFAULT_LOGDIR_SPEC}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

LOGDIR_ARGS=()
if [[ -n "${LOGDIR}" ]]; then
  LOGDIR_ARGS=(--logdir "${LOGDIR}")
else
  LOGDIR_ARGS=(--logdir_spec "${LOGDIR_SPEC}")
fi

if command -v tensorboard >/dev/null 2>&1; then
  exec tensorboard \
    "${LOGDIR_ARGS[@]}" \
    --host "${HOST}" \
    --port "${PORT}"
fi

if python -c "import tensorboard" >/dev/null 2>&1; then
  exec python -m tensorboard.main \
    "${LOGDIR_ARGS[@]}" \
    --host "${HOST}" \
    --port "${PORT}"
fi

cat >&2 <<EOF
TensorBoard is not installed in the current Python environment.

Install it first:
  pip install tensorboard

Then run again:
  bash tensorboard_grasp_direct.sh

By default this opens direct, crop, and vcot logs together.
Override LOGDIR to inspect a specific directory, for example:
  LOGDIR=work_dirs/internvl_chat_v2_5/grasp_direct_hparams bash tensorboard_grasp_direct.sh

Override LOGDIR_SPEC to customize named groups, for example:
  LOGDIR_SPEC=direct:path/to/direct,crop:path/to/crop bash tensorboard_grasp_direct.sh
EOF
exit 1
