#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GP_ROOT}/grasp_paths.sh"
cd "${INTERNVL_CHAT_ROOT}"

DEFAULT_LOGDIR_SPEC="direct:${GRASP_DIRECT_RUN_ROOT},crop:${GRASP_CROP_RUN_ROOT},vcot:${GRASP_VCOT_RUN_ROOT}"
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
  LOGDIR=${GRASP_DIRECT_RUN_ROOT} bash tensorboard_grasp_direct.sh

Override LOGDIR_SPEC to customize named groups, for example:
  LOGDIR_SPEC=direct:path/to/direct,crop:path/to/crop bash tensorboard_grasp_direct.sh
EOF
exit 1
