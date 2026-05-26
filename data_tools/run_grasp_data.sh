#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${GP_ROOT}/grasp_paths.sh"
cd "${GP_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${GRASP_PYTHON_BIN}}"
SOURCE_ROOT="${SOURCE_ROOT:-${GRASP_DATASET_ROOT}}"
SPLITS="${SPLITS:-train test_seen test_unseen}"
EVAL_SPLITS="${EVAL_SPLITS:-test_seen test_unseen}"

echo "===== data stage: grasp data_index/meta ====="
echo "source_root=${SOURCE_ROOT}"
echo "data_index_root=${GRASP_DATA_INDEX_ROOT}"
echo "splits=${SPLITS}"
echo "eval_splits=${EVAL_SPLITS}"

# shellcheck disable=SC2086
"${PYTHON_BIN}" data_tools/ensure_grasp_data.py all \
  --source-root "${SOURCE_ROOT}" \
  --splits ${SPLITS} \
  --eval-splits ${EVAL_SPLITS}

echo "===== data check: generated data_index/meta ====="
# shellcheck disable=SC2086
"${PYTHON_BIN}" data_tools/check_grasp_data.py all \
  --splits ${SPLITS} \
  --eval-splits ${EVAL_SPLITS}

echo "===== data stage complete ====="
