#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${GP_ROOT}/scripts/grasp_runtime_helpers.sh"
cd "${GP_ROOT}"

export PYTHONPATH="${INTERNVL_CHAT_ROOT}:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-63679}"

GPUS="${GPUS:-${GRASP_EVAL_GPUS}}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-$("${PYTHON_BIN}" "${GP_ROOT}/grasp_settings.py" default-name crop)}"
  WORK_DIR="${WORK_DIR:-${GRASP_CROP_RUN_ROOT}/${EXPERIMENT_NAME}}"
  CHECKPOINT="$(latest_checkpoint "${WORK_DIR}")"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/checkpoint-* or WORK_DIR=/path/to/work_dir." >&2
  exit 1
fi
eval "$("${PYTHON_BIN}" "${GP_ROOT}/scripts/grasp_experiment_metadata.py" env --checkpoint "${CHECKPOINT}")"
if [[ "${VCOT_PIPELINE}" != "oracle_crop" ]]; then
  echo "Checkpoint pipeline is ${VCOT_PIPELINE}, expected oracle_crop: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ -z "${VCOT_DATA_INDEX_ROOT}" ]]; then
  echo "Checkpoint config is missing data_index_root: ${VCOT_CONFIG_PATH}" >&2
  exit 1
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${VCOT_EXPERIMENT_NAME}}"
DATASETS="${DATASETS:-test_seen,test_unseen}"
OUT_DIR="${OUT_DIR:-${GRASP_CROP_RESULT_ROOT}/${EXPERIMENT_NAME}}"
DATA_INDEX_ROOT="${DATA_INDEX_ROOT:-${VCOT_DATA_INDEX_ROOT}}"
VCOT_IOU_THRESHOLD="${VCOT_IOU_THRESHOLD:-0.25}"
VCOT_ANGLE_THRESHOLD="${VCOT_ANGLE_THRESHOLD:-30.0}"
DEFAULT_DATASET_SPLITS="$(default_dataset_splits "${DATASETS}")"
if [[ -n "${DEFAULT_DATASET_SPLITS}" ]]; then
  require_data_index_splits "${DATA_INDEX_ROOT}" "${DATASETS}"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  eval/evaluate_crop_grasp.py \
  --checkpoint "${CHECKPOINT}" \
  --datasets "${DATASETS}" \
  --out-dir "${OUT_DIR}" \
  --vcot-iou-threshold "${VCOT_IOU_THRESHOLD}" \
  --vcot-angle-threshold "${VCOT_ANGLE_THRESHOLD}" \
  "$@"
