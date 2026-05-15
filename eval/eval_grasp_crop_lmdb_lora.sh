#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${GP_ROOT}/scripts/grasp_run_common.sh"
cd "${GP_ROOT}"

export PYTHONPATH="${GP_ROOT}/InternVL/internvl_chat:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-63679}"

GPUS="${GPUS:-1}"
USE_LLM_LORA="${USE_LLM_LORA:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-6}"
EXPERIMENT_BBOX_EDGE_EXPAND="${BBOX_EDGE_EXPAND:-15}"
EXPERIMENT_MIN_BBOX_HALF_SIZE="${MIN_BBOX_HALF_SIZE:-50}"
EXPERIMENT_TARGET_COORDINATE_FRAME="${TARGET_COORDINATE_FRAME:-full_image}"
if [[ "${EXPERIMENT_TARGET_COORDINATE_FRAME}" == "crop_image" ]]; then
  TARGET_FRAME_TAG="frame"
elif [[ "${EXPERIMENT_TARGET_COORDINATE_FRAME}" == "full_image" && "${EXPERIMENT_BBOX_EDGE_EXPAND}" == "15" && "${EXPERIMENT_MIN_BBOX_HALF_SIZE}" == "50" ]]; then
  TARGET_FRAME_TAG="object"
elif [[ "${EXPERIMENT_TARGET_COORDINATE_FRAME}" == "full_image" ]]; then
  TARGET_FRAME_TAG="full"
else
  echo "Unsupported TARGET_COORDINATE_FRAME=${EXPERIMENT_TARGET_COORDINATE_FRAME}; expected full_image or crop_image." >&2
  exit 1
fi
DEFAULT_EXPERIMENT_NAME="crop_${TARGET_FRAME_TAG}_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch${MAX_DYNAMIC_PATCH}_edge${EXPERIMENT_BBOX_EDGE_EXPAND}_half${EXPERIMENT_MIN_BBOX_HALF_SIZE}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
WORK_DIR="${WORK_DIR:-${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_crop_hparams/${EXPERIMENT_NAME}}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(latest_checkpoint "${WORK_DIR}")"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/checkpoint-* or WORK_DIR=/path/to/work_dir." >&2
  exit 1
fi
DATASETS="${DATASETS:-test_seen,test_unseen}"
OUT_DIR="${OUT_DIR:-${GP_ROOT}/result/vcot_grasp_crop}"
DATASET_ROOT="${DATASET_ROOT:-}"
VCOT_IOU_THRESHOLD="${VCOT_IOU_THRESHOLD:-0.25}"
VCOT_ANGLE_THRESHOLD="${VCOT_ANGLE_THRESHOLD:-30.0}"
DEFAULT_DATASET_SPLITS="$(default_dataset_splits "${DATASETS}")"
if [[ -n "${DEFAULT_DATASET_SPLITS}" && -z "${DATASET_ROOT}" ]]; then
  python "${GP_ROOT}/scripts/ensure_grasp_data.py" crop \
    --meta-path "${GP_ROOT}/data/vcot_grasp/crop/internvl_meta_train.json" \
    --splits train ${DEFAULT_DATASET_SPLITS} \
    --bbox-edge-expand 15 \
    --min-bbox-half-size 50 \
    --target-coordinate-frame full_image \
    --target-grasp-index 0
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  require_dataset_root_splits "${DATASET_ROOT}" "${DATASETS}"
fi
EXTRA_EVAL_ARGS=()
if [[ -n "${BBOX_EDGE_EXPAND:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--bbox-edge-expand "${BBOX_EDGE_EXPAND}")
fi
if [[ -n "${MIN_BBOX_HALF_SIZE:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--min-bbox-half-size "${MIN_BBOX_HALF_SIZE}")
fi
if [[ -n "${TARGET_COORDINATE_FRAME:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--target-coordinate-frame "${TARGET_COORDINATE_FRAME}")
fi
if [[ -n "${TARGET_GRASP_INDEX:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--target-grasp-index "${TARGET_GRASP_INDEX}")
fi
if [[ -n "${VCOT_CONFIG:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--vcot-config "${VCOT_CONFIG}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  EXTRA_EVAL_ARGS+=(--dataset-root "${DATASET_ROOT}")
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
  --manifest-root "${GP_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --vcot-iou-threshold "${VCOT_IOU_THRESHOLD}" \
  --vcot-angle-threshold "${VCOT_ANGLE_THRESHOLD}" \
  "${EXTRA_EVAL_ARGS[@]}" \
  "$@"
