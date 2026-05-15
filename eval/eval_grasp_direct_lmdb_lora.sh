#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${GP_ROOT}/scripts/grasp_run_common.sh"
cd "${GP_ROOT}"

export PYTHONPATH="${GP_ROOT}/InternVL/internvl_chat:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-63669}"

GPUS="${GPUS:-1}"
USE_LLM_LORA="${USE_LLM_LORA:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-6}"
DEFAULT_EXPERIMENT_NAME="lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch${MAX_DYNAMIC_PATCH}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
if [[ -z "${WORK_DIR:-}" ]]; then
  WORK_DIR="${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_direct_hparams/${EXPERIMENT_NAME}"
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/checkpoint-* or WORK_DIR=/path/to/work_dir." >&2
  exit 1
fi
DATASETS="${DATASETS:-test_seen,test_unseen}"
OUT_DIR="${OUT_DIR:-${GP_ROOT}/result/vcot_grasp_direct}"
VCOT_IOU_THRESHOLD="${VCOT_IOU_THRESHOLD:-0.25}"
VCOT_ANGLE_THRESHOLD="${VCOT_ANGLE_THRESHOLD:-30.0}"
DEFAULT_DATASET_SPLITS="$(default_dataset_splits "${DATASETS}")"
if [[ -n "${DEFAULT_DATASET_SPLITS}" ]]; then
  python "${GP_ROOT}/scripts/ensure_grasp_data.py" direct \
    --meta-path "${GP_ROOT}/data/vcot_grasp/direct/internvl_meta_train.json" \
    --splits train ${DEFAULT_DATASET_SPLITS}
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  eval/evaluate_direct_grasp.py \
  --checkpoint "${CHECKPOINT}" \
  --datasets "${DATASETS}" \
  --manifest-root "${GP_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --vcot-iou-threshold "${VCOT_IOU_THRESHOLD}" \
  --vcot-angle-threshold "${VCOT_ANGLE_THRESHOLD}" \
  "$@"
