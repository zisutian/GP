#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${GP_ROOT}"

export PYTHONPATH="${GP_ROOT}/InternVL/internvl_chat:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-63679}"

GPUS="${GPUS:-1}"
WORK_DIR="${WORK_DIR:-${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_crop_hparams/crop_object_lora16_lr8e-5_ep1_patch6_edge15_half50}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/checkpoint-* or WORK_DIR=/path/to/work_dir." >&2
  exit 1
fi
DATASETS="${DATASETS:-test_seen,test_unseen}"
OUT_DIR="${OUT_DIR:-${GP_ROOT}/result/vcot_grasp_crop}"

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
  "$@"
