#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${GP_ROOT}"

export PYTHONPATH="${GP_ROOT}/InternVL/internvl_chat:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-baseline_lora16_lr4e-5_ep1_patch6}"
CHECKPOINT="${CHECKPOINT:-artifacts/runs/direct/${EXPERIMENT}/checkpoint-11627}"
DATASETS="${DATASETS:-test_seen,test_unseen}"
OUT_DIR="${OUT_DIR:-artifacts/results/direct/${EXPERIMENT}}"
VCOT_IOU_THRESHOLD="${VCOT_IOU_THRESHOLD:-0.25}"
VCOT_ANGLE_THRESHOLD="${VCOT_ANGLE_THRESHOLD:-30.0}"

"${PYTHON_BIN:-python}" eval/evaluate_direct_grasp.py \
  --checkpoint "${CHECKPOINT}" \
  --datasets "${DATASETS}" \
  --out-dir "${OUT_DIR}" \
  --vcot-iou-threshold "${VCOT_IOU_THRESHOLD}" \
  --vcot-angle-threshold "${VCOT_ANGLE_THRESHOLD}" \
  "$@"
