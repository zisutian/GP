#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GP_ROOT}/scripts/grasp_run_common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE_EVAL_RESULTS="${OVERWRITE_EVAL_RESULTS:-False}"
EVAL_DATASETS="${EVAL_DATASETS:-test_seen,test_unseen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

USE_LLM_LORA="${USE_LLM_LORA:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-448}"
TARGET_GRASP_INDEX="${TARGET_GRASP_INDEX:-0}"
TRAIN_SCRIPT="${GP_ROOT}/InternVL/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_grasp_crop_lmdb_lora.sh"
TARGET_GRASP_TAG=""
if [[ "${TARGET_GRASP_INDEX}" != "0" ]]; then
  TARGET_GRASP_TAG="_g${TARGET_GRASP_INDEX}"
fi

export CUDA_VISIBLE_DEVICES

prepare_meta() {
  local name="$1"
  local edge_expand="$2"
  local min_half="$3"
  local target_frame="$4"
  local meta_root="${GP_ROOT}/data/vcot_grasp/crop_hparams/${name}"
  local meta_path="${meta_root}/internvl_meta_train.json"

  "${PYTHON_BIN}" "${GP_ROOT}/scripts/ensure_grasp_data.py" crop \
    --meta-path "${meta_path}" \
    --output-root "${meta_root}" \
    --splits train test_seen test_unseen \
    --bbox-edge-expand "${edge_expand}" \
    --min-bbox-half-size "${min_half}" \
    --target-coordinate-frame "${target_frame}" \
    --target-grasp-index "${TARGET_GRASP_INDEX}" >&2
  echo "${meta_path}"
}

write_config() {
  local name="$1"
  local meta_path="$2"
  local work_dir="$3"
  local max_dynamic_patch="$4"

  "${PYTHON_BIN}" "${GP_ROOT}/scripts/grasp_config.py" write \
    --pipeline oracle_crop \
    --experiment-name "${name}" \
    --meta-path "${meta_path}" \
    --output-dir "${work_dir}" \
    --use-llm-lora "${USE_LLM_LORA}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
    --max-dynamic-patch "${max_dynamic_patch}" \
    --force-image-size "${FORCE_IMAGE_SIZE}" \
    --copy-to-checkpoints
}

run_experiment() {
  local name="$1"
  local edge_expand="$2"
  local min_half="$3"
  local target_frame="$4"
  local max_dynamic_patch="$5"
  local work_dir="${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_crop_hparams/${name}"
  local out_dir="${GP_ROOT}/result/vcot_grasp_crop/hparams/${name}"
  local overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"
  local meta_path
  local eval_datasets
  local eval_overwrite
  local action

  echo "===== ${name} ====="
  meta_path="$(prepare_meta "${name}" "${edge_expand}" "${min_half}" "${target_frame}")"

  action="$(training_action "${work_dir}" "${overwrite_output_dir}")"
  if [[ "${action}" == "skip" ]]; then
    echo "Skip training: existing checkpoint found in ${work_dir}"
  else
    if [[ "${action}" == "overwrite" ]]; then
      echo "Incomplete output directory found; rerunning with overwrite enabled: ${work_dir}"
      overwrite_output_dir=True
    fi
    EXPERIMENT_NAME="${name}" \
    META_PATH="${meta_path}" \
    USE_LLM_LORA="${USE_LLM_LORA}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    MAX_DYNAMIC_PATCH="${max_dynamic_patch}" \
    FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE}" \
    BBOX_EDGE_EXPAND="${edge_expand}" \
    MIN_BBOX_HALF_SIZE="${min_half}" \
    TARGET_COORDINATE_FRAME="${target_frame}" \
    TARGET_GRASP_INDEX="${TARGET_GRASP_INDEX}" \
    TRAINING_LOG_PATH="work_dirs/internvl_chat_v2_5/grasp_crop_hparams/${name}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${TRAIN_SCRIPT}"
  fi

  write_config "${name}" "${meta_path}" "${work_dir}" "${max_dynamic_patch}"

  if should_run_eval; then
    if skip_eval_without_checkpoint "${work_dir}"; then
      return
    fi
    prepare_eval_dir "${out_dir}"
    eval_overwrite="${OVERWRITE_EVAL_RESULTS}"
    if [[ "${action}" != "skip" ]]; then
      eval_overwrite=True
    fi
    eval_datasets="$(missing_eval_datasets "${out_dir}" "${EVAL_DATASETS}" "${eval_overwrite}")"
    if [[ -z "${eval_datasets}" ]]; then
      echo "Skip eval: existing result JSON found for all requested datasets in ${out_dir}"
      return
    fi
    if [[ "${eval_datasets}" != "${EVAL_DATASETS}" ]]; then
      echo "Partial eval: missing datasets ${eval_datasets}; existing results kept in ${out_dir}"
    fi
    WORK_DIR="${work_dir}" \
    OUT_DIR="${out_dir}" \
    DATASET_ROOT="${GP_ROOT}/data/vcot_grasp/crop_hparams/${name}" \
    GPUS=1 \
    DATASETS="${eval_datasets}" \
    bash "${GP_ROOT}/eval/eval_grasp_crop_lmdb_lora.sh" \
      2>&1 | tee -a "${out_dir}/${name}.eval.log"
  fi
}

run_experiment "crop_object_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch6_edge15_half50${TARGET_GRASP_TAG}" 15 50 full_image 6
run_experiment "crop_full_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch6_edge5_half40${TARGET_GRASP_TAG}" 5 40 full_image 6
run_experiment "crop_full_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch6_edge10_half40${TARGET_GRASP_TAG}" 10 40 full_image 6
run_experiment "crop_frame_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch6_edge5_half40${TARGET_GRASP_TAG}" 5 40 crop_image 6
run_experiment "crop_frame_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch6_edge10_half40${TARGET_GRASP_TAG}" 10 40 crop_image 6
run_experiment "crop_frame_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch8_edge10_half40${TARGET_GRASP_TAG}" 10 40 crop_image 8
