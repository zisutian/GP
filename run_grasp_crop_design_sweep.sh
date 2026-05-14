#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_DATASETS="${EVAL_DATASETS:-test_seen,test_unseen}"
PYTHON_BIN="${PYTHON_BIN:-python}"

USE_LLM_LORA="${USE_LLM_LORA:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

export CUDA_VISIBLE_DEVICES

prepare_meta() {
  local name="$1"
  local edge_expand="$2"
  local min_half="$3"
  local target_frame="$4"
  local meta_root="${GP_ROOT}/data/vcot_grasp/crop_hparams/${name}"
  local meta_path="${meta_root}/internvl_meta_train.json"

  if [[ -f "${meta_path}" ]]; then
    echo "${meta_path}"
    return
  fi

  "${PYTHON_BIN}" "${GP_ROOT}/data_tools/prepare_grasp_anything_crop.py" \
    --output-root "${meta_root}" \
    --splits train \
    --bbox-edge-expand "${edge_expand}" \
    --min-bbox-half-size "${min_half}" \
    --target-coordinate-frame "${target_frame}" >&2
  echo "${meta_path}"
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

  echo "===== ${name} ====="
  meta_path="$(prepare_meta "${name}" "${edge_expand}" "${min_half}" "${target_frame}")"

  if [[ "${overwrite_output_dir}" != "True" ]] && \
    { [[ -f "${work_dir}/model.safetensors" ]] || find "${work_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; }; then
    echo "Skip training: existing checkpoint/model found in ${work_dir}"
  else
    if [[ -d "${work_dir}" ]] && [[ "${overwrite_output_dir}" != "True" ]]; then
      echo "Incomplete output directory found; rerunning with overwrite enabled: ${work_dir}"
      overwrite_output_dir=True
    fi
    EXPERIMENT_NAME="${name}" \
    META_PATH="${meta_path}" \
    USE_LLM_LORA="${USE_LLM_LORA}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    MAX_DYNAMIC_PATCH="${max_dynamic_patch}" \
    BBOX_EDGE_EXPAND="${edge_expand}" \
    MIN_BBOX_HALF_SIZE="${min_half}" \
    TRAINING_LOG_PATH="work_dirs/internvl_chat_v2_5/grasp_crop_hparams/${name}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${GP_ROOT}/train_grasp_crop_lmdb_lora.sh"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if ! find "${work_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
      echo "Skip eval: no checkpoint found in ${work_dir}"
      return
    fi
    mkdir -p "${out_dir}"
    WORK_DIR="${work_dir}" \
    OUT_DIR="${out_dir}" \
    GPUS=1 \
    DATASETS="${EVAL_DATASETS}" \
    BBOX_EDGE_EXPAND="${edge_expand}" \
    MIN_BBOX_HALF_SIZE="${min_half}" \
    TARGET_COORDINATE_FRAME="${target_frame}" \
    bash "${GP_ROOT}/eval/eval_grasp_crop_lmdb_lora.sh" \
      2>&1 | tee -a "${out_dir}/${name}.eval.log"
  fi
}

run_experiment "crop_object_lora16_lr8e-5_ep1_patch6_edge15_half50" 15 50 full_image 6
run_experiment "crop_full_lora16_lr8e-5_ep1_patch6_edge5_half40" 5 40 full_image 6
run_experiment "crop_full_lora16_lr8e-5_ep1_patch6_edge10_half40" 10 40 full_image 6
run_experiment "crop_frame_lora16_lr8e-5_ep1_patch6_edge5_half40" 5 40 crop_image 6
run_experiment "crop_frame_lora16_lr8e-5_ep1_patch6_edge10_half40" 10 40 crop_image 6
run_experiment "crop_frame_lora16_lr8e-5_ep1_patch8_edge10_half40" 10 40 crop_image 8
