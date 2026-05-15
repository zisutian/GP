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
FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-448}"
TRAIN_SCRIPT="${GP_ROOT}/InternVL/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_grasp_direct_lmdb_lora.sh"

export CUDA_VISIBLE_DEVICES

prepare_meta() {
  local meta_path="${GP_ROOT}/data/vcot_grasp/direct/internvl_meta_train.json"
  "${PYTHON_BIN}" "${GP_ROOT}/scripts/ensure_grasp_data.py" direct \
    --meta-path "${meta_path}" \
    --splits train test_seen test_unseen >&2
  echo "${meta_path}"
}

write_config() {
  local name="$1"
  local lora_rank="$2"
  local learning_rate="$3"
  local epochs="$4"
  local max_dynamic_patch="$5"
  local meta_path="$6"
  local work_dir="$7"
  "${PYTHON_BIN}" "${GP_ROOT}/scripts/grasp_config.py" write \
    --pipeline direct_grasp \
    --experiment-name "${name}" \
    --meta-path "${meta_path}" \
    --output-dir "${work_dir}" \
    --use-llm-lora "${lora_rank}" \
    --learning-rate "${learning_rate}" \
    --num-train-epochs "${epochs}" \
    --max-dynamic-patch "${max_dynamic_patch}" \
    --force-image-size "${FORCE_IMAGE_SIZE}" \
    --copy-to-checkpoints
}

run_experiment() {
  local name="$1"
  local lora_rank="$2"
  local learning_rate="$3"
  local epochs="$4"
  local max_dynamic_patch="$5"
  local work_dir="${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_direct_hparams/${name}"
  local out_dir="${GP_ROOT}/result/vcot_grasp_direct/hparams/${name}"
  local overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"
  local meta_path
  local eval_datasets
  local eval_overwrite
  local action

  echo "===== ${name} ====="
  meta_path="$(prepare_meta)"
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
    USE_LLM_LORA="${lora_rank}" \
    LEARNING_RATE="${learning_rate}" \
    NUM_TRAIN_EPOCHS="${epochs}" \
    MAX_DYNAMIC_PATCH="${max_dynamic_patch}" \
    FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE}" \
    TRAINING_LOG_PATH="work_dirs/internvl_chat_v2_5/grasp_direct_hparams/${name}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${TRAIN_SCRIPT}"
  fi

  write_config "${name}" "${lora_rank}" "${learning_rate}" "${epochs}" "${max_dynamic_patch}" "${meta_path}" "${work_dir}"

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
    GPUS=1 \
    DATASETS="${eval_datasets}" \
    bash "${GP_ROOT}/eval/eval_grasp_direct_lmdb_lora.sh" \
      2>&1 | tee -a "${out_dir}/${name}.eval.log"
  fi
}

run_experiment "baseline_lora16_lr4e-5_ep1_patch6" 16 4e-5 1 6
run_experiment "lora8_lr4e-5_ep1_patch6" 8 4e-5 1 6
run_experiment "lora32_lr4e-5_ep1_patch6" 32 4e-5 1 6
run_experiment "lora16_lr2e-5_ep1_patch6" 16 2e-5 1 6
run_experiment "lora16_lr8e-5_ep1_patch6" 16 8e-5 1 6
run_experiment "lora16_lr4e-5_ep2_patch6" 16 4e-5 2 6
run_experiment "lora16_lr4e-5_ep1_patch4" 16 4e-5 1 4
run_experiment "lora16_lr4e-5_ep1_patch1" 16 4e-5 1 1
