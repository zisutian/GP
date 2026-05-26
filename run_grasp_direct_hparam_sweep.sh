#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GP_ROOT}/scripts/grasp_runtime_helpers.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GRASP_DIRECT_CUDA_VISIBLE_DEVICES}}"
GPUS="${GPUS:-${GRASP_TRAIN_GPUS}}"
BATCH_SIZE="${BATCH_SIZE:-${GRASP_BATCH_SIZE}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${GRASP_PER_DEVICE_BATCH_SIZE}}"
RUN_EVAL="${RUN_EVAL:-${GRASP_RUN_EVAL}}"
OVERWRITE_EVAL_RESULTS="${OVERWRITE_EVAL_RESULTS:-${GRASP_OVERWRITE_EVAL_RESULTS}}"
EVAL_DATASETS="${EVAL_DATASETS:-${GRASP_EVAL_DATASETS}}"
PYTHON_BIN="${PYTHON_BIN:-${GRASP_PYTHON_BIN}}"
FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-${GRASP_FORCE_IMAGE_SIZE}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${GRASP_DIRECT_TRAIN_SCRIPT}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${GRASP_DIRECT_EVAL_SCRIPT}}"

export CUDA_VISIBLE_DEVICES

require_direct_data() {
  local meta_path="$1"
  local splits=(train)
  if should_run_eval; then
    for split in $(default_dataset_splits "${EVAL_DATASETS}"); do
      splits+=("${split}")
    done
  fi
  "${PYTHON_BIN}" "${GP_ROOT}/data_tools/check_grasp_data.py" direct \
    --meta-path "${meta_path}" \
    --data-index-root "${GRASP_DIRECT_INDEX_ROOT}" \
    --splits "${splits[@]}"
}

write_config() {
  local name="$1"
  local lora_rank="$2"
  local learning_rate="$3"
  local epochs="$4"
  local max_dynamic_patch="$5"
  local meta_path="$6"
  local work_dir="$7"
  "${PYTHON_BIN}" "${GP_ROOT}/scripts/grasp_experiment_metadata.py" write \
    --pipeline direct_grasp \
    --experiment-name "${name}" \
    --meta-path "${meta_path}" \
    --data-index-root "${GRASP_DIRECT_INDEX_ROOT}" \
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
  local work_dir="${GRASP_DIRECT_RUN_ROOT}/${name}"
  local out_dir="${GRASP_DIRECT_RESULT_ROOT}/${name}"
  local overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"
  local meta_path
  local eval_datasets
  local eval_overwrite
  local action

  stage "experiment: ${name}"
  meta_path="${GRASP_DIRECT_META_PATH}"

  stage "data check"
  require_direct_data "${meta_path}"

  stage "train"
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
    GRASP_DATASET_ROOT="${GRASP_DATASET_ROOT}" \
    DATA_INDEX_ROOT="${GRASP_DIRECT_INDEX_ROOT}" \
    OUTPUT_ROOT="${GRASP_DIRECT_RUN_ROOT}" \
    MODEL_PATH="${GRASP_MODEL_PATH}" \
    TRAINING_LOG_PATH="${work_dir}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${TRAIN_SCRIPT}"
  fi

  stage "config"
  write_config "${name}" "${lora_rank}" "${learning_rate}" "${epochs}" "${max_dynamic_patch}" "${meta_path}" "${work_dir}"

  if should_run_eval; then
    stage "eval"
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
    CUDA_VISIBLE_DEVICES="${GRASP_DIRECT_EVAL_CUDA_VISIBLE_DEVICES}" \
    GPUS="${GRASP_EVAL_GPUS}" \
    DATASETS="${eval_datasets}" \
    bash "${EVAL_SCRIPT}" \
      2>&1 | tee -a "${out_dir}/${name}.eval.log"
  else
    stage "eval"
    echo "Skip eval: RUN_EVAL=${RUN_EVAL}"
  fi
}

while IFS=$'\t' read -r name lora_rank learning_rate epochs max_dynamic_patch; do
  run_experiment "${name}" "${lora_rank}" "${learning_rate}" "${epochs}" "${max_dynamic_patch}"
done < <("${PYTHON_BIN}" "${GP_ROOT}/grasp_settings.py" experiments direct)
