#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GP_ROOT}/scripts/grasp_runtime_helpers.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GRASP_VCOT_CUDA_VISIBLE_DEVICES}}"
GPUS="${GPUS:-${GRASP_TRAIN_GPUS}}"
BATCH_SIZE="${BATCH_SIZE:-${GRASP_BATCH_SIZE}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${GRASP_PER_DEVICE_BATCH_SIZE}}"
RUN_EVAL="${RUN_EVAL:-${GRASP_RUN_EVAL}}"
OVERWRITE_EVAL_RESULTS="${OVERWRITE_EVAL_RESULTS:-${GRASP_OVERWRITE_EVAL_RESULTS}}"
EVAL_DATASETS="${EVAL_DATASETS:-${GRASP_EVAL_DATASETS}}"
PYTHON_BIN="${PYTHON_BIN:-${GRASP_PYTHON_BIN}}"

FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE:-${GRASP_FORCE_IMAGE_SIZE}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${GRASP_VCOT_TRAIN_SCRIPT}}"

export CUDA_VISIBLE_DEVICES

require_vcot_data() {
  local name="$1"
  local bbox_ratio="$2"
  local edge_expand="$3"
  local min_half="$4"
  local target_frame="$5"
  local target_grasp_index="$6"
  local meta_root="${GRASP_VCOT_INDEX_ROOT}/${name}"
  local crop_root="${meta_root}/crop"
  local meta_path="${meta_root}/internvl_meta_train.json"
  local splits=()
  local args

  if should_run_eval; then
    for split in $(default_dataset_splits "${EVAL_DATASETS}"); do
      splits+=("${split}")
    done
  fi
  args=(
    "${GP_ROOT}/data_tools/check_grasp_data.py" vcot
    --meta-path "${meta_path}"
    --data-index-root "${meta_root}"
    --crop-root "${crop_root}"
    --bbox-root "${GRASP_BBOX_INDEX_ROOT}"
    --bbox-edge-expand "${edge_expand}"
    --min-bbox-half-size "${min_half}"
    --target-coordinate-frame "${target_frame}"
    --target-grasp-index "${target_grasp_index}"
  )
  if [[ "${#splits[@]}" -gt 0 ]]; then
    args+=(--eval-splits "${splits[@]}")
  fi
  "${PYTHON_BIN}" "${args[@]}"
}

write_config() {
  local name="$1"
  local bbox_ratio="$2"
  local edge_expand="$3"
  local min_half="$4"
  local target_frame="$5"
  local max_dynamic_patch="$6"
  local meta_path="$7"
  local work_dir="$8"
  local use_lora="$9"
  local learning_rate="${10}"
  local epochs="${11}"
  local target_grasp_index="${12}"
  local data_index_root="${GRASP_VCOT_INDEX_ROOT}/${name}"
  local crop_root="${GRASP_VCOT_INDEX_ROOT}/${name}/crop"

  "${PYTHON_BIN}" "${GP_ROOT}/scripts/grasp_experiment_metadata.py" write \
    --pipeline predicted_vcot \
    --experiment-name "${name}" \
    --meta-path "${meta_path}" \
    --data-index-root "${data_index_root}" \
    --output-dir "${work_dir}" \
    --crop-root "${crop_root}" \
    --bbox-root "${GRASP_BBOX_INDEX_ROOT}" \
    --use-llm-lora "${use_lora}" \
    --learning-rate "${learning_rate}" \
    --num-train-epochs "${epochs}" \
    --max-dynamic-patch "${max_dynamic_patch}" \
    --force-image-size "${FORCE_IMAGE_SIZE}" \
    --bbox-ratio "${bbox_ratio}" \
    --bbox-edge-expand "${edge_expand}" \
    --min-bbox-half-size "${min_half}" \
    --target-coordinate-frame "${target_frame}" \
    --target-grasp-index "${target_grasp_index}" \
    --copy-to-checkpoints
}

run_experiment() {
  local name="$1"
  local use_lora="$2"
  local learning_rate="$3"
  local epochs="$4"
  local bbox_ratio="$5"
  local edge_expand="$6"
  local min_half="$7"
  local target_frame="$8"
  local max_dynamic_patch="$9"
  local target_grasp_index="${10}"
  local crop_root="${GRASP_VCOT_INDEX_ROOT}/${name}/crop"
  local work_dir="${GRASP_VCOT_RUN_ROOT}/${name}"
  local out_dir="${GRASP_VCOT_RESULT_ROOT}/${name}"
  local overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"
  local meta_path
  local eval_datasets
  local eval_overwrite
  local action

  stage "experiment: ${name}"
  meta_path="${GRASP_VCOT_INDEX_ROOT}/${name}/internvl_meta_train.json"

  stage "data check"
  require_vcot_data "${name}" "${bbox_ratio}" "${edge_expand}" "${min_half}" "${target_frame}" "${target_grasp_index}"

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
    USE_LLM_LORA="${use_lora}" \
    LEARNING_RATE="${learning_rate}" \
    NUM_TRAIN_EPOCHS="${epochs}" \
    MAX_DYNAMIC_PATCH="${max_dynamic_patch}" \
    BBOX_RATIO="${bbox_ratio}" \
    BBOX_EDGE_EXPAND="${edge_expand}" \
    MIN_BBOX_HALF_SIZE="${min_half}" \
    TARGET_COORDINATE_FRAME="${target_frame}" \
    TARGET_GRASP_INDEX="${target_grasp_index}" \
    FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE}" \
    CROP_ROOT="${crop_root}" \
    BBOX_ROOT="${GRASP_BBOX_INDEX_ROOT}" \
    GRASP_DATASET_ROOT="${GRASP_DATASET_ROOT}" \
    DATA_INDEX_ROOT="${GRASP_VCOT_INDEX_ROOT}/${name}" \
    OUTPUT_ROOT="${GRASP_VCOT_RUN_ROOT}" \
    MODEL_PATH="${GRASP_MODEL_PATH}" \
    TRAINING_LOG_PATH="${work_dir}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${TRAIN_SCRIPT}"
  fi

  stage "config"
  write_config "${name}" "${bbox_ratio}" "${edge_expand}" "${min_half}" "${target_frame}" "${max_dynamic_patch}" "${meta_path}" "${work_dir}" "${use_lora}" "${learning_rate}" "${epochs}" "${target_grasp_index}"

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
    GPUS="${GRASP_EVAL_GPUS}" \
    DATASETS="${eval_datasets}" \
    bash "${GP_ROOT}/eval/eval_grasp_vcot_lmdb_lora.sh" \
      2>&1 | tee -a "${out_dir}/${name}.eval.log"
  else
    stage "eval"
    echo "Skip eval: RUN_EVAL=${RUN_EVAL}"
  fi
}

while IFS=$'\t' read -r name use_lora learning_rate epochs bbox_ratio edge_expand min_half target_frame max_dynamic_patch target_grasp_index; do
  run_experiment "${name}" "${use_lora}" "${learning_rate}" "${epochs}" "${bbox_ratio}" "${edge_expand}" "${min_half}" "${target_frame}" "${max_dynamic_patch}" "${target_grasp_index}"
done < <("${PYTHON_BIN}" "${GP_ROOT}/grasp_settings.py" experiments vcot)
