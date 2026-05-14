#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_DATASETS="${EVAL_DATASETS:-test_seen,test_unseen}"

export CUDA_VISIBLE_DEVICES

run_experiment() {
  local name="$1"
  local lora_rank="$2"
  local learning_rate="$3"
  local epochs="$4"
  local max_dynamic_patch="$5"
  local work_dir="${GP_ROOT}/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_direct_hparams/${name}"
  local out_dir="${GP_ROOT}/result/vcot_grasp_direct/hparams/${name}"
  local overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"

  echo "===== ${name} ====="
  if [[ "${overwrite_output_dir}" != "True" ]] && \
    { [[ -f "${work_dir}/model.safetensors" ]] || find "${work_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; }; then
    echo "Skip training: existing checkpoint/model found in ${work_dir}"
  else
    if [[ -d "${work_dir}" ]] && [[ "${overwrite_output_dir}" != "True" ]]; then
      echo "Incomplete output directory found; rerunning with overwrite enabled: ${work_dir}"
      overwrite_output_dir=True
    fi
    EXPERIMENT_NAME="${name}" \
    USE_LLM_LORA="${lora_rank}" \
    LEARNING_RATE="${learning_rate}" \
    NUM_TRAIN_EPOCHS="${epochs}" \
    MAX_DYNAMIC_PATCH="${max_dynamic_patch}" \
    TRAINING_LOG_PATH="work_dirs/internvl_chat_v2_5/grasp_direct_hparams/${name}/logs/train.log" \
    OVERWRITE_OUTPUT_DIR="${overwrite_output_dir}" \
    GPUS="${GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    bash "${GP_ROOT}/train_grasp_direct_lmdb_lora.sh"
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
