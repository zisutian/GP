set -euo pipefail
set -x

GP_ROOT=${GP_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"}
source "${GP_ROOT}/scripts/grasp_runtime_helpers.sh"

GPUS=${GPUS:-${GRASP_TRAIN_GPUS}}
BATCH_SIZE=${BATCH_SIZE:-${GRASP_BATCH_SIZE}}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-${GRASP_PER_DEVICE_BATCH_SIZE}}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
LOG_LEVEL=${LOG_LEVEL:-warning}
LOG_LEVEL_REPLICA=${LOG_LEVEL_REPLICA:-error}

cd "${INTERNVL_CHAT_ROOT}"

USE_LLM_LORA=${USE_LLM_LORA:-${GRASP_DIRECT_USE_LLM_LORA}}
LEARNING_RATE=${LEARNING_RATE:-${GRASP_DIRECT_LEARNING_RATE}}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-${GRASP_DIRECT_NUM_TRAIN_EPOCHS}}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-${GRASP_DIRECT_MAX_DYNAMIC_PATCH}}
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-${GRASP_FORCE_IMAGE_SIZE}}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR:-False}

DEFAULT_EXPERIMENT_NAME="baseline_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch${MAX_DYNAMIC_PATCH}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}

export PYTHONPATH="${GP_ROOT}:$(pwd):${PYTHONPATH:-}"
export MASTER_PORT=${MASTER_PORT:-34229}
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-${LOG_LEVEL}}
export LAUNCHER=pytorch

MODEL_PATH=${MODEL_PATH:-"${GRASP_MODEL_PATH}"}
META_PATH=${META_PATH:-"${GRASP_DIRECT_META_PATH}"}
DATA_INDEX_ROOT=${DATA_INDEX_ROOT:-"$(dirname "${META_PATH}")"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${GRASP_DIRECT_RUN_ROOT}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${OUTPUT_DIR}/logs"}
TRAINING_LOG_PATH=${TRAINING_LOG_PATH:-"${LOG_DIR}/train.log"}

stage "data check: ${EXPERIMENT_NAME}"
python "${GP_ROOT}/data_tools/check_grasp_data.py" direct \
  --meta-path "${META_PATH}" \
  --data-index-root "${DATA_INDEX_ROOT}" \
  --splits train

stage "config: ${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

VCOT_CONFIG_PATH="${OUTPUT_DIR}/vcot_config.json"
python "${GP_ROOT}/scripts/grasp_experiment_metadata.py" write \
  --pipeline direct_grasp \
  --experiment-name "${EXPERIMENT_NAME}" \
  --meta-path "${META_PATH}" \
  --data-index-root "${DATA_INDEX_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --config-path "${VCOT_CONFIG_PATH}" \
  --use-llm-lora "${USE_LLM_LORA}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-dynamic-patch "${MAX_DYNAMIC_PATCH}" \
  --force-image-size "${FORCE_IMAGE_SIZE}"
echo "Direct config: ${VCOT_CONFIG_PATH}"

if [ -d "${OUTPUT_DIR}" ] && [ "${OVERWRITE_OUTPUT_DIR}" != "True" ]; then
  if ! find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    OVERWRITE_OUTPUT_DIR=True
  fi
fi

stage "train: ${EXPERIMENT_NAME}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --log_level ${LOG_LEVEL} \
  --log_level_replica ${LOG_LEVEL_REPLICA} \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir ${OVERWRITE_OUTPUT_DIR} \
  --force_image_size ${FORCE_IMAGE_SIZE} \
  --max_dynamic_patch ${MAX_DYNAMIC_PATCH} \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora ${USE_LLM_LORA} \
  --unfreeze_lm_head True \
  --vision_select_layer -1 \
  --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
  --bf16 True \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps 1 \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${TRAINING_LOG_PATH}"

stage "config copy: ${EXPERIMENT_NAME}"
python "${GP_ROOT}/scripts/grasp_experiment_metadata.py" write \
  --pipeline direct_grasp \
  --experiment-name "${EXPERIMENT_NAME}" \
  --meta-path "${META_PATH}" \
  --data-index-root "${DATA_INDEX_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --config-path "${VCOT_CONFIG_PATH}" \
  --use-llm-lora "${USE_LLM_LORA}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-dynamic-patch "${MAX_DYNAMIC_PATCH}" \
  --force-image-size "${FORCE_IMAGE_SIZE}" \
  --copy-to-checkpoints
