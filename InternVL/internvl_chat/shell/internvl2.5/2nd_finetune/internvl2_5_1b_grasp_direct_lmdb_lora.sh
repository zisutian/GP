set -euo pipefail
set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
LOG_LEVEL=${LOG_LEVEL:-warning}
LOG_LEVEL_REPLICA=${LOG_LEVEL_REPLICA:-error}
GP_ROOT=${GP_ROOT:-"/home/2025201095KZJ1/code/VCoTGrasp/GP"}

cd "${GP_ROOT}/InternVL/internvl_chat"

USE_LLM_LORA=${USE_LLM_LORA:-16}
LEARNING_RATE=${LEARNING_RATE:-8e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-6}
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-448}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR:-False}

DEFAULT_EXPERIMENT_NAME="lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch${MAX_DYNAMIC_PATCH}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}

export PYTHONPATH="${GP_ROOT}:$(pwd):${PYTHONPATH:-}"
export MASTER_PORT=${MASTER_PORT:-34229}
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-${LOG_LEVEL}}
export LAUNCHER=pytorch

MODEL_PATH=${MODEL_PATH:-"${GP_ROOT}/InternVL/pretrained/OpenGVLab/InternVL2_5-1B"}
META_PATH=${META_PATH:-"${GP_ROOT}/data/vcot_grasp/direct/internvl_meta_train.json"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"work_dirs/internvl_chat_v2_5/grasp_direct_hparams"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${OUTPUT_DIR}/logs"}
TRAINING_LOG_PATH=${TRAINING_LOG_PATH:-"${LOG_DIR}/train.log"}

python "${GP_ROOT}/scripts/ensure_grasp_data.py" direct \
  --meta-path "${META_PATH}" \
  --splits train

mkdir -p "${LOG_DIR}"

VCOT_CONFIG_PATH="${OUTPUT_DIR}/vcot_config.json"
python "${GP_ROOT}/scripts/grasp_config.py" write \
  --pipeline direct_grasp \
  --experiment-name "${EXPERIMENT_NAME}" \
  --meta-path "${META_PATH}" \
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

python "${GP_ROOT}/scripts/grasp_config.py" write \
  --pipeline direct_grasp \
  --experiment-name "${EXPERIMENT_NAME}" \
  --meta-path "${META_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --config-path "${VCOT_CONFIG_PATH}" \
  --use-llm-lora "${USE_LLM_LORA}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-dynamic-patch "${MAX_DYNAMIC_PATCH}" \
  --force-image-size "${FORCE_IMAGE_SIZE}" \
  --copy-to-checkpoints
