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

USE_LLM_LORA=${USE_LLM_LORA:-${GRASP_CROP_USE_LLM_LORA}}
LEARNING_RATE=${LEARNING_RATE:-${GRASP_CROP_LEARNING_RATE}}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-${GRASP_CROP_NUM_TRAIN_EPOCHS}}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-${GRASP_CROP_MAX_DYNAMIC_PATCH}}
BBOX_EDGE_EXPAND=${BBOX_EDGE_EXPAND:-${GRASP_CROP_BBOX_EDGE_EXPAND}}
MIN_BBOX_HALF_SIZE=${MIN_BBOX_HALF_SIZE:-${GRASP_CROP_MIN_BBOX_HALF_SIZE}}
TARGET_COORDINATE_FRAME=${TARGET_COORDINATE_FRAME:-${GRASP_CROP_TARGET_COORDINATE_FRAME}}
TARGET_GRASP_INDEX=${TARGET_GRASP_INDEX:-${GRASP_CROP_TARGET_GRASP_INDEX}}
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

if [ "${TARGET_COORDINATE_FRAME}" = "crop_image" ]; then
  TARGET_FRAME_TAG="frame"
elif [ "${TARGET_COORDINATE_FRAME}" = "full_image" ] && [ "${BBOX_EDGE_EXPAND}" = "15" ] && [ "${MIN_BBOX_HALF_SIZE}" = "50" ]; then
  TARGET_FRAME_TAG="object"
elif [ "${TARGET_COORDINATE_FRAME}" = "full_image" ]; then
  TARGET_FRAME_TAG="full"
else
  echo "Unsupported TARGET_COORDINATE_FRAME=${TARGET_COORDINATE_FRAME}; expected full_image or crop_image." >&2
  exit 1
fi

TARGET_GRASP_TAG=""
if [ "${TARGET_GRASP_INDEX}" != "0" ]; then
  TARGET_GRASP_TAG="_g${TARGET_GRASP_INDEX}"
fi

DEFAULT_EXPERIMENT_NAME="crop_${TARGET_FRAME_TAG}_lora${USE_LLM_LORA}_lr${LEARNING_RATE}_ep${NUM_TRAIN_EPOCHS}_patch${MAX_DYNAMIC_PATCH}_edge${BBOX_EDGE_EXPAND}_half${MIN_BBOX_HALF_SIZE}${TARGET_GRASP_TAG}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}

export PYTHONPATH="${GP_ROOT}:$(pwd):${PYTHONPATH:-}"
export MASTER_PORT=${MASTER_PORT:-34239}
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-${LOG_LEVEL}}
export LAUNCHER=pytorch

MODEL_PATH=${MODEL_PATH:-"${GRASP_MODEL_PATH}"}
META_PATH=${META_PATH:-"${GRASP_CROP_META_PATH}"}
DATA_INDEX_ROOT=${DATA_INDEX_ROOT:-"$(dirname "${META_PATH}")"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${GRASP_CROP_RUN_ROOT}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${OUTPUT_DIR}/logs"}
TRAINING_LOG_PATH=${TRAINING_LOG_PATH:-"${LOG_DIR}/train.log"}

stage "data check: ${EXPERIMENT_NAME}"
python "${GP_ROOT}/data_tools/check_grasp_data.py" crop \
  --meta-path "${META_PATH}" \
  --data-index-root "${DATA_INDEX_ROOT}" \
  --splits train \
  --bbox-edge-expand "${BBOX_EDGE_EXPAND}" \
  --min-bbox-half-size "${MIN_BBOX_HALF_SIZE}" \
  --target-coordinate-frame "${TARGET_COORDINATE_FRAME}" \
  --target-grasp-index "${TARGET_GRASP_INDEX}"

stage "config: ${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

VCOT_CONFIG_PATH="${OUTPUT_DIR}/vcot_config.json"
python "${GP_ROOT}/scripts/grasp_experiment_metadata.py" write \
  --pipeline oracle_crop \
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
echo "Crop config: ${VCOT_CONFIG_PATH}"

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
  --pipeline oracle_crop \
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
