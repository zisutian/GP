set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
LOG_LEVEL=${LOG_LEVEL:-warning}
LOG_LEVEL_REPLICA=${LOG_LEVEL_REPLICA:-error}
GP_ROOT=${GP_ROOT:-"/home/2025201095KZJ1/code/VCoTGrasp/GP"}

export PYTHONPATH="${GP_ROOT}:$(pwd):${PYTHONPATH:-}"
export MASTER_PORT=${MASTER_PORT:-34229}
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-${LOG_LEVEL}}
export LAUNCHER=pytorch

MODEL_PATH=${MODEL_PATH:-"${GP_ROOT}/InternVL/pretrained/OpenGVLab/InternVL2_5-1B"}
META_PATH=${META_PATH:-"${GP_ROOT}/data/vcot_grasp/direct/internvl_meta_train.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"work_dirs/internvl_chat_v2_5/internvl2_5_1b_grasp_direct_lmdb_lora"}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
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
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --unfreeze_lm_head True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
