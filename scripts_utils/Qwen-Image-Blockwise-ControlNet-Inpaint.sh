#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen-image}"
export WANDB_NAME="${WANDB_NAME:-Qwen-Image-2511-ControlNet-Ablation-Official}"

# 1. 路径定义
CACHE_PATH="./data/Qwen-Image-Edit-2511_lora-rank512-split-cache"
OUTPUT_PATH="./models/train/Qwen-Image-2511-ControlNet-Inpaint"

# 冠军方案：ZeRO-3 + BF16 + 无梯度检查点
ACCELERATE_CONFIG="--config_file accelerate_zero3_config.yaml"

# 2. 执行训练
# 使用官方预训练 ControlNet 权重: DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors
accelerate launch $ACCELERATE_CONFIG org/examples/qwen_image/model_training/train.py \
  --dataset_base_path "$CACHE_PATH" \
  --data_file_keys "image,edit_image,back_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors" \
  --learning_rate 1 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "blockwise_controlnet" \
  --extra_inputs "edit_image,back_mask" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --find_unused_parameters \
  --task "sft:train" \
  --save_steps 1000 \
  --gradient_accumulation_steps 32 \
  --dataset_num_workers 8 \
  --cfg_drop_prob 0.1 \
  --disable_epoch_resume \
  --zero_cond_t
