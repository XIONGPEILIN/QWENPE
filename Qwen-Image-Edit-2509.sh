#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen-image}"
export WANDB_NAME="${WANDB_NAME:-Qwen-Image-Edit-2509-root}"


# 两阶段拆分训练：阶段 1 预处理缓存，阶段 2 正式训练

CACHE_PATH="./data/Qwen-Image-Edit-2509_lora-rank512-split-cache"
OUTPUT_PATH="./train/Qwen-Image-Edit-2509_lora-rank512"

# # 阶段 1：仅跑前处理（文本编码、VAE 等），生成缓存
# accelerate launch DiffSynth-Studio/examples/qwen_image/model_training/train.py \
#   --dataset_base_path pico-banana-400k-subject_driven/openimages \
#   --dataset_metadata_path dataset_qwen_pe_all.json \
#   --data_file_keys "image,edit_image,ref_gt,back_mask" \
#   --extra_inputs "edit_image,ref_gt,back_mask" \
#   --max_pixels 1048576 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "$CACHE_PATH" \
#   --lora_base_model "dit" \
#   --trainable_models "ste" \
#   --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
#   --lora_rank 512 \
#   --use_gradient_checkpointing \
#   --dataset_num_workers 8 \
#   --find_unused_parameters \
#   --task "sft:data_process"

# 阶段 2：加载缓存，只训练 DiT

accelerate launch DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path "$CACHE_PATH" \
  --data_file_keys "image,edit_image,ref_gt,back_mask" \
  --extra_inputs "edit_image,ref_gt,back_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "dit" \
  --trainable_models "ste" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 512 \
  --dataset_num_workers 2 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --find_unused_parameters \
  --task "sft:train" \
  --save_steps 1000 \
  --use_gradient_checkpointing \
  --lora_checkpoint 'train/Qwen-Image-Edit-2509_lora-rank512-old/step-15000.safetensors' \
  --disable_epoch_resume
