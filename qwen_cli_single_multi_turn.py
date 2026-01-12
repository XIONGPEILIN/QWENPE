#!/usr/bin/env python
import os
import sys
import json
import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file
from queue import Empty

# -----------------------------------------------------------------------------
# Qwen Setup & Helper Functions
# -----------------------------------------------------------------------------

def load_ste_and_lora(pipe, ckpt_path):
    """Load STE and LoRA weights into the pipeline."""
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Missing checkpoint: {ckpt_path}")
        return
    
    print(f"Loading weights from {ckpt_path}...")
    state = load_file(str(ckpt_path), device="cpu")

    # Load STE (Spatial Temporal Encoder)
    ste_prefix = "pipe.ste."
    ste_state = {k[len(ste_prefix):]: v for k, v in state.items() if k.startswith(ste_prefix)}
    if ste_state:
        if hasattr(pipe, 'ste') and pipe.ste is not None:
            print(f"  - Loading {len(ste_state)} STE tensors")
            pipe.ste.load_state_dict(ste_state, strict=False)
        else:
            print(f"  - Skipping {len(ste_state)} STE tensors (module not in pipe)")

    # Load LoRA
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        print(f"  - Loading {len(lora_state)} LoRA tensors")
        # Move LoRA weights to the correct device and dtype
        lora_state = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in lora_state.items()}
        pipe.load_lora(pipe.dit, state_dict=lora_state)
    
    print("Checkpoint loaded successfully.")

def preprocess_image(image_pil, max_pixels=1048576):
    orig_width, orig_height = image_pil.size
    curr_pixels = orig_width * orig_height

    factor = (max_pixels / curr_pixels) ** 0.5
    inter_width = int(orig_width * factor)
    inter_height = int(orig_height * factor)
    image_pil = image_pil.resize((inter_width, inter_height), Image.LANCZOS)

    # Align to 16 pixels
    target_width = ((inter_width + 15) // 16) * 16
    target_height = ((inter_height + 15) // 16) * 16
    
    if target_width != inter_width or target_height != inter_height:
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image
    
    return image_pil, inter_width, inter_height, target_width, target_height

def process_mask(mask_path, inter_width, inter_height, target_width, target_height):
    input_mask = Image.open(mask_path)
    if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
        alpha = input_mask.convert('RGBA').split()[-1]
        if alpha.getextrema() != (255, 255):
            raw_mask = alpha.point(lambda p: 255 if p < 255 else 0).resize((inter_width, inter_height), Image.NEAREST)
        else:
            raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
    else:
        raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)

    if target_width != inter_width or target_height != inter_height:
        back_mask = Image.new("L", (target_width, target_height), 0)
        back_mask.paste(raw_mask, (0, 0))
    else:
        back_mask = raw_mask
    return back_mask

# -----------------------------------------------------------------------------
# Main & GPU Workers
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Qwen Image Edit on Qwen PE Dataset.")
    parser.add_argument("--json_path", required=True, type=str, help="Path to JSON dataset")
    parser.add_argument("--input_path", required=True, type=str, help="Base path for images")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory")
    parser.add_argument("--lora_path", required=True, type=str, help="Path to LoRA safetensors")
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2511", type=str, help="Qwen model ID")
    parser.add_argument("--steps", default=50, type=int, help="Inference steps")
    parser.add_argument("--cfg_scale", default=2.0, type=float, help="CFG scale")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--inpaint_blend_alpha", default=0.0, type=float, help="Blending alpha for inpainting")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--gpu_ids", default=None, type=str, help="GPU IDs (e.g. '0,1')")
    parser.add_argument("--use_bbox_mask", action="store_true", help="Use BBox mask instead of pixel mask for blending")

    args = parser.parse_args()

    # Import pipeline here to avoid issues with multiprocessing before setup
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
    
    print(f"[INFO] Using {args.num_gpus} GPU(s): {gpu_ids}")

    if not os.path.exists(args.json_path):
        print(f"[ERROR] JSON not found: {args.json_path}")
        sys.exit(1)
    
    with open(args.json_path, 'r') as fp:
        data_list = json.load(fp)
    print(f"[INFO] Loaded {len(data_list)} entries")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    run_multi_gpu(args, data_list, gpu_ids)

def run_multi_gpu(args, data_list, gpu_ids):
    mp.set_start_method('spawn', force=True)
    
    tasks = []
    for entry in data_list:
        rel_img_path = entry['edit_image'][0]
        rel_mask_path = entry['back_mask']
        prompt = entry['prompt']
        
        img_path = os.path.join(args.input_path, rel_img_path)
        mask_path = os.path.join(args.input_path, rel_mask_path)
        
        img_basename = os.path.basename(rel_img_path)
        out_path = os.path.join(args.output_path, img_basename)
        
        if not os.path.exists(out_path):
            tasks.append((img_path, mask_path, prompt, out_path))
                
    total_tasks = len(tasks)
    print(f"[INFO] Total tasks: {total_tasks}")
    
    if total_tasks > 0:
        task_queue = mp.Queue()
        for t in tasks: task_queue.put(t)
        
        done_counter = mp.Value('i', 0)
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, done_counter, total_tasks, args))
            p.start()
            processes.append(p)
            
        for p in processes: p.join()

def gpu_worker(gpu_id, task_queue, done_counter, total_tasks, args):
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Qwen...")
    
    # 2511 vs 2509 file pattern check
    origin_file_pattern = "transformer/diffusion_pytorch_model*.safetensors"
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern=origin_file_pattern),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    load_ste_and_lora(pipe, args.lora_path)

    while True:
        try:
            task = task_queue.get(timeout=3)
        except Empty:
            break
        
        img_path, mask_path, prompt, out_path = task
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_pil, iw, ih, tw, th = preprocess_image(image_pil)
            back_mask = process_mask(mask_path, iw, ih, tw, th)

            output_image, _ = pipe(
                prompt=prompt,
                edit_image=[image_pil],
                edit_image_auto_resize=False,
                back_mask=back_mask,
                height=th,
                width=tw,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                inpaint_blend_alpha=args.inpaint_blend_alpha,
                use_bbox_mask=args.use_bbox_mask,
            )
            output_image.save(out_path)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            
        with done_counter.get_lock():
            done_counter.value += 1
            if done_counter.value % 10 == 0:
                print(f"[GPU {gpu_id}] Progress: {done_counter.value}/{total_tasks}")

if __name__ == "__main__":
    main()