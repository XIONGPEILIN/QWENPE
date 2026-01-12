#!/usr/bin/env python
import os
import sys
import json
import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
from diffusers import FluxFillPipeline
from queue import Empty

"""
Flux Fill Inference Script for Qwen PE Dataset (Top 1000).
Reads a flat JSON list (e.g., dataset_qwen_pe_top1000.json) and processes edits.
"""

# -----------------------------------------------------------------------------
# Flux Helper Functions
# -----------------------------------------------------------------------------

def calculate_optimal_dimensions(image: Image.Image):
    original_width, original_height = image.size
    MIN_ASPECT_RATIO = 9 / 16
    MAX_ASPECT_RATIO = 16 / 9
    FIXED_DIMENSION = 1024

    original_aspect_ratio = original_width / original_height

    if original_aspect_ratio > 1:
        width = FIXED_DIMENSION
        height = round(FIXED_DIMENSION / original_aspect_ratio)
    else:
        height = FIXED_DIMENSION
        width = round(FIXED_DIMENSION * original_aspect_ratio)

    width = (width // 8) * 8
    height = (height // 8) * 8

    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = (height * MAX_ASPECT_RATIO // 8) * 8
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = (width / MIN_ASPECT_RATIO // 8) * 8

    width = max(width, 576) if width == FIXED_DIMENSION else width
    height = max(height, 576) if height == FIXED_DIMENSION else height

    return width, height

def process_mask(mask_path, width, height):
    if not os.path.exists(mask_path):
        return None
    
    input_mask = Image.open(mask_path)
    final_mask = Image.new("L", (width, height), 0)

    # Handle Alpha channel transparency -> Mask
    if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
        alpha = input_mask.convert('RGBA').split()[-1]
        if alpha.getextrema() != (255, 255):
            m = alpha.point(lambda p: 255 if p < 255 else 0).resize((width, height), Image.NEAREST)
            final_mask = ImageChops.lighter(final_mask, m)
        else:
            m = input_mask.convert("L").resize((width, height), Image.NEAREST)
            final_mask = ImageChops.lighter(final_mask, m)
    else:
        m = input_mask.convert("L").resize((width, height), Image.NEAREST)
        final_mask = ImageChops.lighter(final_mask, m)
        
    return final_mask

# -----------------------------------------------------------------------------
# Main & GPU Workers
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Flux Fill on Qwen PE Dataset.")
    parser.add_argument("--json_path", required=True, type=str, help="Path to the JSON dataset file (e.g. dataset_qwen_pe_top1000.json)")
    parser.add_argument("--input_path", required=True, type=str, help="Base path for images (e.g., .../pico-banana-400k-subject_driven/openimages/)")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory")
    parser.add_argument("--steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", default=3.5, type=float, help="Guidance scale")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", default=None, type=str, help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')")

    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
    
    print(f"[INFO] Using {args.num_gpus} GPU(s): {gpu_ids}")

    # Load JSON
    if not os.path.exists(args.json_path):
        print(f"[ERROR] JSON file not found: {args.json_path}")
        sys.exit(1)
    
    try:
        with open(args.json_path, 'r') as fp:
            data_list = json.load(fp)
        print(f"[INFO] Loaded {len(data_list)} entries from {args.json_path}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON file: {args.json_path}")
        print(f"[ERROR] JSON decode error: {e}")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Dispatch based on GPU count
    if args.num_gpus == 1:
        run_single_gpu(args, data_list, gpu_ids[0])
    else:
        run_multi_gpu(args, data_list, gpu_ids)

    print("[INFO] Processing completed.")


def run_single_gpu(args, data_list, gpu_id):
    """Run on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"Loading Flux model to {device}...")
    try:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    def generate(input_img_path, mask_img_path, prompt, output_img_path):
        if not os.path.exists(input_img_path):
            print(f"[ERROR] Input image not found: {input_img_path}")
            return False
        if not mask_img_path or not os.path.exists(mask_img_path):
            print(f"[ERROR] Mask not found: {mask_img_path}")
            return False
        
        try:
            image = Image.open(input_img_path).convert("RGB")
            width, height = calculate_optimal_dimensions(image)
            image = image.resize((width, height), Image.LANCZOS)
            
            final_mask = process_mask(mask_img_path, width, height)
            if final_mask is None:
                 return False

            result_image = pipe(
                prompt=prompt,
                image=image,
                mask_image=final_mask,
                height=height,
                width=width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=torch.Generator(device).manual_seed(args.seed)
            ).images[0]
            
            result_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[ERROR] Generation failed for {output_img_path}: {e}")
            return False

    print("Starting processing...")
    for entry in tqdm(data_list):
        # Extract paths from JSON entry
        # JSON structure: {"prompt": "...", "edit_image": ["path/to/img.png"], "back_mask": "path/to/mask.png", ...}
        
        rel_img_path = entry['edit_image'][0]
        rel_mask_path = entry['back_mask']
        prompt = entry['prompt']
        
        # Construct full paths
        img_path = os.path.join(args.input_path, rel_img_path)
        mask_path = os.path.join(args.input_path, rel_mask_path)
        
        # Determine output filename (use basename of input image)
        img_basename = os.path.basename(rel_img_path)
        out_path = os.path.join(args.output_path, img_basename)
        
        if os.path.exists(out_path):
            continue
            
        generate(img_path, mask_path, prompt, out_path)


def run_multi_gpu(args, data_list, gpu_ids):
    """Run on multiple GPUs in parallel."""
    mp.set_start_method('spawn', force=True)
    
    print("Collecting tasks...")
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
    print(f"[INFO] Total tasks to process: {total_tasks}")
    
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
        print(f"[INFO] Processing completed.")


def gpu_worker(gpu_id, task_queue, done_counter, total_tasks, args):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Flux...")
    try:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}")
        return

    def generate(input_path, mask_path, prompt, output_path):
        try:
            image = Image.open(input_path).convert("RGB")
            width, height = calculate_optimal_dimensions(image)
            image = image.resize((width, height), Image.LANCZOS)
            final_mask = process_mask(mask_path, width, height)
            
            res = pipe(
                prompt=prompt, image=image, mask_image=final_mask,
                height=height, width=width,
                guidance_scale=args.guidance_scale, num_inference_steps=args.steps,
                generator=torch.Generator(device).manual_seed(args.seed)
            ).images[0]
            res.save(output_path)
            return True
        except Exception as e:
            print(f"[GPU {gpu_id}] Gen Error: {e}")
            return False

    while True:
        try:
            task = task_queue.get(timeout=3)
        except Empty:
            break
        
        img_path, mask_path, prompt, out_path = task
        # Double check existence
        if not os.path.exists(out_path):
            # Ensure output dir exists (though main creates the root, keeping for safety)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            generate(img_path, mask_path, prompt, out_path)
            
        with done_counter.get_lock():
            done_counter.value += 1
            if done_counter.value % 10 == 0:
                print(f"[GPU {gpu_id}] Progress: {done_counter.value}/{total_tasks}")
    
    print(f"[GPU {gpu_id}] Worker finished.")


if __name__ == "__main__":
    main()