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

    if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
        alpha = input_mask.convert('RGBA').split()[-1]
        if alpha.getextrema() != (255, 255):
            # Alpha < 255 (Transparent) -> Edit (White in mask)
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
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python magicbrash/flux_fill.py --input_path "magicbrash/test/images" --output_path "magicbrash/fluxfill_results_test" --device "cuda"
def main():
    parser = argparse.ArgumentParser(description="Run Flux Fill on dataset (single/multi-turn).")
    parser.add_argument("--input_path", required=True, type=str, help="Path to images directory (e.g., ./dev/images)")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory")
    parser.add_argument("--steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", default=3.5, type=float, help="Guidance scale")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--skip_iter", action="store_true", help="Skip iterative (multi-turn) generation")
    parser.add_argument("--num_gpus", default=7, type=int, help="Number of GPUs to use for parallel generation")
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
    json_path = os.path.join(args.input_path, '..', 'edit_sessions.json')
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        print(f"[ERROR] Please ensure edit_sessions.json exists in the parent directory of input_path")
        sys.exit(1)
    
    try:
        with open(json_path, 'r') as fp:
            data_json = json.load(fp)
        print(f"[INFO] Loaded {len(data_json)} sessions from {json_path}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON file: {json_path}")
        print(f"[ERROR] JSON decode error: {e}")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Dispatch based on GPU count
    if args.num_gpus == 1:
        run_single_gpu(args, data_json, gpu_ids[0])
    else:
        run_multi_gpu(args, data_json, gpu_ids)

    print("[INFO] Processing completed.")


def run_single_gpu(args, data_json, gpu_id):
    """Run on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"Loading Flux model to {device}...")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)

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

    # 1. Iterative (Multi-Turn)
    if not args.skip_iter:
        print("Iterative (Multi-Turn) Editing...")
        for image_id, datas in tqdm(data_json.items()):
            save_dir = os.path.join(args.output_path, image_id)
            os.makedirs(save_dir, exist_ok=True)
            
            for turn_id, data in enumerate(datas):
                image_name = data['input']
                mask_name = data.get('mask')
                
                # Input Image
                if turn_id == 0:
                    img_path = os.path.join(args.input_path, image_id, image_name)
                else:
                    prev_name = f"{image_id}_1.png" if turn_id == 1 else f"{image_id}_iter_{turn_id}.png"
                    img_path = os.path.join(save_dir, prev_name)
                    if not os.path.exists(img_path):
                        print(f"[WARNING] Missing previous output: {img_path}")
                        continue
                
                # Mask
                if mask_name:
                    mask_path = os.path.join(args.input_path, image_id, mask_name)
                else:
                    # Flux requires a mask. If no mask provided in JSON, we can't edit properly with FluxFill.
                    print(f"[WARNING] No mask provided for {image_id} turn {turn_id}, skipping.")
                    continue

                # Output Path
                out_name = f"{image_id}_1.png" if turn_id == 0 else f"{image_id}_iter_{turn_id+1}.png"
                out_path = os.path.join(save_dir, out_name)
                
                if os.path.exists(out_path): continue
                
                generate(img_path, mask_path, data['instruction'], out_path)

    # 2. Independent (Single-Turn)
    print("Independent (Single-Turn) Editing...")
    for image_id, datas in tqdm(data_json.items()):
        save_dir = os.path.join(args.output_path, image_id)
        os.makedirs(save_dir, exist_ok=True)
        
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            img_path = os.path.join(args.input_path, image_id, image_name)
            
            if mask_name:
                mask_path = os.path.join(args.input_path, image_id, mask_name)
            else:
                print(f"[WARNING] No mask provided for {image_id} turn {turn_id}, skipping.")
                continue

            out_name = f"{image_id}_1.png" if turn_id == 0 else f"{image_id}_inde_{turn_id+1}.png"
            out_path = os.path.join(save_dir, out_name)
            
            if os.path.exists(out_path): continue
            
            generate(img_path, mask_path, data['instruction'], out_path)


def run_multi_gpu(args, data_json, gpu_ids):
    """Run on multiple GPUs in parallel."""
    mp.set_start_method('spawn', force=True)
    
    # 1. Multi-turn
    if not args.skip_iter:
        print("[INFO] Running multi-turn tasks (distributing sessions)...")
        session_queue = mp.Queue()
        session_ids = list(data_json.keys())
        total_sessions = len(session_ids)
        
        for sid in session_ids:
            session_queue.put((sid, data_json[sid]))
            
        done_counter = mp.Value('i', 0)
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(target=multi_turn_worker, args=(gpu_id, session_queue, done_counter, total_sessions, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        print(f"[INFO] Multi-turn processing completed.")

    # 2. Single-turn
    print("Collecting single-turn tasks...")
    tasks = []
    for image_id, datas in data_json.items():
        save_dir = os.path.join(args.output_path, image_id)
        
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            img_path = os.path.join(args.input_path, image_id, image_name)
            mask_path = os.path.join(args.input_path, image_id, mask_name) if mask_name else None
            
            out_name = f"{image_id}_1.png" if turn_id == 0 else f"{image_id}_inde_{turn_id+1}.png"
            out_path = os.path.join(save_dir, out_name)
            
            if not os.path.exists(out_path) and mask_path:
                tasks.append((img_path, mask_path, data['instruction'], out_path))
                
    total_tasks = len(tasks)
    print(f"[INFO] Total single-turn tasks: {total_tasks}")
    
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
        print(f"[INFO] Single-turn processing completed.")


def multi_turn_worker(gpu_id, session_queue, done_counter, total_sessions, args):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Flux...")
    try:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}")
        return

    def generate(input_path, mask_path, prompt, output_path):
        if not os.path.exists(input_path) or not os.path.exists(mask_path): return False
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
            print(f"[GPU {gpu_id}] Gen Error {output_path}: {e}")
            return False

    while True:
        try:
            # Wait for a short time, if empty for a while, exit
            session = session_queue.get(timeout=3)
        except Empty:
            break
            
        image_id, datas = session
        save_dir = os.path.join(args.output_path, image_id)
        os.makedirs(save_dir, exist_ok=True)
        
        for turn_id, data in enumerate(datas):
            # Input
            if turn_id == 0:
                img_path = os.path.join(args.input_path, image_id, data['input'])
            else:
                prev_name = f"{image_id}_1.png" if turn_id == 1 else f"{image_id}_iter_{turn_id}.png"
                img_path = os.path.join(save_dir, prev_name)
                if not os.path.exists(img_path): continue
            
            # Mask
            mask_path = os.path.join(args.input_path, image_id, data['mask']) if data.get('mask') else None
            if not mask_path: continue
            
            # Output
            out_name = f"{image_id}_1.png" if turn_id == 0 else f"{image_id}_iter_{turn_id+1}.png"
            out_path = os.path.join(save_dir, out_name)
            
            if not os.path.exists(out_path):
                generate(img_path, mask_path, data['instruction'], out_path)
        
        with done_counter.get_lock():
            done_counter.value += 1
            if done_counter.value % 5 == 0:
                print(f"[GPU {gpu_id}] Sessions {done_counter.value}/{total_sessions} done")
    
    print(f"[GPU {gpu_id}] Multi-turn worker finished.")


def gpu_worker(gpu_id, task_queue, done_counter, total_tasks, args):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Flux (Single-turn)...")
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
        # Double check existence (another worker might have done it, though queue should be unique)
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            generate(img_path, mask_path, prompt, out_path)
            
        with done_counter.get_lock():
            done_counter.value += 1
            if done_counter.value % 20 == 0:
                print(f"[GPU {gpu_id}] Tasks {done_counter.value}/{total_tasks} done")
    
    print(f"[GPU {gpu_id}] Single-turn worker finished.")


if __name__ == "__main__":
    main()