#!/usr/bin/env python
import os
import sys
import json
import argparse
import torch
import torch.multiprocessing as mp
from queue import Empty
from tqdm import tqdm
from PIL import Image, ImageChops
import traceback

# Ensure ACE_plus and its modules are in the path
repo_root = "/host/ssd2/xiong-p/qwenpe/ACE_plus"
sys.path.append(repo_root)

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

    width = (width // 16) * 16
    height = (height // 16) * 16

    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = (height * MAX_ASPECT_RATIO // 16) * 16
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = (width / MIN_ASPECT_RATIO // 16) * 16

    width = int(max(width, 576)) if width == FIXED_DIMENSION else int(width)
    height = int(max(height, 576)) if height == FIXED_DIMENSION else int(height)

    return width, height

def process_mask(mask_path, width, height):
    if not os.path.exists(mask_path):
        return None
    
    input_mask = Image.open(mask_path)
    final_mask = Image.new("L", (width, height), 0)

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

def gpu_worker(gpu_id, task_queue, results_queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from scepter.modules.utils.config import Config
        from scepter.modules.utils.file_system import FS
        from inference.ace_plus_diffusers import ACEPlusDiffuserInference
    except Exception as e:
        print(f"[GPU {gpu_id}] Import failed: {e}")
        traceback.print_exc()
        return
    
    try:
        fs_list = [
            Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
            Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
            Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
        ]
        for one_fs in fs_list: FS.init_fs_client(one_fs)
    except Exception as e:
        print(f"[GPU {gpu_id}] FS Init failed: {e}")
        traceback.print_exc()
        return

    print(f"[GPU {gpu_id}] Initializing ACE++ Pipeline...")
    try:
        pipe_cfg = Config(load=True, cfg_file=os.path.join(repo_root, "config/ace_plus_diffusers_infer.yaml"))
        pipe = ACEPlusDiffuserInference()
        pipe.init_from_cfg(pipe_cfg)
        
        local_lora_path = os.path.join(repo_root, "models/local_editing/local_editing/comfyui_local_lora16.safetensors")
        if not os.path.exists(local_lora_path):
            lora_path = "ms://ali-vilab/ACE_plus/local_editing.safetensors"
            local_lora_path = FS.get_from(lora_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to init pipeline: {e}")
        traceback.print_exc()
        return

    def run_inference(img_path, mask_path, prompt, out_path):
        try:
            image_raw = Image.open(img_path).convert("RGB")
            width, height = calculate_optimal_dimensions(image_raw)
            input_img = image_raw.resize((width, height), Image.LANCZOS)
            
            input_mask = process_mask(mask_path, width, height)
            if input_mask is None:
                return False
            
            image, seed = pipe(
                reference_image=None,
                edit_image=input_img,
                edit_mask=input_mask,
                prompt=prompt,
                output_height=height,
                output_width=width,
                sampler='flow_euler',
                sample_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                repainting_scale=1.0,
                lora_path=local_lora_path
            )
            image.save(out_path)
            return True
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in run_inference: {e}")
            traceback.print_exc()
            return False

    while True:
        try:
            task = task_queue.get(timeout=2)
        except Empty:
            break
        
        img_path, mask_path, prompt, out_path = task
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            success = run_inference(img_path, mask_path, prompt, out_path)
            results_queue.put(('inde', success))
        else:
            results_queue.put(('inde', True))

    print(f"[GPU {gpu_id}] Worker finished.")

def main():
    parser = argparse.ArgumentParser(description="ACE++ ICE-Bench Inference")
    parser.add_argument("--input_json", required=True, type=str, help="Path to input JSON")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory")
    parser.add_argument("--data_prefix", required=True, type=str, help="Prefix for image paths in JSON")
    parser.add_argument("--steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--guide_scale", default=50, type=float, help="Guidance scale")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7", type=str, help="Comma-separated GPU IDs")

    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"[ERROR] JSON file not found: {args.input_json}")
        sys.exit(1)
    
    with open(args.input_json, 'r') as fp:
        data_json = json.load(fp)
    print(f"[INFO] Loaded {len(data_json)} samples from {args.input_json}")

    os.makedirs(args.output_dir, exist_ok=True)

    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue()
    results_queue = mp.Queue()

    total_tasks = 0
    for item in data_json:
        item_id = item['item_id']
        prompt = item['prompt']
        # ICE-Bench JSON contains edit_image as list
        img_path = os.path.join(args.data_prefix, item['edit_image'][0])
        mask_path = os.path.join(args.data_prefix, item['back_mask'])
        
        out_path = os.path.join(args.output_dir, f"{item_id}.png")
        task_queue.put((img_path, mask_path, prompt, out_path))
        total_tasks += 1

    print(f"[INFO] Total tasks: {total_tasks}")

    if total_tasks == 0:
        print("[INFO] No tasks to process.")
        return

    gpu_list = [int(x.strip()) for x in args.gpu_ids.split(",")]
    processes = [mp.Process(target=gpu_worker, args=(gid, task_queue, results_queue, args)) for gid in gpu_list]
    for p in processes: p.start()

    with tqdm(total=total_tasks, desc="ACE++ Inference Progress") as pbar:
        completed = 0
        while completed < total_tasks:
            try:
                msg = results_queue.get(timeout=1)
                pbar.update(1)
                completed += 1
            except Empty:
                if not any(p.is_alive() for p in processes):
                    break
            except KeyboardInterrupt:
                print("\nInterrupted. Cleaning up...")
                for p in processes: p.terminate()
                break

    for p in processes: p.join()
    print(f"\nDone. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
