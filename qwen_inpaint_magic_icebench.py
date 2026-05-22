#!/usr/bin/env python
"""
Batch image generation script using Qwen-Image-Edit + Blockwise-ControlNet-Inpaint.
Adapted for ICE-Bench.
"""

import os
import sys
import json
import torch
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
from PIL import Image, ImageChops
from queue import Empty


# -----------------------------------------------------------------------------
# Image & Mask Preprocessing
# -----------------------------------------------------------------------------

def calculate_optimal_dimensions(image: Image.Image, target_size=1328):
    """Calculate optimal dimensions aligned to 16 pixels, with target_size on the long side."""
    original_width, original_height = image.size

    if original_width >= original_height:
        width = target_size
        height = round(target_size * original_height / original_width)
    else:
        height = target_size
        width = round(target_size * original_width / original_height)

    # Align to 16
    width = (width // 16) * 16
    height = (height // 16) * 16

    # Ensure minimum size
    width = max(width, 16)
    height = max(height, 16)

    return width, height


def process_mask(mask_path, width, height):
    """Load and process mask image to target dimensions."""
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


def mask_l_to_rgb(mask_l: Image.Image) -> Image.Image:
    """Convert L-mode mask to RGB for ControlNet inpaint_mask (expects RGB)."""
    return Image.merge("RGB", [mask_l, mask_l, mask_l])


# -----------------------------------------------------------------------------
# Main & GPU Workers
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Qwen-Image-Edit + ControlNet-Inpaint on ICE-Bench dataset.")
    parser.add_argument("--input_path", required=True, type=str, help="Path to ICE-Bench dataset folder (contains data/images)")
    parser.add_argument("--json_path", required=True, type=str, help="Path to the task JSON file")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory for generated images")
    parser.add_argument("--gen_info_path", required=True, type=str, help="Path to save the gen_info.json mapping file")
    parser.add_argument("--steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", default=None, type=str, help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--target_size", default=1024, type=int, help="Target resolution on the long side (default: 1024)")
    parser.add_argument("--download_only", action="store_true", help="Only download models, do not run generation")

    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))

    print(f"[INFO] Using {args.num_gpus} GPU(s): {gpu_ids}")

    if not os.path.exists(args.json_path):
        print(f"[ERROR] JSON file not found: {args.json_path}")
        sys.exit(1)

    try:
        with open(args.json_path, 'r') as fp:
            data_list = json.load(fp)
        print(f"[INFO] Loaded {len(data_list)} tasks from {args.json_path}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON file: {args.json_path}")
        print(f"[ERROR] JSON decode error: {e}")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.download_only:
        download_models()
        print("[INFO] Model download completed.")
        return

    # Prepare tasks
    tasks = []
    for data in data_list:
        item_id = data['item_id']
        prompt = data['prompt']
        
        # Images are specified relatively in edit_image[0]
        img_rel_path = data['edit_image'][0] if 'edit_image' in data and data['edit_image'] else None
        mask_rel_path = data.get('back_mask', None)

        if not img_rel_path:
            continue
            
        img_path = os.path.join(args.input_path, img_rel_path)
        mask_path = os.path.join(args.input_path, mask_rel_path) if mask_rel_path else None
        
        out_name = f"{item_id}.png"
        out_path = os.path.join(args.output_path, out_name)
        
        tasks.append((item_id, img_path, mask_path, prompt, out_path))

    total_tasks = len(tasks)
    print(f"[INFO] Total tasks: {total_tasks}")

    # Dispatch based on GPU count
    if args.num_gpus == 1:
        gen_info = run_single_gpu(args, tasks, gpu_ids[0])
    else:
        gen_info = run_multi_gpu(args, tasks, gpu_ids)

    # Save gen_info.json
    os.makedirs(os.path.dirname(args.gen_info_path), exist_ok=True)
    with open(args.gen_info_path, 'w') as f:
        json.dump(gen_info, f, indent=4)
        
    print(f"[INFO] Processing completed. Saved {len(gen_info)} items to {args.gen_info_path}")


def download_models():
    from diffsynth.pipelines.qwen_image import ModelConfig
    print("[INFO] Downloading model weights (single process)...")
    configs = [
        ModelConfig(model_id="Qwen/Qwen-Image-Edit",   origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image",         origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image",         origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit",   origin_file_pattern="processor/"),
    ]
    for cfg in configs:
        print(f"[INFO]   Downloading: {cfg.model_id} / {cfg.origin_file_pattern}")
        cfg.download_if_necessary()
    print("[INFO] All model weights downloaded.")


def load_qwen_pipeline(device):
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }


    # Reserve 0.5 GB headroom
    vram_limit = torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 0.5

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors", **vram_config),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        vram_limit=vram_limit,
    )
    return pipe


def generate_image(pipe, input_img_path, mask_img_path, prompt, output_img_path, args, gpu_id=0):
    from diffsynth.pipelines.qwen_image import ControlNetInput

    if not os.path.exists(input_img_path):
        print(f"[GPU {gpu_id}] [ERROR] Input image not found: {input_img_path}")
        return False
    if not mask_img_path or not os.path.exists(mask_img_path):
        print(f"[GPU {gpu_id}] [ERROR] Mask not found: {mask_img_path}")
        return False

    try:
        image = Image.open(input_img_path).convert("RGB")
        width, height = calculate_optimal_dimensions(image, target_size=args.target_size)
        image = image.resize((width, height), Image.LANCZOS)

        final_mask_l = process_mask(mask_img_path, width, height)
        if final_mask_l is None:
            print(f"[GPU {gpu_id}] [ERROR] Mask processing failed: {mask_img_path}")
            return False

        inpaint_mask_rgb = mask_l_to_rgb(final_mask_l)

        result_image= pipe(
            prompt=prompt,
            seed=args.seed,
            input_image=image,
            inpaint_mask=inpaint_mask_rgb,
            blockwise_controlnet_inputs=[ControlNetInput(image=image, inpaint_mask=inpaint_mask_rgb)],
            num_inference_steps=args.steps,
            edit_image=image,
            height=height,
            width=width,
        )

        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        result_image.save(output_img_path)
        return True
    except Exception as e:
        print(f"[GPU {gpu_id}] [ERROR] Generation failed for {output_img_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_single_gpu(args, tasks, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"Loading Qwen-Image-Edit + ControlNet-Inpaint to {device}...")
    pipe = load_qwen_pipeline(device)

    gen_info = {}
    for task in tqdm(tasks):
        item_id, img_path, mask_path, prompt, out_path = task
        if not os.path.exists(out_path):
            generate_image(pipe, img_path, mask_path, prompt, out_path, args, gpu_id)
            
        # The gen_info paths should be relative to ICE-Bench directory
        # e.g., results/qwen_inpaint_icebench/images/item_id.png
        # Find relative path from ICE-Bench root
        # If output_path is results/method/images, we prepend it.
        rel_path = os.path.relpath(out_path, os.path.join(args.input_path, ".."))
        # As an alternative, if args.input_path is "dataset", out_path is "results/...", it's already a relative path!
        gen_info[item_id] = out_path

    return gen_info


def run_multi_gpu(args, tasks, gpu_ids):
    mp.set_start_method('spawn', force=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, len(tasks), args))
        p.start()
        processes.append(p)

    gen_info = {}
    with tqdm(total=len(tasks)) as pbar:
        completed = 0
        while completed < len(tasks):
            try:
                item_id, out_path, success = result_queue.get(timeout=1)
                # Ensure the path saved in gen_info is relative correctly to ICE-Bench
                gen_info[item_id] = out_path
                completed += 1
                pbar.update(1)
            except Empty:
                any_alive = any(p.is_alive() for p in processes)
                if not any_alive and completed < len(tasks):
                    print("[WARNING] All workers died before finishing tasks.")
                    break

    for p in processes:
        p.join()

    return gen_info


def gpu_worker(gpu_id, task_queue, result_queue, total_tasks, args):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model...")
    try:
        pipe = load_qwen_pipeline(device)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}")
        return

    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            break

        item_id, img_path, mask_path, prompt, out_path = task
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            success = generate_image(pipe, img_path, mask_path, prompt, out_path, args, gpu_id)
        else:
            success = True

        result_queue.put((item_id, out_path, success))

    print(f"[GPU {gpu_id}] Worker finished.")


if __name__ == "__main__":
    main()
