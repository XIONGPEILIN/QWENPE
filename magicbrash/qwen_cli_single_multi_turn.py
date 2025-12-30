import os
import sys
import json
import torch
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from queue import Empty

# 辅助函数：加载 LoRA 和 STE
def load_ste_and_lora(pipe, ckpt_path):
    """Load STE and LoRA weights into the pipeline."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    print(f"Loading weights from {ckpt_path}...")
    state = load_file(str(ckpt_path), device="cpu")

    # Load STE (Spatial Temporal Encoder)
    ste_prefix = "pipe.ste."
    ste_state = {k[len(ste_prefix):]: v for k, v in state.items() if k.startswith(ste_prefix)}
    if ste_state:
        print(f"  - Loading {len(ste_state)} STE tensors")
        pipe.ste.load_state_dict(ste_state, strict=False)

    # Load LoRA
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        print(f"  - Loading {len(lora_state)} LoRA tensors")
        # Move LoRA weights to the correct device and dtype
        lora_state = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in lora_state.items()}
        pipe.load_lora(pipe.dit, state_dict=lora_state)
    
    print("Checkpoint loaded successfully.")

# 辅助函数：图像预处理
def preprocess_image(image_pil, max_pixels=1048576):
    orig_width, orig_height = image_pil.size
    curr_pixels = orig_width * orig_height

    # Always resize to match max_pixels approx, whether scaling up or down
    factor = (max_pixels / curr_pixels) ** 0.5
    inter_width = int(orig_width * factor)
    inter_height = int(orig_height * factor)
    # print(f"Scaling from {orig_width}x{orig_height} to {inter_width}x{inter_height}")
    image_pil = image_pil.resize((inter_width, inter_height), Image.LANCZOS)

    # Align to 16 pixels
    target_width = ((inter_width + 15) // 16) * 16
    target_height = ((inter_height + 15) // 16) * 16
    
    if target_width != inter_width or target_height != inter_height:
        # print(f"Padding to {target_width}x{target_height}")
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image
    
    return image_pil, inter_width, inter_height, target_width, target_height

def process_mask(mask_path, inter_width, inter_height, target_width, target_height):
    if not os.path.exists(mask_path):
        print(f"[WARNING] Mask path does not exist: {mask_path}")
        return None

    input_mask = Image.open(mask_path)
    if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
        alpha = input_mask.convert('RGBA').split()[-1]
        if alpha.getextrema() != (255, 255):
            raw_mask = alpha.point(lambda p: 255 if p < 255 else 0).resize((inter_width, inter_height), Image.NEAREST)
        else:
            raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
    else:
        raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)

    # Pad Mask
    if target_width != inter_width or target_height != inter_height:
        back_mask = Image.new("L", (target_width, target_height), 0)
        back_mask.paste(raw_mask, (0, 0))
    else:
        back_mask = raw_mask
    return back_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--input_path", required=True, type=str, help="Path to images directory (e.g., ./dev/images)")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory")
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--cfg_scale", default=2.0, type=float)
    parser.add_argument("--blend_alpha", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--skip_iter", action="store_true", help="Skip iterative (multi-turn) generation")
    parser.add_argument("--num_gpus", default=torch.cuda.device_count(), type=int, help="Number of GPUs to use for parallel generation")
    parser.add_argument("--gpu_ids", default=None, type=str, help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). If not set, uses first num_gpus GPUs.")
    parser.add_argument("--inpaint_restoration_mode", default="mask", choices=["bbox", "mask"], help="Restoration mode for inpainting")
    
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
    
    print(f"[INFO] Using {args.num_gpus} GPU(s): {gpu_ids}")

    # Load JSON
    # Assumes edit_sessions.json is in the parent directory of input_path, similar to edit_cli...py
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

    # Single GPU mode (original behavior)
    if args.num_gpus == 1:
        run_single_gpu(args, data_json, gpu_ids[0])
    else:
        # Multi-GPU mode
        run_multi_gpu(args, data_json, gpu_ids)

    print("[INFO] Processing completed.")


def run_single_gpu(args, data_json, gpu_id):
    """Run on a single GPU (original behavior)."""
    device = f"cuda:{gpu_id}"
    
    # Init Pipeline
    print(f"Initializing Qwen Pipeline on {device}...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    load_ste_and_lora(pipe, args.ckpt)

    def generate_image(input_img_path, mask_img_path, prompt, output_img_path):
        # Check input image exists
        if not os.path.exists(input_img_path):
            print(f"[ERROR] Input image not found: {input_img_path}")
            return False
        
        try:
            # Preprocess
            image_pil = Image.open(input_img_path).convert("RGB")
            image_pil, inter_w, inter_h, target_w, target_h = preprocess_image(image_pil)
        except Exception as e:
            print(f"[ERROR] Failed to load/preprocess image: {input_img_path}")
            print(f"[ERROR] Exception: {e}")
            return False
        
        back_mask = None
        if mask_img_path:
            if os.path.exists(mask_img_path):
                try:
                    back_mask = process_mask(mask_img_path, inter_w, inter_h, target_w, target_h)
                except Exception as e:
                    print(f"[WARNING] Failed to process mask: {mask_img_path}")
                    print(f"[WARNING] Exception: {e}")
                    back_mask = None
            else:
                print(f"[WARNING] Mask file not found: {mask_img_path}")

        # Prompt construction
        full_prompt = "Picture 1 is the image to modify. " + prompt

        try:
            # Inference
            output_image, _ = pipe(
                prompt=full_prompt,
                edit_image=[image_pil],
                edit_image_auto_resize=False,
                back_mask=back_mask,
                height=target_h,
                width=target_w,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                inpaint_blend_alpha=args.blend_alpha,
                inpaint_restoration_mode=args.inpaint_restoration_mode,
                progress_bar_cmd=lambda x: x # Silent inner progress bar to avoid clutter
            )
            
            output_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to generate image for: {output_img_path}")
            print(f"[ERROR] Prompt: {prompt}")
            print(f"[ERROR] Exception: {e}")
            return False

    # 1. Iterative (Multi-Turn) Editing - 先执行
    if not args.skip_iter:
        print("Iterative (Multi-Turn) Editing......")
        for image_id, datas in tqdm(data_json.items()):
            # image_id like "139306"
            save_output_dir_path = os.path.join(args.output_path, image_id)
            os.makedirs(save_output_dir_path, exist_ok=True)
            
            for turn_id, data in enumerate(datas):
                # data['input'] like "139306-input.png" or "139306-output1.png"
                image_name = data['input']
                mask_name = data.get('mask') # mask path relative to session dir
                
                # Determine Input Image Path
                if turn_id == 0:
                    image_path = os.path.join(args.input_path, image_id, image_name)
                else:
                    # Use output from previous turn
                    if turn_id == 1:
                        prev_output_name = f"{image_id}_1.png"
                    else:
                        prev_output_name = f"{image_id}_iter_{turn_id}.png"
                    image_path = os.path.join(save_output_dir_path, prev_output_name)
                    
                    # Fallback: if previous output doesn't exist, maybe fail or skip?
                    # Strict multi-turn relies on previous output.
                    if not os.path.exists(image_path):
                        print(f"[WARNING] Missing previous output: {image_path}")
                        print(f"[WARNING] Skipping turn {turn_id+1} for image_id={image_id}")
                        continue

                # Determine Mask Path
                mask_path = None
                if mask_name:
                    mask_path = os.path.join(args.input_path, image_id, mask_name)

                # Determine Output Image Path
                if turn_id == 0:
                    save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
                else:
                    save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_iter_{turn_id + 1}.png")

                if os.path.exists(save_output_img_path):
                    continue

                success = generate_image(image_path, mask_path, data['instruction'], save_output_img_path)
                if not success:
                    print(f"[ERROR] Multi-turn: Failed to generate for image_id={image_id}, turn={turn_id+1}")

    # 2. Independent (Single-Turn) Editing - 后执行
    print("Independent (Single-Turn) Editing......")
    for image_id, datas in tqdm(data_json.items()):
        save_output_dir_path = os.path.join(args.output_path, image_id)
        os.makedirs(save_output_dir_path, exist_ok=True)

        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            
            # Always use GT input
            image_path = os.path.join(args.input_path, image_id, image_name)
            
            mask_path = None
            if mask_name:
                mask_path = os.path.join(args.input_path, image_id, mask_name)

            # Output: turn_id=0 -> _1.png (shared), turn_id>0 -> _inde_X.png
            if turn_id == 0:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
            else:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_inde_{turn_id + 1}.png")

            if os.path.exists(save_output_img_path):
                continue
            
            success = generate_image(image_path, mask_path, data['instruction'], save_output_img_path)
            if not success:
                print(f"[ERROR] Single-turn: Failed to generate for image_id={image_id}, turn={turn_id+1}")


def gpu_worker(gpu_id, task_queue, done_counter, total_tasks, args):
    """Worker process for a single GPU."""
    device = f"cuda:{gpu_id}"
    
    # Init Pipeline
    print(f"[GPU {gpu_id}] Initializing Qwen Pipeline on {device}...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    load_ste_and_lora(pipe, args.ckpt)
    print(f"[GPU {gpu_id}] Pipeline initialized.")

    def generate_image(input_img_path, mask_img_path, prompt, output_img_path):
        # Check input image exists
        if not os.path.exists(input_img_path):
            print(f"[GPU {gpu_id}][ERROR] Input image not found: {input_img_path}")
            return False
        
        try:
            # Preprocess
            image_pil = Image.open(input_img_path).convert("RGB")
            image_pil, inter_w, inter_h, target_w, target_h = preprocess_image(image_pil)
        except Exception as e:
            print(f"[GPU {gpu_id}][ERROR] Failed to load/preprocess image: {input_img_path}")
            print(f"[GPU {gpu_id}][ERROR] Exception: {e}")
            return False
        
        back_mask = None
        if mask_img_path:
            if os.path.exists(mask_img_path):
                try:
                    back_mask = process_mask(mask_img_path, inter_w, inter_h, target_w, target_h)
                except Exception as e:
                    print(f"[GPU {gpu_id}][WARNING] Failed to process mask: {mask_img_path}")
                    back_mask = None
            else:
                print(f"[GPU {gpu_id}][WARNING] Mask file not found: {mask_img_path}")

        # Prompt construction
        full_prompt = "Picture 1 is the image to modify. " + prompt

        try:
            # Inference
            output_image, _ = pipe(
                prompt=full_prompt,
                edit_image=[image_pil],
                edit_image_auto_resize=False,
                back_mask=back_mask,
                height=target_h,
                width=target_w,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                inpaint_blend_alpha=args.blend_alpha,
                inpaint_restoration_mode=args.inpaint_restoration_mode,
                progress_bar_cmd=lambda x: x
            )
            
            output_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[GPU {gpu_id}][ERROR] Failed to generate image for: {output_img_path}")
            print(f"[GPU {gpu_id}][ERROR] Exception: {e}")
            return False

    # Process tasks from queue
    empty_count = 0
    while True:
        try:
            task = task_queue.get(timeout=0.5)
            empty_count = 0  # Reset counter when we get a task
        except Empty:
            # Queue is empty, check if we should exit
            empty_count += 1
            # Wait for a few empty cycles to make sure queue is truly empty
            if empty_count >= 6:  # 3 seconds of empty queue
                break
            continue
        
        input_img_path, mask_img_path, prompt, output_img_path, task_info = task
        
        # Skip if already exists
        if os.path.exists(output_img_path):
            with done_counter.get_lock():
                done_counter.value += 1
            continue
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        
        success = generate_image(input_img_path, mask_img_path, prompt, output_img_path)
        
        with done_counter.get_lock():
            done_counter.value += 1
            current = done_counter.value
        
        if not success:
            print(f"[GPU {gpu_id}][ERROR] {task_info}")
        else:
            print(f"[GPU {gpu_id}] Completed {current}/{total_tasks}: {os.path.basename(output_img_path)}")

    print(f"[GPU {gpu_id}] Worker finished.")


def run_multi_gpu(args, data_json, gpu_ids):
    """Run on multiple GPUs in parallel."""
    mp.set_start_method('spawn', force=True)
    
    # 1. Handle multi-turn first - distribute sessions across GPUs
    if not args.skip_iter:
        print("[INFO] Running multi-turn tasks (distributing sessions across GPUs)...")
        
        # Create session queue
        session_queue = mp.Queue()
        session_ids = list(data_json.keys())
        total_sessions = len(session_ids)
        
        # Add all sessions to queue
        for session_id in session_ids:
            session_queue.put((session_id, data_json[session_id]))
        
        done_counter = mp.Value('i', 0)
        
        # Start multi-turn workers
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(target=multi_turn_worker, args=(gpu_id, session_queue, done_counter, total_sessions, args))
            p.start()
            processes.append(p)
        
        # Wait for all multi-turn processes to finish
        for p in processes:
            p.join()
        
        print(f"[INFO] Multi-turn parallel processing completed. {done_counter.value}/{total_sessions} sessions done.")
    
    # 2. Collect all single-turn tasks (independent tasks that can be parallelized)
    single_turn_tasks = []
    
    print("Collecting single-turn tasks...")
    for image_id, datas in data_json.items():
        save_output_dir_path = os.path.join(args.output_path, image_id)
        
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            
            # Always use GT input for single-turn
            image_path = os.path.join(args.input_path, image_id, image_name)
            
            mask_path = None
            if mask_name:
                mask_path = os.path.join(args.input_path, image_id, mask_name)

            # Output: turn_id=0 -> _1.png (shared), turn_id>0 -> _inde_X.png
            if turn_id == 0:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
            else:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_inde_{turn_id + 1}.png")

            if not os.path.exists(save_output_img_path):
                task_info = f"Single-turn: image_id={image_id}, turn={turn_id+1}"
                single_turn_tasks.append((image_path, mask_path, data['instruction'], save_output_img_path, task_info))

    total_tasks = len(single_turn_tasks)
    print(f"[INFO] Total single-turn tasks to process: {total_tasks}")
    
    if total_tasks == 0:
        print("[INFO] All single-turn tasks already completed.")
        return
    
    # Create task queue and counter
    task_queue = mp.Queue()
    done_counter = mp.Value('i', 0)
    
    # Add tasks to queue
    for task in single_turn_tasks:
        task_queue.put(task)
    
    # Start worker processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, done_counter, total_tasks, args))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print(f"[INFO] Single-turn parallel processing completed. {done_counter.value}/{total_tasks} tasks done.")


def multi_turn_worker(gpu_id, session_queue, done_counter, total_sessions, args):
    """Worker process for multi-turn editing on a single GPU."""
    device = f"cuda:{gpu_id}"
    
    # Init Pipeline
    print(f"[GPU {gpu_id}] Initializing Qwen Pipeline on {device} for multi-turn editing...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    load_ste_and_lora(pipe, args.ckpt)
    print(f"[GPU {gpu_id}] Pipeline initialized for multi-turn.")

    def generate_image(input_img_path, mask_img_path, prompt, output_img_path):
        if not os.path.exists(input_img_path):
            print(f"[GPU {gpu_id}][ERROR] Input image not found: {input_img_path}")
            return False
        
        try:
            image_pil = Image.open(input_img_path).convert("RGB")
            image_pil, inter_w, inter_h, target_w, target_h = preprocess_image(image_pil)
        except Exception as e:
            print(f"[GPU {gpu_id}][ERROR] Failed to load/preprocess image: {input_img_path}")
            return False
        
        back_mask = None
        if mask_img_path and os.path.exists(mask_img_path):
            try:
                back_mask = process_mask(mask_img_path, inter_w, inter_h, target_w, target_h)
            except Exception as e:
                back_mask = None

        full_prompt = "Picture 1 is the image to modify. " + prompt

        try:
            output_image, _ = pipe(
                prompt=full_prompt,
                edit_image=[image_pil],
                edit_image_auto_resize=False,
                back_mask=back_mask,
                height=target_h,
                width=target_w,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                inpaint_blend_alpha=args.blend_alpha,
                inpaint_restoration_mode=args.inpaint_restoration_mode,
                progress_bar_cmd=lambda x: x
            )
            output_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[GPU {gpu_id}][ERROR] Failed to generate image for: {output_img_path}")
            return False

    # Process sessions from queue
    empty_count = 0
    while True:
        try:
            session = session_queue.get(timeout=0.5)
            empty_count = 0
        except Empty:
            empty_count += 1
            if empty_count >= 6:  # 3 seconds of empty queue
                break
            continue
        
        image_id, datas = session
        save_output_dir_path = os.path.join(args.output_path, image_id)
        os.makedirs(save_output_dir_path, exist_ok=True)
        
        # Process all turns for this session sequentially
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            
            if turn_id == 0:
                image_path = os.path.join(args.input_path, image_id, image_name)
            else:
                if turn_id == 1:
                    prev_output_name = f"{image_id}_1.png"
                else:
                    prev_output_name = f"{image_id}_iter_{turn_id}.png"
                image_path = os.path.join(save_output_dir_path, prev_output_name)
                
                if not os.path.exists(image_path):
                    print(f"[GPU {gpu_id}][WARNING] Missing previous output: {image_path}")
                    continue

            mask_path = None
            if mask_name:
                mask_path = os.path.join(args.input_path, image_id, mask_name)

            # Output: turn_id=0 -> _1.png (shared), turn_id>0 -> _iter_X.png
            if turn_id == 0:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
            else:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_iter_{turn_id + 1}.png")

            if os.path.exists(save_output_img_path):
                continue

            success = generate_image(image_path, mask_path, data['instruction'], save_output_img_path)
            if not success:
                print(f"[GPU {gpu_id}][ERROR] Multi-turn: Failed for image_id={image_id}, turn={turn_id+1}")
        
        with done_counter.get_lock():
            done_counter.value += 1
            current = done_counter.value
        
        print(f"[GPU {gpu_id}] Multi-turn completed {current}/{total_sessions}: {image_id}")
    
    print(f"[GPU {gpu_id}] Multi-turn worker finished.")


def run_multi_turn_sequential(args, data_json, gpu_id):
    """Run multi-turn editing sequentially on a single GPU (used for single-GPU mode)."""
    device = f"cuda:{gpu_id}"
    
    # Init Pipeline
    print(f"Initializing Qwen Pipeline on {device} for multi-turn editing...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    load_ste_and_lora(pipe, args.ckpt)

    def generate_image(input_img_path, mask_img_path, prompt, output_img_path):
        if not os.path.exists(input_img_path):
            print(f"[ERROR] Input image not found: {input_img_path}")
            return False
        
        try:
            image_pil = Image.open(input_img_path).convert("RGB")
            image_pil, inter_w, inter_h, target_w, target_h = preprocess_image(image_pil)
        except Exception as e:
            print(f"[ERROR] Failed to load/preprocess image: {input_img_path}")
            return False
        
        back_mask = None
        if mask_img_path and os.path.exists(mask_img_path):
            try:
                back_mask = process_mask(mask_img_path, inter_w, inter_h, target_w, target_h)
            except Exception as e:
                back_mask = None

        full_prompt = "Picture 1 is the image to modify. " + prompt

        try:
            output_image, _ = pipe(
                prompt=full_prompt,
                edit_image=[image_pil],
                edit_image_auto_resize=False,
                back_mask=back_mask,
                height=target_h,
                width=target_w,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                inpaint_blend_alpha=args.blend_alpha,
                inpaint_restoration_mode=args.inpaint_restoration_mode,
                progress_bar_cmd=lambda x: x
            )
            output_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to generate image for: {output_img_path}")
            return False

    print("Iterative (Multi-Turn) Editing......")
    for image_id, datas in tqdm(data_json.items()):
        save_output_dir_path = os.path.join(args.output_path, image_id)
        os.makedirs(save_output_dir_path, exist_ok=True)
        
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            mask_name = data.get('mask')
            
            if turn_id == 0:
                image_path = os.path.join(args.input_path, image_id, image_name)
            else:
                if turn_id == 1:
                    prev_output_name = f"{image_id}_1.png"
                else:
                    prev_output_name = f"{image_id}_iter_{turn_id}.png"
                image_path = os.path.join(save_output_dir_path, prev_output_name)
                
                if not os.path.exists(image_path):
                    print(f"[WARNING] Missing previous output: {image_path}")
                    continue

            mask_path = None
            if mask_name:
                mask_path = os.path.join(args.input_path, image_id, mask_name)

            # Output: turn_id=0 -> _1.png (shared), turn_id>0 -> _iter_X.png
            if turn_id == 0:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
            else:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_iter_{turn_id + 1}.png")

            if os.path.exists(save_output_img_path):
                continue

            success = generate_image(image_path, mask_path, data['instruction'], save_output_img_path)
            if not success:
                print(f"[ERROR] Multi-turn: Failed for image_id={image_id}, turn={turn_id+1}")

if __name__ == "__main__":
    main()
