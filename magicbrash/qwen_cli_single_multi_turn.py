import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

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
    parser.add_argument("--blend_alpha", default=0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--skip_iter", action="store_true", help="Skip iterative (multi-turn) generation")
    
    args = parser.parse_args()

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

    # Init Pipeline
    print(f"Initializing Qwen Pipeline on {args.device}...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
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
                progress_bar_cmd=lambda x: x # Silent inner progress bar to avoid clutter
            )
            
            output_image.save(output_img_path)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to generate image for: {output_img_path}")
            print(f"[ERROR] Prompt: {prompt}")
            print(f"[ERROR] Exception: {e}")
            return False

    # 1. Iterative (Multi-Turn) Editing
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
                    # Previous output name pattern: img_id + '_1.png' or '_iter_X.png'
                    prev_turn_idx = turn_id # 1-based index of previous turn is turn_id
                    if prev_turn_idx == 1:
                        prev_output_name = f"{image_id}_1.png"
                    else:
                        prev_output_name = f"{image_id}_iter_{prev_turn_idx}.png"
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

    # 2. Independent (Single-Turn) Editing
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

            if turn_id == 0:
                # Same as first turn of iterative, usually already generated
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_1.png")
            else:
                save_output_img_path = os.path.join(save_output_dir_path, f"{image_id}_inde_{turn_id + 1}.png")

            if os.path.exists(save_output_img_path):
                continue
            
            success = generate_image(image_path, mask_path, data['instruction'], save_output_img_path)
            if not success:
                print(f"[ERROR] Single-turn: Failed to generate for image_id={image_id}, turn={turn_id+1}")

    print("[INFO] Processing completed.")

if __name__ == "__main__":
    main()
