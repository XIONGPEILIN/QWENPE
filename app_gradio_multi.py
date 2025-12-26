import os
import sys
from pathlib import Path
import torch
import gradio as gr
from PIL import Image
from safetensors.torch import load_file

# -----------------------------------------------------------------------------
# 1. 环境与路径设置
# -----------------------------------------------------------------------------
REPO_ROOT_SENTINEL = "dataset_qwen_pe_all.json"

def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        if (cur / REPO_ROOT_SENTINEL).exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate {REPO_ROOT_SENTINEL} upwards from {start}")

repo_root = find_repo_root(Path(__file__).resolve())
sys.path.append(str(repo_root / "DiffSynth-Studio"))

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# -----------------------------------------------------------------------------
# 2. 模型加载逻辑
# -----------------------------------------------------------------------------
def load_ste_and_lora(pipe, ckpt_path: Path):
    """加载 STE 和 LoRA 权重到 pipeline"""
    if not ckpt_path.exists():
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

print("Initializing Pipeline...")
# 使用指定的模型配置
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:6",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

lora_path = repo_root / "train/Qwen-Image-Edit-2509_lora-rank512-cfg/step-30000.safetensors"

load_ste_and_lora(pipe, lora_path)

def preprocess_image(image_pil, max_pixels=1048576):
    orig_width, orig_height = image_pil.size
    curr_pixels = orig_width * orig_height

    # Always resize to match max_pixels approx, whether scaling up or down
    factor = (max_pixels / curr_pixels) ** 0.5
    inter_width = int(orig_width * factor)
    inter_height = int(orig_height * factor)
    print(f"Scaling from {orig_width}x{orig_height} to {inter_width}x{inter_height}")
    image_pil = image_pil.resize((inter_width, inter_height), Image.LANCZOS)

    # Align to 16 pixels
    target_width = ((inter_width + 15) // 16) * 16
    target_height = ((inter_height + 15) // 16) * 16
    
    if target_width != inter_width or target_height != inter_height:
        print(f"Padding to {target_width}x{target_height}")
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image
    
    return image_pil, inter_width, inter_height, target_width, target_height

# -----------------------------------------------------------------------------
# 3. 推理函数
# -----------------------------------------------------------------------------
def predict(input_dict, input_image2, input_mask, prompt, cfg_scale, steps, seed, inpaint_blend_alpha, progress=gr.Progress(track_tqdm=True)):
    """
    input_dict: Gradio Image editor return (contains 'image' and 'mask')
    input_image2: Optional second image for dual-input editing
    input_mask: Optional uploaded mask image
    """
    if input_dict is None:
        return None
    
    image_pil = input_dict["background"].convert("RGB")
    image_pil, inter_width, inter_height, target_width, target_height = preprocess_image(image_pil)
    
    # 处理 Mask
    if input_mask is not None:
        # 提取 Mask：支持 Alpha 通道或黑白灰度
        if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
            alpha = input_mask.convert('RGBA').split()[-1]
            # 如果 Alpha 通道包含透明信息（非全不透明）
            if alpha.getextrema() != (255, 255):
                # 逻辑反转：Alpha < 255 (透明/半透明) -> Mask=255 (修改)
                #           Alpha = 255 (不透明) -> Mask=0 (不修改)
                raw_mask = alpha.point(lambda p: 255 if p < 255 else 0).resize((inter_width, inter_height), Image.NEAREST)
            else:
                # 全不透明，按普通灰度处理 (白色=修改)
                raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
        else:
            raw_mask = input_mask.convert("L").resize((inter_width, inter_height), Image.NEAREST)
    else:
        # 处理在编辑器中涂抹的 Mask
        mask_layer = None
        if input_dict.get("layers") and len(input_dict["layers"]) > 0:
            mask_layer = input_dict["layers"][0]

        if mask_layer:
            alpha = mask_layer.split()[-1]
            alpha = alpha.resize((inter_width, inter_height), Image.NEAREST)
            raw_mask = Image.eval(alpha, lambda a: 255 if a > 0 else 0)
        else:
            raw_mask = Image.new("L", (inter_width, inter_height), 0)

    # 对 Mask 进行与主图相同的 Padding
    if target_width != inter_width or target_height != inter_height:
        back_mask = Image.new("L", (target_width, target_height), 0)
        back_mask.paste(raw_mask, (0, 0))
    else:
        back_mask = raw_mask

    # 处理第二张图片
    edit_images = [image_pil]
    if input_image2 is not None:
        image2_processed, _, _, _, _ = preprocess_image(input_image2.convert("RGB"))
        edit_images.append(image2_processed)

    width, height = target_width, target_height
    prompt = "Picture 1 is the image to modify. " + prompt
    print(f"Processing: Prompt='{prompt}', Final Size={width}x{height}, Seed={seed}")

    output_image, sub_image = pipe(
        prompt=prompt,
        edit_image=edit_images, # Pipeline expects a list
        edit_image_auto_resize=False,
        back_mask=back_mask,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
        inpaint_blend_alpha=float(inpaint_blend_alpha),
    )
    
    return output_image, sub_image

# -----------------------------------------------------------------------------
# 4. Gradio 界面
# -----------------------------------------------------------------------------
css = """
#col-container {max-width: 1000px; margin-left: auto; margin-right: auto;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Qwen Image Edit (LoRA 14000 Step)")
        gr.Markdown("上传图片并涂抹(Mask)你想要编辑的区域。")
        
        with gr.Row():
            with gr.Column():
                # 使用 ImageEditor 允许涂抹
                input_image = gr.ImageEditor(
                    label="Input Image & Mask",
                    type="pil",
                    brush=gr.Brush(colors=["#000000"], color_mode="fixed"), # 只是视觉上的画笔
                    eraser=gr.Eraser(),
                    layers=False, # 简化的编辑器
                    height=600,
                )
                input_image2 = gr.Image(label="Reference Image (Optional)", type="pil")
                input_mask = gr.Image(label="Upload Mask (Optional)", type="pil", image_mode="RGBA")
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the edit (e.g. 'Make the banana red')")
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    inpaint_blend_alpha = gr.Slider(label="Inpaint Blend Alpha", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    seed = gr.Number(label="Seed", value=0)
                
                run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                result_main = gr.Image(label="Main Output")
                result_sub = gr.Image(label="Sub Image (Detail)")

    run_btn.click(
        fn=predict,
        inputs=[input_image, input_image2, input_mask, prompt, cfg_scale, steps, seed, inpaint_blend_alpha],
        outputs=[result_main, result_sub]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7999, share=True)
