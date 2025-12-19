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
    device="cuda:2",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# 加载 14000 step 的 LoRA
# 假设是 'new' 文件夹下的 (即当前训练的主目录)
lora_path = repo_root / "train/Qwen-Image-Edit-2509_lora-rank512/step-14000.safetensors"
if not lora_path.exists():
    # 如果找不到，尝试一下 old 目录，或者报错
    print(f"Warning: {lora_path} not found. Checking 'old' directory...")
    lora_path_old = repo_root / "train/Qwen-Image-Edit-2509_lora-rank512-old/step-14000.safetensors"
    if lora_path_old.exists():
        lora_path = lora_path_old
    else:
        raise FileNotFoundError("Could not find step-14000.safetensors in either train directory.")

load_ste_and_lora(pipe, lora_path)

# -----------------------------------------------------------------------------
# 3. 推理函数
# -----------------------------------------------------------------------------
def predict(input_dict, prompt, cfg_scale, steps, seed):
    """
    input_dict: Gradio Image editor return (contains 'image' and 'mask')
    """
    if input_dict is None:
        return None
    
    image_pil = input_dict["background"].convert("RGB")
    orig_width, orig_height = image_pil.size
    
    # 1. 等比例缩放以符合 max_pixels (1048576)
    max_pixels = 1048576
    curr_pixels = orig_width * orig_height
    if curr_pixels > max_pixels:
        factor = (max_pixels / curr_pixels) ** 0.5
        inter_width = int(orig_width * factor)
        inter_height = int(orig_height * factor)
        print(f"Scaling down from {orig_width}x{orig_height} to {inter_width}x{inter_height}")
        image_pil = image_pil.resize((inter_width, inter_height), Image.LANCZOS)
    else:
        inter_width, inter_height = orig_width, orig_height

    # 2. 计算 16 的倍数目标尺寸
    target_width = ((inter_width + 15) // 16) * 16
    target_height = ((inter_height + 15) // 16) * 16
    
    # 3. 执行 Padding
    if target_width != inter_width or target_height != inter_height:
        print(f"Padding from {inter_width}x{inter_height} to {target_width}x{target_height}")
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image
    
    # 4. 处理 Mask (同样缩放 + Padding)
    mask_layer = input_dict["layers"][0]
    if mask_layer:
        alpha = mask_layer.split()[-1]
        # 如果缩放过，mask 也要缩放
        if curr_pixels > max_pixels:
            alpha = alpha.resize((inter_width, inter_height), Image.NEAREST)
        raw_mask = Image.eval(alpha, lambda a: 255 if a > 0 else 0)
    else:
        raw_mask = Image.new("L", (inter_width, inter_height), 0)

    # Padding mask
    if target_width != inter_width or target_height != inter_height:
        back_mask = Image.new("L", (target_width, target_height), 0)
        back_mask.paste(raw_mask, (0, 0))
    else:
        back_mask = raw_mask

    width, height = image_pil.size
    print(f"Processing: Prompt='{prompt}', Final Size={width}x{height}, Seed={seed}")

    output_image, sub_image = pipe(
        prompt=prompt,
        edit_image=[image_pil], # Pipeline expects a list
        back_mask=back_mask,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
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
                )
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the edit (e.g. 'Make the banana red')")
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    seed = gr.Number(label="Seed", value=42)
                
                run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                result_main = gr.Image(label="Main Output")
                result_sub = gr.Image(label="Sub Image (Detail)")

    run_btn.click(
        fn=predict,
        inputs=[input_image, prompt, cfg_scale, steps, seed],
        outputs=[result_main, result_sub]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9999, share=True)
