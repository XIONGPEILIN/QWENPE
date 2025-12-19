import os
import sys
from pathlib import Path
import torch
import gradio as gr
from PIL import Image

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

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput

# -----------------------------------------------------------------------------
# 2. 模型加载逻辑 (使用 cuda:2 + ControlNet)
# -----------------------------------------------------------------------------
device = "cuda:2"
print(f"Initializing ControlNet Pipeline on {device}...")

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# -----------------------------------------------------------------------------
# 3. 推理函数
# -----------------------------------------------------------------------------
def predict(input_dict, prompt, cfg_scale, steps, seed):
    if input_dict is None:
        return None
    
    image_pil = input_dict["background"].convert("RGB")
    orig_width, orig_height = image_pil.size
    
    # 1. 等比例缩放以符合 max_pixels
    max_pixels = 1048576
    curr_pixels = orig_width * orig_height
    if curr_pixels > max_pixels:
        factor = (max_pixels / curr_pixels) ** 0.5
        inter_width = int(orig_width * factor)
        inter_height = int(orig_height * factor)
        image_pil = image_pil.resize((inter_width, inter_height), Image.LANCZOS)
    else:
        inter_width, inter_height = orig_width, orig_height

    # 2. 计算 16 的倍数
    target_width = ((inter_width + 15) // 16) * 16
    target_height = ((inter_height + 15) // 16) * 16
    
    # 3. Padding 原图
    if target_width != inter_width or target_height != inter_height:
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image
    
    # 4. 处理 Mask
    mask_layer = input_dict["layers"][0]
    if mask_layer:
        alpha = mask_layer.split()[-1]
        if curr_pixels > max_pixels:
            alpha = alpha.resize((inter_width, inter_height), Image.NEAREST)
        raw_mask = Image.eval(alpha, lambda a: 255 if a > 0 else 0)
    else:
        raw_mask = Image.new("L", (inter_width, inter_height), 0)

    if target_width != inter_width or target_height != inter_height:
        inpaint_mask = Image.new("L", (target_width, target_height), 0)
        inpaint_mask.paste(raw_mask, (0, 0))
    else:
        inpaint_mask = raw_mask

    inpaint_mask_rgb = inpaint_mask.convert("RGB")
    width, height = image_pil.size
    print(f"Processing: Prompt='{prompt}', Size={width}x{height}, Device={device}")

    # ControlNet Inpaint 仅返回单张图片
    output_image = pipe(
        prompt=prompt, 
        seed=int(seed),
        input_image=image_pil, 
        inpaint_mask=inpaint_mask_rgb,
        blockwise_controlnet_inputs=[ControlNetInput(image=image_pil, inpaint_mask=inpaint_mask_rgb)],
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        edit_image=image_pil, 
        height=height,
        width=width,
    )
    
    return output_image

# -----------------------------------------------------------------------------
# 4. Gradio 界面
# -----------------------------------------------------------------------------
css = "#col-container {max-width: 1000px; margin-left: auto; margin-right: auto;}"

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"# Qwen Image ControlNet Inpaint ({device})")
        gr.Markdown("上传图片并涂抹想要 **重绘** 的区域 (Inpaint Mask)。")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.ImageEditor(
                    label="Input Image & Inpaint Mask",
                    type="pil",
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"), 
                    eraser=gr.Eraser(),
                    layers=False, 
                )
                prompt = gr.Textbox(label="Prompt", placeholder="Describe what to fill in")
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=40, step=1)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    seed = gr.Number(label="Seed", value=0)
                
                run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                result_image = gr.Image(label="Output Image")

    run_btn.click(
        fn=predict,
        inputs=[input_image, prompt, cfg_scale, steps, seed],
        outputs=[result_image]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, share=True)