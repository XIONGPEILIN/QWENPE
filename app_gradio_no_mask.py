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

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# -----------------------------------------------------------------------------
# 2. 模型加载逻辑 (使用 cuda:1)
# -----------------------------------------------------------------------------
device = "cuda:1"
print(f"Initializing Base Pipeline on {device} (No Mask)...")

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# -----------------------------------------------------------------------------
# 3. 推理函数
# -----------------------------------------------------------------------------
def predict(image_pil, prompt, cfg_scale, steps, seed):
    if image_pil is None:
        return None
    
    image_pil = image_pil.convert("RGB")
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
    
    # 3. 执行 Padding
    if target_width != inter_width or target_height != inter_height:
        new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        new_image.paste(image_pil, (0, 0))
        image_pil = new_image

    width, height = image_pil.size
    print(f"Processing: Prompt='{prompt}', Final Size={width}x{height}, Device={device}")

    # 此模型版本仅返回单张图片
    output_image = pipe(
        prompt=prompt,
        edit_image=[image_pil], 
        height=height,
        width=width,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
    )
    
    return output_image

# -----------------------------------------------------------------------------
# 4. Gradio 界面
# -----------------------------------------------------------------------------
css = "#col-container {max-width: 1000px; margin-left: auto; margin-right: auto;}"

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"# Qwen Image Edit (Base Model - {device} - No Mask)")
        gr.Markdown("上传图片并输入指令进行全局编辑。")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                prompt = gr.Textbox(label="Prompt", placeholder="e.g. 'Turn the image into a pencil sketch'")
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    seed = gr.Number(label="Seed", value=42)
                
                run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                result_image = gr.Image(label="Output Image")

    run_btn.click(
        fn=predict,
        inputs=[input_image, prompt, cfg_scale, steps, seed],
        outputs=[result_image]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)