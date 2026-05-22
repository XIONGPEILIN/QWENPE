import os
import sys
from pathlib import Path
import torch
import gradio as gr
from PIL import Image
from safetensors.torch import load_file

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# -----------------------------------------------------------------------------
# 2. 模型加载逻辑
# -----------------------------------------------------------------------------
def load_lora_only(pipe, ckpt_path: Path):
    print(f"Loading LoRA weights from {ckpt_path}...")
    state = load_file(str(ckpt_path), device="cpu")

    # Load LoRA
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        print(f"  - Loading {len(lora_state)} LoRA tensors")
        # Move LoRA weights to the correct device and dtype
        lora_state = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in lora_state.items()}
        pipe.load_lora(pipe.dit, state_dict=lora_state)
    
    print("LoRA loaded successfully.")

print("Initializing Pipeline...")
# 使用指定的模型配置
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:1",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# lora_path = "train/Qwen-Image-Edit-2511_lora-rank512-cfg/step-28000.safetensors"
# load_lora_only(pipe, lora_path)

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
def predict(input_image, prompt, cfg_scale, steps, seed, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        return None
    
    image_pil = input_image.convert("RGB")
    image_pil, inter_width, inter_height, target_width, target_height = preprocess_image(image_pil)
    
    print(f"Processing: Prompt='{prompt}', Final Size={target_width}x{target_height}, Seed={seed}")

    output_image = pipe(
        prompt=prompt,
        input_image=image_pil, # Standard image input for diffsynth
        height=target_height,
        width=target_width,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
    )
    
    # 裁剪掉 Padding 部分
    output_image = output_image.crop((0, 0, inter_width, inter_height))
    
    return output_image

# -----------------------------------------------------------------------------
# 4. Gradio 界面
# -----------------------------------------------------------------------------
css = """
#col-container {max-width: 1000px; margin-left: auto; margin-right: auto;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Qwen Image Generation (LoRA Only)")
        gr.Markdown("上传背景图并输入 Prompt。")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Background Image", type="pil")
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the generation (e.g. 'Make the banana red')")
                
                with gr.Accordion("Advanced Settings", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=10.0, value=2.0, step=0.1)
                    seed = gr.Number(label="Seed", value=0)
                
                run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                result_main = gr.Image(label="Main Output")
                send_back_btn = gr.Button("Send to Input (继续修改)", variant="secondary")

    def send_back(image):
        return image

    run_btn.click(
        fn=predict,
        inputs=[input_image, prompt, cfg_scale, steps, seed],
        outputs=[result_main]
    )

    send_back_btn.click(
        fn=send_back,
        inputs=[result_main],
        outputs=[input_image]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7999, share=True)

