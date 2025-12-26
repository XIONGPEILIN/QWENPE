import gradio as gr
import numpy as np
import torch
import random 
from diffusers import FluxFillPipeline
from PIL import Image, ImageChops


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# 加载模型
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

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


def infer(edit_images, input_mask, prompt, seed=42, randomize_seed=False, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if edit_images is None or edit_images["background"] is None:
        return None, seed

    # DEBUG: Save received mask
    if input_mask:
        print(f"DEBUG: Received input_mask mode: {input_mask.mode}")
        input_mask.save("debug_received_mask.png")

    # 1. 准备底图
    image = edit_images["background"].convert("RGB")
    width, height = calculate_optimal_dimensions(image)
    image = image.resize((width, height), Image.LANCZOS)

    # 2. 准备 Mask
    final_mask = Image.new("L", (width, height), 0)
    
    # A. 来自上传的 Mask 图片
    if input_mask is not None:
        if input_mask.mode in ('RGBA', 'LA') or (input_mask.mode == 'P' and 'transparency' in input_mask.info):
            alpha = input_mask.convert('RGBA').split()[-1]
            if alpha.getextrema() != (255, 255):
                # 逻辑反转：Alpha < 255 (透明/半透明) -> Mask=255 (修改)
                #           Alpha = 255 (不透明) -> Mask=0 (不修改)
                m = alpha.point(lambda p: 255 if p < 255 else 0).resize((width, height), Image.NEAREST)
                final_mask = ImageChops.lighter(final_mask, m)
            else:
                m = input_mask.convert("L").resize((width, height), Image.NEAREST)
                final_mask = ImageChops.lighter(final_mask, m)
        else:
            m = input_mask.convert("L").resize((width, height), Image.NEAREST)
            final_mask = ImageChops.lighter(final_mask, m)

    # B. 来自手绘图层
    if edit_images.get("layers") and len(edit_images["layers"]) > 0:
        layer_alpha = edit_images["layers"][0].split()[-1]
        m = layer_alpha.resize((width, height), Image.NEAREST)
        final_mask = ImageChops.lighter(final_mask, m)

    # DEBUG: Check final mask
    fm_arr = np.array(final_mask)
    white_px = np.sum(fm_arr > 128)
    total_px = fm_arr.size
    print(f"DEBUG: Final Mask Stats - Size: {final_mask.size}, White Pixels (Edit): {white_px} ({white_px/total_px:.2%})")
    if white_px == 0:
        print("WARNING: Final mask is all black! (No edit area)")
    if white_px == total_px:
        print("WARNING: Final mask is all white! (Edit everything)")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # 3. 运行推理
    result_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=final_mask,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    
    return result_image, seed

css="""
#col-container { margin: 0 auto; max-width: 1100px; }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# FLUX.1 Fill [dev] with Mask Upload Support")
        
        with gr.Row():
            with gr.Column():
                edit_image = gr.ImageEditor(
                    label='Upload Image & Draw Mask',
                    type='pil',
                    sources=["upload"],
                    image_mode='RGB',
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                    height=500
                )
                # 强制使用 RGBA 模式以保留 Alpha 通道
                input_mask = gr.Image(label="OR Upload Mask (Alpha or B&W)", type="pil", image_mode="RGBA")
                prompt = gr.Text(
                    label="Prompt",
                    placeholder="Describe what to fill in the masked area",
                )
                run_button = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                result = gr.Image(label="Result")
                
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, step=0.5, value=3.5)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=50)

    run_button.click(
        fn=infer,
        inputs=[edit_image, input_mask, prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
        outputs=[result, seed]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
