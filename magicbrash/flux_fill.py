import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageChops
from diffusers import FluxFillPipeline

MAX_SEED = np.iinfo(np.int32).max

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

def main():
    parser = argparse.ArgumentParser(description="Run Flux Fill on local images.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the mask image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output", type=str, default="output_flux.png", help="Path to save the result")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    print(f"Loading model to {args.device}...")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(args.device)

    print(f"Loading images...")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    image = Image.open(args.image).convert("RGB")
    input_mask = Image.open(args.mask)

    # Calculate dimensions
    width, height = calculate_optimal_dimensions(image)
    print(f"Resizing to {width}x{height}")
    image = image.resize((width, height), Image.LANCZOS)

    # Process Mask
    final_mask = Image.new("L", (width, height), 0)
    
    # Handle Mask logic (similar to original app)
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

    # Debug info
    fm_arr = np.array(final_mask)
    white_px = np.sum(fm_arr > 128)
    total_px = fm_arr.size
    print(f"Mask Stats - Size: {final_mask.size}, Edit Pixels: {white_px} ({white_px/total_px:.2%})")

    print("Running inference...")
    result_image = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=final_mask,
        height=height,
        width=width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=torch.Generator(args.device).manual_seed(args.seed)
    ).images[0]

    print(f"Saving result to {args.output}")
    result_image.save(args.output)

if __name__ == "__main__":
    main()
