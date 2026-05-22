from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
import torch


REPO_ROOT_SENTINEL = "dataset_qwen_pe_top1000.json"


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / REPO_ROOT_SENTINEL).exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate {REPO_ROOT_SENTINEL} upwards from {start}")


def load_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    # QwenImageVAE expects 3-channel image input.
    return Image.open(path).convert("RGB")


def align_image_sizes(image_a: Image.Image, image_b: Image.Image) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    orig_a = image_a.size
    orig_b = image_b.size
    if orig_a == orig_b:
        return image_a, image_b, orig_a, orig_b

    # Keep A as reference and resize B to match, so latent tensors can be averaged.
    image_b = image_b.resize(orig_a, Image.Resampling.LANCZOS)
    return image_a, image_b, orig_a, orig_b


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE latent interpolation for two images (Qwen VAE only)")
    parser.add_argument(
        "--image_a",
        type=Path,
        default=Path("transparency_increase_image.png"),
    )
    parser.add_argument(
        "--image_b",
        type=Path,
        default=Path("generated_image.png"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/yanai-lab/xiong-p/qwen/vae_latent_avg_outputs"),
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.append(str(repo_root / "DiffSynth-Studio"))

    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.qwen_image import QwenImagePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    if pipe.vae is None:
        raise RuntimeError("Failed to load VAE: pipe.vae is None.")

    image_a = load_rgb_image(args.image_a)
    image_b = load_rgb_image(args.image_b)
    image_a, image_b, original_size_a, original_size_b = align_image_sizes(image_a, image_b)

    if original_size_a != original_size_b:
        print(f"[Info] size mismatch detected: A={original_size_a}, B={original_size_b}; resized B to {image_a.size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratios = [i / 10 for i in range(1, 10)]
    generated_paths: list[Path] = []

    with torch.no_grad():
        tensor_a = pipe.preprocess_image(image_a).to(device=device, dtype=torch_dtype)
        tensor_b = pipe.preprocess_image(image_b).to(device=device, dtype=torch_dtype)

        latent_a = pipe.vae.encode(tensor_a)
        latent_b = pipe.vae.encode(tensor_b)

        for ratio in ratios:
            latent_mid = latent_a * ratio + latent_b * (1.0 - ratio)
            recon_mid = pipe.vae.decode(latent_mid)
            middle_image = pipe.vae_output_to_image(recon_mid)
            output_image_path = args.output_dir / f"middle_latent_mix_a{ratio:.1f}_b{1.0-ratio:.1f}.png"
            middle_image.save(output_image_path)
            generated_paths.append(output_image_path)

    output_meta_path = args.output_dir / "run_info.txt"

    with open(output_meta_path, "w", encoding="utf-8") as f:
        f.write(f"image_a={args.image_a}\n")
        f.write(f"image_b={args.image_b}\n")
        f.write(f"original_size_a={original_size_a}\n")
        f.write(f"original_size_b={original_size_b}\n")
        f.write(f"size={image_a.size}\n")
        f.write(f"device={device}\n")
        f.write(f"torch_dtype={torch_dtype}\n")
        f.write(f"model_id={args.model_id}\n")
        f.write(f"ratios={','.join(f'{r:.1f}' for r in ratios)}\n")
        for p in generated_paths:
            f.write(f"generated={p}\n")

    for p in generated_paths:
        print(f"Saved image: {p}")
    print(f"Saved meta: {output_meta_path}")


if __name__ == "__main__":
    main()
