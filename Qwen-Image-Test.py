from pathlib import Path
from typing import List, Tuple
import json
import random

from PIL import Image
import torch
from safetensors.torch import load_file

REPO_ROOT_SENTINEL = "dataset_qwen_pe.json"


def find_repo_root(start: Path) -> Path:
    """Walk upwards until dataset_qwen_pe.json is found."""
    cur = start
    for _ in range(10):
        if (cur / REPO_ROOT_SENTINEL).exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate {REPO_ROOT_SENTINEL} upwards from {start}")


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGB")


def load_mask(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing mask: {path}")
    return Image.open(path).convert("L")


def load_ste_and_lora(pipe, ckpt_path: Path) -> Tuple[int, int]:
    """
    Load STE weights (prefix pipe.ste.) and LoRA tensors into DiT from a safetensors file.
    Returns (num_ste_tensors, num_lora_tensors).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = load_file(str(ckpt_path), device="cpu")

    ste_prefix = "pipe.ste."
    ste_state = {k[len(ste_prefix):]: v for k, v in state.items() if k.startswith(ste_prefix)}
    if ste_state:
        pipe.ste.load_state_dict(ste_state, strict=False)

    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state)

    return len(ste_state), len(lora_state)


def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    import sys

    sys.path.append(str(repo_root / "DiffSynth-Studio"))

    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    # Load dataset and randomly select 10 samples
    dataset_path = repo_root / "dataset_qwen_pe_all.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    random.seed(811)
    selected_samples = random.sample(dataset, min(20, len(dataset)))
    
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    print(f"Testing {len(selected_samples)} randomly selected samples with seed=811")
    image_folder = "pico-banana-400k-subject_driven/openimages"

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:1",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    # Load STE + LoRA weights from training checkpoint
    ste_lora_ckpt = repo_root / "train/Qwen-Image-Edit-2509_lora-rank512/step-15000.safetensors"
    ste_num, lora_num = load_ste_and_lora(pipe, ste_lora_ckpt)
    print(f"Loaded {ste_num} STE tensors and {lora_num} LoRA tensors from {ste_lora_ckpt}\n")

    # Test each selected sample
    for idx, sample in enumerate(selected_samples, 1):
        print(f"Processing sample {idx}/10: {sample['image']}")
        
        try:
            image_path = repo_root / image_folder / sample["image"]
            edit_paths: List[Path] = [repo_root / image_folder / p for p in sample["edit_image"]]
            back_mask_path = repo_root / image_folder / sample["back_mask"]

            target_image = load_rgb(image_path)
            edit_images = [load_rgb(p) for p in edit_paths]
            back_mask = load_mask(back_mask_path)

            width, height = target_image.size

            # Create sample directory
            sample_dir = repo_root / "test-cfg4" / str(idx)
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save debug images
            edit_images[0].save(sample_dir / "debug_edit_image.png")
            target_image.save(sample_dir / "target_image.png")
            back_mask.save(sample_dir / "back_mask.png")

            image, sub_image = pipe(
                prompt=sample["prompt"],
                edit_image=edit_images,
                back_mask=back_mask,
                height=height,
                width=width,
                num_inference_steps=50,
                cfg_scale=4.0,
                seed=0,
                # pe_mask_dir=repo_root / "pe_masks",
            )

            # Save output images
            image.save(sample_dir / "output.png")
            sub_image.save(sample_dir / "output_sub.png")
            print(f"  ✓ Saved output to: {sample_dir}\n")
        
        except Exception as e:
            print(f"  ✗ Error processing sample {idx}: {e}\n")
            continue
    
    print("Completed testing all 10 samples")


if __name__ == "__main__":
    main()
