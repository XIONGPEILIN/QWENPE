from pathlib import Path
from typing import List, Tuple
import json
import random
import sys
import argparse
import os

from PIL import Image
import torch
from safetensors.torch import load_file

REPO_ROOT_SENTINEL = "dataset_qwen_pe_top1000.json"


def find_repo_root(start: Path) -> Path:
    """Walk upwards until dataset_qwen_pe_top1000.json is found."""
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


def load_lbm_checkpoint(pipe, ckpt_path: Path):
    """
    Load both LoRA and STE weights from LBM checkpoint.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    # Load to CPU first
    state = load_file(str(ckpt_path), device="cpu")

    # 1. Load STE weights (prefixed with pipe.ste. in training)
    ste_prefix = "pipe.ste."
    ste_state = {k[len(ste_prefix):]: v.to(device=pipe.device, dtype=pipe.torch_dtype) 
                 for k, v in state.items() if k.startswith(ste_prefix)}
    if ste_state:
        pipe.ste.load_state_dict(ste_state, strict=False)
        print(f"Successfully loaded {len(ste_state)} STE keys.")

    # 2. Load LoRA weights
    lora_state = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) 
                  for k, v in state.items() if "lora_" in k}
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state)
        print(f"Successfully loaded {len(lora_state)} LoRA keys.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=0, help="Index of this worker")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers")
    args = parser.parse_args()

    worker_id = args.worker_id
    num_workers = args.num_workers

    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.append(str(repo_root / "DiffSynth-Studio"))

    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    # Load dataset
    dataset_path = repo_root / REPO_ROOT_SENTINEL
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Random selection with fixed seed
    random.seed(666)
    selected_samples = random.sample(dataset, min(24, len(dataset)))
    
    indexed_samples = list(enumerate(selected_samples, 1))
    my_samples = indexed_samples[worker_id::num_workers]

    print(f"[Worker {worker_id}/{num_workers}] Starting. Assigned {len(my_samples)} samples.")

    if not my_samples:
        print(f"[Worker {worker_id}] No samples to process. Exiting.")
        return

    device = "cuda"
    
    # VRAM config
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }

    try:
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ],
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
            vram_limit=torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) - 2,
        )
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to load pipeline: {e}")
        return

    # Updated to LBM checkpoint
    ckpt_path = repo_root / "train/Qwen-Image-Edit-LBM_lora-rank512/step-10000.safetensors"
    ckpt_name = "LBM-step10000"

    print(f"[Worker {worker_id}] Loading LBM checkpoint: {ckpt_path}")
    try:
        load_lbm_checkpoint(pipe, ckpt_path)
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading checkpoint: {e}")
        return

    image_folder = "pico-banana-400k-subject_driven/openimages"

    for global_idx, sample in my_samples:
        print(f"[Worker {worker_id}] Processing sample {global_idx} (Checkpint: {ckpt_name})")
        
        try:
            image_path = repo_root / image_folder / sample["image"]
            edit_paths: List[Path] = [repo_root / image_folder / p for p in sample["edit_image"]]
            back_mask_path = repo_root / image_folder / sample["back_mask"]

            target_image = load_rgb(image_path)
            edit_images = [load_rgb(p) for p in edit_paths]
            back_mask = load_mask(back_mask_path)

            width, height = target_image.size

            for cfg in [1.0, 4.0]:
                print(f"[Worker {worker_id}] Processing sample {global_idx} (CFG: {cfg})")

                # Create sample directory
                sample_dir = repo_root / "compare" / f"{ckpt_name}_cfg{int(cfg)}" / str(global_idx)
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Save debug images and sample JSON
                with open(sample_dir / "sample.json", "w", encoding="utf-8") as f:
                    json.dump(sample, f, indent=4, ensure_ascii=False)
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
                    cfg_scale=cfg,
                    seed=42,
                    inpaint_blend_alpha=0.1,
                    use_bbox_mask=False,
                )

                image.save(sample_dir / "output.png")
                if sub_image is not None:
                    sub_image.save(sample_dir / "output_sub.png")

        except Exception as e:
            print(f"[Worker {worker_id}] Error processing sample {global_idx}: {e}")

    print(f"[Worker {worker_id}] Done.")


if __name__ == "__main__":
    main()
