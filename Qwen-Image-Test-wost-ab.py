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


def load_lora(pipe, ckpt_path: Path) -> int:
    """
    Load LoRA tensors into DiT from a safetensors file.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    # Load to CPU first
    state = load_file(str(ckpt_path), device="cpu")

    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        # Move LoRA weights to the correct device and dtype before loading
        lora_state = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in lora_state.items()}
        pipe.load_lora(pipe.dit, state_dict=lora_state)

    return len(lora_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True, help="Index of this worker (0 to num_workers-1)")
    parser.add_argument("--num_workers", type=int, required=True, help="Total number of workers")
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
    
    # Dynamic Load Balancing: Shuffle work list uniquely per worker
    work_list = list(enumerate(selected_samples, 1))
    random.seed(worker_id)
    random.shuffle(work_list)
    
    lock_dir = repo_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Worker {worker_id}/{num_workers}] Starting dynamic processing. Pool size: {len(work_list)}")

    if not work_list:
        print(f"[Worker {worker_id}] No samples. Exiting.")
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
        # pipe = QwenImagePipeline.from_pretrained(
        #     torch_dtype=torch.bfloat16,
        #     device=device,
        #     model_configs=[
        #         ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        #         ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        #         ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        #     ],
        #     processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        #     vram_limit=torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 2,
        # )
        pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
            )
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to load pipeline: {e}")
        return

    checkpoints = [
        {
            "path": repo_root / "train/Qwen-Image-Edit-2511_lora-rank512-cfg-wo_ste_subloss/step-20000.safetensors",
            "name": "woste-20000"
        },
    ]

    image_folder = "pico-banana-400k-subject_driven/openimages"

    for ckpt_info in checkpoints:
        ckpt_path = ckpt_info["path"]
        ckpt_name = ckpt_info["name"]

        print(f"[Worker {worker_id}] Loading checkpoint: {ckpt_name}")
        try:
            load_lora(pipe, ckpt_path)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading checkpoint {ckpt_name}: {e}")
            continue

        for global_idx, sample in work_list:
            # Dynamic Locking
            lock_file = lock_dir / f"{ckpt_name}_{global_idx}.lock"
            if lock_file.exists():
                continue
            try:
                # Atomic claim
                with open(lock_file, "x") as f:
                    f.write(str(worker_id))
            except FileExistsError:
                continue

            print(f"[Worker {worker_id}] Claimed sample {global_idx} (Checkpint: {ckpt_name})")
            
            try:
                image_path = repo_root / image_folder / sample["image"]
                edit_paths: List[Path] = [repo_root / image_folder / p for p in sample["edit_image"]]
                back_mask_path = repo_root / image_folder / sample["back_mask"]

                target_image = load_rgb(image_path)
                edit_images = [load_rgb(p) for p in edit_paths]
                back_mask = load_mask(back_mask_path)

                width, height = target_image.size

                cfgs = [2.0]
                alphas = [0.1]

                ablation_configs = []
                for no_sub in [False, True]:
                    for mask_sub in [False, True]:
                        # Test both True and False for mask_sub_self when mask_sub is True.
                        # If mask_sub is False, mask_sub_self is typically irrelevant, so we keep it False.
                        sub_self_options = [False, True] if mask_sub else [False]
                        for mask_sub_self in sub_self_options:
                            name = f"noSub{no_sub}_maskSub{mask_sub}_maskSelf{mask_sub_self}"
                            if not no_sub and not mask_sub and not mask_sub_self:
                                name = "baseline"
                            ablation_configs.append({
                                "name": name,
                                "no_sub": no_sub,
                                "mask_sub": mask_sub,
                                "mask_sub_self": mask_sub_self
                            })

                for cfg in cfgs:
                    for alpha in alphas:
                        for ab_cfg in ablation_configs:
                            ab_name = ab_cfg["name"]
                            ab_no_sub = ab_cfg["no_sub"]
                            ab_mask = ab_cfg["mask_sub"]
                            ab_mask_self = ab_cfg["mask_sub_self"]

                            print(f"[Worker {worker_id}] Processing sample {global_idx} (CFG: {cfg}, Alpha: {alpha}, Ablation: {ab_name})")

                            # Create sample directory per Grid Point
                            sample_dir = repo_root / "compare" / "qwen_results_woste_ablation" / f"{ckpt_name}_cfg{int(cfg)}_alpha{alpha}_{ab_name}" / str(global_idx)
                            
                            if (sample_dir / "output.png").exists():
                                print(f"[Worker {worker_id}] Skipping already completed: {sample_dir}")
                                continue

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
                                inpaint_blend_alpha=alpha,
                                use_bbox_mask=True,
                                ablation_token_gate=1,
                                ablation_no_sub_noise=ab_no_sub,
                                ablation_mask_sub=ab_mask,
                                ablation_mask_sub_self=ab_mask_self,
                            )

                            image.save(sample_dir / "output.png")
                            if sub_image is not None:
                                sub_image.save(sample_dir / "output_sub.png")

            except Exception as e:
                print(f"[Worker {worker_id}] Error processing sample {global_idx}: {e}")

    print(f"[Worker {worker_id}] Done.")


if __name__ == "__main__":
    main()
