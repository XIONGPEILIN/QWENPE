# Qwen Image Edit (DiffSynth-Studio) Project

## Project Overview

This project is a workspace focused on training and deploying image editing models, primarily **Qwen-Image-Edit-2511** (current focus) and its predecessor 2509, utilizing the **DiffSynth-Studio** framework. It includes scripts for data preparation, model training (using LoRA and Split Training), and interactive inference via a Gradio Web UI.

**Core Technologies:**
*   **DiffSynth-Studio:** The underlying diffusion model engine (located in `DiffSynth-Studio/`).
*   **Qwen-Image-Edit (2511):** The primary model architecture being fine-tuned. 2511 is the latest version in use.
*   **PyTorch & Accelerate:** For model training and inference.
*   **Gradio:** For the web-based demonstration interface.

## Directory Structure

*   `DiffSynth-Studio/`: The core library source code (likely a submodule or clone).
*   `app_gradio.py`: Main entry point for the interactive Web UI.
*   `Qwen-Image-Edit-2511.sh`: Primary training script for the 2511 model.
*   `Qwen-Image-Edit-2509.sh`: Training script for the 2509 model.
*   `generate_dataset_*.py`: Scripts for dataset preparation and JSON generation.
*   `*.json`: Dataset metadata files (e.g., `dataset_qwen_pe_train_crop.json`).
*   `train/`: Directory storing training checkpoints (LoRA weights).
*   `data/`: Directory for dataset images and caching.

## Usage Guide

### 1. Running the Web UI (Inference)

To launch the interactive image editing interface:

```bash
python app_gradio.py
```

*   **Port:** 7999 (by default).
*   **Functionality:** Allows uploading an image, drawing a mask, providing a prompt, and generating an edited version.
*   **Dependencies:** Requires `DiffSynth-Studio` in the python path (handled by the script) and specific LoRA checkpoints.

### 2. Training (Focus: 2511)

Training is typically handled via shell scripts that utilize `accelerate`. The current priority is **2511**.

**Example: `Qwen-Image-Edit-2511.sh`**

This script uses a split training process:
1.  **Phase 1 (Data Processing):** Pre-computes text embeddings and VAE latents.
2.  **Phase 2 (Training):** Trains the DiT with specific requirements for 2511.

**Crucial for 2511:**
*   **`--zero_cond_t` flag:** Mandatory for 2511 model training.
*   **Expanded LoRA Modules:** Includes `img_in`, `txt_in`, and `proj_out`.
*   **Gradient Accumulation:** Typically set to 16.

```bash
bash Qwen-Image-Edit-2511.sh
```

**Key Environment Variables:**
*   `WANDB_PROJECT`: WandB project name (default: `qwen-image`).
*   `WANDB_NAME`: WandB run name.

### 3. Data Preparation

Dataset metadata is stored in JSON files containing lists of entries with:
*   `prompt`: Text description of the edit.
*   `image`: Target image path.
*   `edit_image`: Source/Input image path (list).
*   `ref_gt`: Reference ground truth.
*   `back_mask`: Background mask path.

Scripts like `generate_dataset_json_mp.py` are used to generate these JSON files from raw image directories.

### 4. LBM (Latent Bridge Matching) Training

LBM is a new training paradigm implemented in this project to speed up and improve image-to-image translation tasks.

**Core Mechanisms:**
*   **Bridge Paths:**
    *   **Main Stream**: Interpolates from source image latents ($x_0$) to target image latents ($x_1$).
    *   **Sub Stream**: Interpolates from **pure white image latents** to target image latents.
*   **Start Logic**: The `sub_noise` key in cache files stores pre-computed white image latents encoded via VAE for maximum precision.
*   **Loss Function**: `LatentBridgeMatchingLoss` in `diffsynth/diffusion/loss.py`.

**Training Constraints:**
*   **Optimizer**: Uses `ProdigyPlusScheduleFree`.
*   **No Gradient Clipping**: **DO NOT** enable `gradient_clipping` in DeepSpeed/Accelerate config as it conflicts with the optimizer's logic.
*   **ZeRO-3 Configuration**: Optimized for 96GB VRAM GPUs (e.g., RTX 6000 Blackwell).
    *   `offload_param_device: none` and `offload_optimizer_device: none` to keep everything on GPU for speed.
    *   Large communication buckets (`5e8`) and `overlap_comm: true` for throughput.

**Launch Command:**
```bash
bash Qwen-Image-Edit-LBM.sh
```

## Development Conventions

*   **Path Resolution:** Scripts often dynamically add `DiffSynth-Studio` to `sys.path`. Ensure the directory structure remains consistent.
*   **Model Configs:** The project uses `ModelConfig` objects from `diffsynth` to define model paths and patterns.
*   **LoRA Loading:** Custom logic exists (e.g., in `app_gradio.py`) to load STE (Spatial Temporal Encoder) and LoRA weights specifically for Qwen-Image-Edit.
