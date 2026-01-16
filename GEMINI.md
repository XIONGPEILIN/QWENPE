# 中文回答所有问题
# Qwen-Image-Edit Project Context

## Project Overview
This project focuses on the development, training, and evaluation of **Qwen-Image-Edit** models, leveraging the **DiffSynth-Studio** framework. The primary goal is to improve image editing capabilities (e.g., object replacement, removal) using Parameter-Efficient Fine-Tuning (PEFT/LoRA) and investigating architectural decisions like Spatio-Temporal Encodings (STE).

## Directory Structure & Key Components

### Core Code
*   **`DiffSynth-Studio/`**: The underlying diffusion model library. Contains the core logic for model loading, pipelines, and training loops.
*   **`ACE_plus/`**: Comparison model (ACE++) implementation.
*   **`apps_demo/`**: Gradio-based web UIs for interactive demonstrations (`app_gradio.py`).

### Scripts & Workflows
*   **Training (`Qwen-Image-Edit-*.sh`)**: Shell scripts driving the training process. Training is typically split into two stages:
    1.  **Data Caching**: Pre-processing images/text into latents (stored in `data/`).
    2.  **Training**: Fine-tuning the DiT model (stored in `train/`) using the cached data.
*   **Inference (`qwen_cli_single_multi_turn.py`, `Qwen-Image-Test.py`)**: Scripts for running model inference. `qwen_cli_single_multi_turn.py` supports multi-turn interactions.
*   **Evaluation (`evaluate_metrics.py`, `batch_eval_cfgs.py`)**: Tools for computing metrics like SigLIP, DINO, DreamSim, L1/L2.
*   **Automation (`auto_launch_training.sh`)**: A utility to monitor GPU usage and automatically launch training jobs when resources are available.

### Data
*   **`dataset_*.json`**: Dataset definitions.
    *   **Schema**:
        ```json
        {
          "prompt": "Instruction text...",
          "image": "path/to/source.png",
          "edit_image": ["path/to/target.png"],
          "ref_gt": "path/to/ground_truth.png",
          "back_mask": "path/to/mask.png"
        }
        ```
*   **`data/`**: Storage for pre-computed training caches (latents).
*   **`pico-banana-400k-subject_driven/`**: Raw image data repository.
*   **`final_comparison_results/`**: Aggregated results comparing Qwen against Flux, ACE++, etc.

## Usage Guide

### 1. Environment
*   **Virtual Env**: Located in `.venv/`. Ensure it is activated.
*   **Configuration**: `accelerate_config.yaml` manages distributed training settings.

### 2. Running Training
Training is launched via shell scripts that wrap `accelerate launch`.
Example (`Qwen-Image-Edit-2511.sh`):
```bash
# Set WANDB variables
export WANDB_PROJECT="qwen-image"
export WANDB_NAME="My-Experiment"

# Execute the script (handles caching and training)
bash Qwen-Image-Edit-2511.sh
```
*Note: Check `auto_launch_training.sh` for automated scheduling on busy nodes.*

### 3. Inference & Testing
To run inference on the top 1000 dataset:
```bash
python Qwen-Image-Test.py \
    --model_path "path/to/checkpoint" \
    --output_dir "my_results/"
```

### 4. Evaluation
To evaluate generated results against ground truth:
```bash
python evaluate_metrics.py \
    --pred_dir "my_results/" \
    --gt_dir "path/to/ground_truths/" \
    --json_path "dataset_qwen_pe_top1000.json"
```

## Development Conventions
*   **Paths**: Scripts often use absolute paths (e.g., `/export/ssd2/...`). Be careful when moving code between environments.
*   **DiffSynth-Studio**: This is a critical dependency. Changes to core modeling logic often happen inside this submodule.
*   **Results**: Experiment outputs are structured by model and configuration (e.g., `qwen_results_top1000`, `flux_results_top1000`).
