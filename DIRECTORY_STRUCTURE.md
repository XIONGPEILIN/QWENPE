# Directory Structure & File Description

This document provides a comprehensive overview of the Qwen Image Edit workspace, including core framework code, training scripts, datasets, and experiment results.

## Core Framework & Code
*   **`DiffSynth-Studio/`**: The core diffusion model library. Contains pipeline definitions, loss functions, and model architectures.
*   **`ACE_plus/`**: Implementation and inference code for the ACE++ model.
*   **`.venv/`**: Main Python virtual environment (standardized to use `/export/ssd2/` physical paths for gp41/gp42 compatibility).

## Primary Scripts (Root Directory)
*   **`Qwen-Image-Edit-2511.sh`**: Main shell script for launching training jobs.
*   **`qwen_cli_single_multi_turn.py`**: High-performance multi-GPU inference script for Qwen models.
*   **`run_parallel.sh`**: Multi-GPU wrapper for running parallel test batches.
*   **`evaluate_metrics.py`**: The core evaluation engine (calculates SigLIP2, DINOv3, DreamSim, MSE/MAE).
*   **`batch_eval_cfgs.py`**: Orchestrates evaluation across multiple CFG result folders.
*   **`Qwen-Image-Test.py`**: Standard inference/testing script for Qwen models (with STE).
*   **`Qwen-Image-Test-woste.py`**: Inference script modified for the **without STE** ablation study.

## Consolidated Experiment Results (Top-1000 Dataset)
These directories contain generated images and evaluation metrics for models tested on the same 1000-sample high-quality dataset.

*   **`final_comparison_results/`**: **[Consolidated]** Final summary JSONs and full CSV reports for all 5 compared models (Qwen Base, Qwen Ablation, Flux, ACE++, MagicBrush).
*   **`qwen_results_top1000/`**: Inference outputs and metrics for the **Qwen-Image-Edit (Baseline)**.
*   **`flux_results_top1000/`**: Inference outputs and metrics for **Flux.1-Fill-dev**.
*   **`ace_plus_results_top1000/`**: Inference outputs and metrics for **ACE++**.
*   **`qwen_results_woste_top1000_pixelmask_cfg4/`**: Inference outputs for the **Qwen Ablation (without STE)** using Pixel Mask fusion and CFG 4.0.
*   **`eval_results_woste_20000/`**: Earlier evaluation results for MagicBrush and CFG sweeps.
*   **`eval_results_woste_20000_batch/`**: Backup of batch evaluation results for various ablation CFG scales.

## Training & Data
*   **`train/`**: Contains LoRA checkpoints and logs.
    *   `Qwen-Image-Edit-2511_lora-rank512-cfg-wo_ste_subloss/`: Checkpoints for the STE-less ablation study.
*   **`data/`**: Training data caches.
    *   `Qwen-Image-Edit-2511_lora-rank512-split-cache/`: Pre-computed VAE/Text-encoder latents.
*   **`pico-banana-400k-subject_driven/openimages/`**: The base image repository.
*   **`dataset_qwen_pe_top1000.json`**: The primary 1000-sample evaluation dataset metadata.
*   **`pe_masks/`**: Ground truth masks for the Qwen-PE evaluation set.

## Support Directories
*   **`scripts_data_prep/`**: Scripts for generating, cleaning, and verifying dataset JSONs.
*   **`scripts_utils/`**: General utilities, legacy scripts (e.g., 2509 model), and connectivity tests.
*   **`apps_demo/`**: Gradio-based web applications for interactive model testing.
*   **`notebooks_viz/`**: Jupyter notebooks for data exploration and result visualization.
*   **`quality_analysis_logs/`**: Logs and scripts from ghosting/seam artifact analysis.
*   **`temp_assets/`**: Temporary images and artifacts.