#!/bin/bash


# Define Evaluation Script
EVAL_SCRIPT="evaluate_siglip2_qwen.py"

# List of directories to evaluate
DIRS=(
    "pico_test/ace_plus_results_top1000_adaptive"
    "pico_test/flux_results_top1000"
    "pico_test/qwen_results_top1000"
    "pico_test/qwen_results_noste_30k_top1000"
    "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4"
    "pico_test/ace_plus_results_top1000"
    "pico_test/magicbrush_results_top1000"
)

echo "Starting Batch Evaluation for SigLIP2 (Text Alignment)..."
echo "Found ${#DIRS[@]} directories to process."

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "----------------------------------------------------------------"
        echo "Processing: $dir"
        echo "----------------------------------------------------------------"
        
        # Run evaluation using all GPUs (0-7)
        python "$EVAL_SCRIPT" \
            --pred_dir "$dir" \
            --json_path "dataset_qwen_pe_top1000_captioned.json" \
            --gpu_ids "0,1,2,3,4,5,6,7"
            
    else
        echo "[WARNING] Directory not found: $dir"
    fi
done

echo "----------------------------------------------------------------"
echo "All evaluations complete!"
echo "----------------------------------------------------------------"
