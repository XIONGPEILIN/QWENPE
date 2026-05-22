#!/bin/bash

set -e
set -o pipefail

# --- Configuration ---
# Python environment for the generation script
ACE_PLUS_VENV="/host/ssd2/xiong-p/qwenpe/ACE_plus/.venv/bin/python"

# Path to the newly created adapter script.
ACE_PLUS_SCRIPT_PATH="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ace_plus_icebench.py"

# Input JSON for generation
INPUT_JSON="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ICE-Bench/dataset/selected_5_tasks_flat_black_remove_qwen_inpaint.json"

# Base directory for all outputs of this script
RUN_OUTPUT_BASE_DIR="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ICE-Bench/results/ace_plus_magicbrush"

# Directory where generated images will be saved
GENERATION_OUTPUT_DIR="$RUN_OUTPUT_BASE_DIR/generated_images"

# Path to the dataset
# This is the base path for the image paths inside the JSON
DATASET_PATH="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ICE-Bench/dataset"

# Python environment for the evaluation script
ICE_BENCH_VENV="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ICE-Bench/.venv/bin/python"

# --- 1. Image Generation ---

echo "--- Starting Image Generation ---"

# Create output directory
mkdir -p "$GENERATION_OUTPUT_DIR"

echo "Running generation. If the script fails, please check the following command:"
echo "$ACE_PLUS_VENV $ACE_PLUS_SCRIPT_PATH --input_json $INPUT_JSON --output_dir $GENERATION_OUTPUT_DIR --data_prefix $DATASET_PATH"

if [ -f "$ACE_PLUS_SCRIPT_PATH" ]; then
    "$ACE_PLUS_VENV" "$ACE_PLUS_SCRIPT_PATH" \
        --input_json "$INPUT_JSON" \
        --output_dir "$GENERATION_OUTPUT_DIR" \
        --data_prefix "$DATASET_PATH" \
        --gpu_ids "0,1,2,3,4,5,6,7" # Using GPU 0 by default, can be modified
else
    echo "WARNING: Generation script '$ACE_PLUS_SCRIPT_PATH' not found."
    echo "Skipping generation. Assuming images are already in '$GENERATION_OUTPUT_DIR'."
fi

echo "--- Image Generation Finished ---"


# --- 2. Create gen_info.json for evaluation ---

echo "--- Creating gen_info.json ---"

GEN_INFO_PATH="$RUN_OUTPUT_BASE_DIR/gen_info.json"

"$ICE_BENCH_VENV" -c '
import json
import os

input_json_path = "'"$INPUT_JSON"'"
generation_output_dir = "'"$GENERATION_OUTPUT_DIR"'"
gen_info_path = "'"$GEN_INFO_PATH"'"

with open(input_json_path, "r") as f:
    data = json.load(f)

gen_info = {}
for item in data:
    item_id = item["item_id"]
    output_filename = f"{item_id}.png"
    image_path = os.path.join(generation_output_dir, output_filename)
    
    if os.path.exists(image_path):
        gen_info[item_id] = image_path
    else:
        print(f"Warning: Could not find generated image for item_id {item_id} at {image_path}")

with open(gen_info_path, "w") as f:
    json.dump(gen_info, f, indent=4)

print(f"gen_info.json created at {gen_info_path} with {len(gen_info)} entries.")
'

# --- 3. Run Evaluation ---

echo "--- Starting Evaluation ---"

EVAL_RESULT_PATH="$RUN_OUTPUT_BASE_DIR/eval_result.txt"
ICE_BENCH_DIR="/host/home/yanai-lab/Sotsuken24/xiong-p/qwen/ICE-Bench"

cd "$ICE_BENCH_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=5

"$ICE_BENCH_VENV" eval.py \
  -m "dataset/selected_5_tasks_flat_eval_original.jsonl" \
  -f "$GEN_INFO_PATH" \
  -s "$EVAL_RESULT_PATH"

echo "--- Evaluation Finished ---"


# --- 4. Calculate Scores ---

echo "--- Calculating Scores ---"

"$ICE_BENCH_VENV" cal_scores.py \
  -f "$EVAL_RESULT_PATH"

echo "--- Score Calculation Finished ---"
echo "All done. Results are in $RUN_OUTPUT_BASE_DIR"
