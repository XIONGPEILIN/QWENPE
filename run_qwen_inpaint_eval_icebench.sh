#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=5


cd /home/yanai-lab/xiong-p/qwen/ICE-Bench

WORKSPACE="/home/yanai-lab/xiong-p/qwen"
PYTHON="/host/ssd2/xiong-p/diffsy/.venv/bin/python"

METHOD="qwen_inpaint_icebench"
RESULTS_DIR="results/${METHOD}"
IMAGE_OUT_DIR="${RESULTS_DIR}/images"
GEN_INFO_JSON="${RESULTS_DIR}/gen_info.json"

JSON_FILE="dataset/selected_5_tasks_flat.json"
INPUT_DIR="dataset" # Images are at dataset/data/images/...

GPU_IDS="0,1,2,3,4,5,6,7"

echo "========================================================"
echo " [Step 1] Running Qwen-Image-Edit for ICE-Bench"
echo "========================================================"

mkdir -p "${IMAGE_OUT_DIR}"

# CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
#     "${PYTHON}" "${WORKSPACE}/qwen_inpaint_magic_icebench.py" \
#     --input_path "${INPUT_DIR}" \
#     --json_path "${JSON_FILE}" \
#     --output_path "${IMAGE_OUT_DIR}" \
#     --gen_info_path "${GEN_INFO_JSON}" \
#     --gpu_ids "${GPU_IDS}" \
#     --num_gpus 8 \
#     --steps 50 \
#     --target_size 1024

echo "========================================================"
echo " [Step 2] Run ICE-Bench evaluation"
echo "========================================================"
# ICE-Bench virtual environment
ICE_PYTHON=".venv/bin/python"

${ICE_PYTHON} eval.py \
  -m dataset/selected_5_tasks_flat_eval_original.jsonl \
  -f "${GEN_INFO_JSON}" \
  -s "${RESULTS_DIR}/eval_result.txt"

echo "========================================================"
echo " [Step 3] Calculate Scores"
echo "========================================================"

${ICE_PYTHON} cal_scores.py \
  -f "${RESULTS_DIR}/eval_result.txt"

echo "All done! Results saved in ${RESULTS_DIR}"
