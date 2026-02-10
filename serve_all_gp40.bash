#!/bin/bash

# Load user environment
source ~/.bashrc

# Activate vLLM virtual environment
source /host/ssd2/xiong-p/env/vllm/.venv/bin/activate

# Configuration
MODEL="Qwen/Qwen3-VL-32B-Instruct"
MAX_LEN=16384
GPU_UTIL=0.90
TP_SIZE=2

# Function to start a server instance
start_server() {
    local gpus=$1
    local port=$2
    local log_file="serve_gp40_port_${port}.log"
    
    echo "Starting vLLM on GPUs ${gpus} with port ${port}..."
    CUDA_VISIBLE_DEVICES="${gpus}" nohup vllm serve "${MODEL}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --max-model-len "${MAX_LEN}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --limit-mm-per-prompt.video 0 \
        --port "${port}" > "${log_file}" 2>&1 &
        
    echo "Instance on port ${port} started (PID $!). Logs: ${log_file}"
}

# Start 4 parallel instances on all NVLink pairs
start_server "0,1" 7512
start_server "2,3" 7513
start_server "4,5" 7514
start_server "6,7" 7515

echo "All 4 instances launched!"
echo "You can check logs with: tail -f serve_gp40_port_*.log"
