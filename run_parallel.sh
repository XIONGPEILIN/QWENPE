#!/bin/bash

# Configuration
SCRIPT="Qwen-Image-Test-LBM.py"
GPU_IDS=(0 4 5 )  # Array of GPU IDs to use
NUM_WORKERS=3
LOG_DIR="logs"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting $NUM_WORKERS workers on GPUs: ${GPU_IDS[*]}"
echo "Script: $SCRIPT"
echo "Logs will be saved to: $LOG_DIR"

# Launch workers
for ((i=0; i<NUM_WORKERS; i++)); do
    gpu_idx=$((i % ${#GPU_IDS[@]}))
    gpu_id=${GPU_IDS[$gpu_idx]}
    
    echo "Launching worker $i on GPU $gpu_id..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT" \
        --worker_id "$i" \
        --num_workers "$NUM_WORKERS" \
        > "$LOG_DIR/worker_$i.log" 2>&1 &
        
    pids[$i]=$!
done

echo "All workers launched. Waiting for completion..."

# Wait for all background processes
for pid in ${pids[*]}; do
    wait $pid
done

echo "All workers finished."

