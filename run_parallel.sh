#!/bin/bash

# Configuration
SCRIPT="Qwen-Image-Test-wost-ab.py"
GPU_IDS=(0 1 2 3 4)  # Array of GPU IDs to use
NUM_WORKERS=5
LOG_DIR="logs"

# Kill existing processes
echo "Killing existing $SCRIPT processes..."
pkill -f "python $SCRIPT" || true
sleep 2

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up locks
rm -f locks/*.lock

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
rm -f locks/*.lock
echo "All workers finished."

