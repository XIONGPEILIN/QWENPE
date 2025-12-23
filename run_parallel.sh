#!/bin/bash

# Target GPUs (Avoiding 1 and 5)
GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_WORKERS=${#GPU_IDS[@]}

echo "Starting $NUM_WORKERS workers on GPUs: ${GPU_IDS[*]}"

pids=()

for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    WORKER_ID=$i
    
    echo "Launching Worker $WORKER_ID on GPU $GPU_ID..."
    
    # We run in background, redirecting logs to a file per worker
    CUDA_VISIBLE_DEVICES=$GPU_ID python Qwen-Image-Test.py \
        --worker_id $WORKER_ID \
        --num_workers $NUM_WORKERS \
        > "worker_${WORKER_ID}_gpu_${GPU_ID}.log" 2>&1 &
    
    pids+=($!)
    
    # Sleep to avoid launch congestion
    sleep 1
done

echo "All workers launched. Waiting for completion..."
echo "You can check progress with: tail -f worker_*.log"

# Wait for all background processes
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks finished."
