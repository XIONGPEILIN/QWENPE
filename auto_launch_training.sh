#!/bin/bash
# filepath: auto_launch_training.sh

# 配置
CHECK_INTERVAL=10        # 检查间隔（秒）
IDLE_THRESHOLD=60          # 空闲阈值（秒）
MIN_MEMORY_MB=100          # 低于此显存(MB)认为空闲
TARGET_GPUS=(1 2 3 4 5 6 7)  # 监控的GPU列表
REQUIRED_GPUS=4            # 需要的空闲GPU数量
TRAINING_SCRIPT="Qwen-Image-Edit-2509.sh"

# 记录每个GPU开始空闲的时间
declare -A idle_start_time

echo "=========================================="
echo "GPU 监控脚本启动"
echo "监控 GPU: ${TARGET_GPUS[*]}"
echo "需要 $REQUIRED_GPUS 个空闲GPU"
echo "空闲阈值: ${IDLE_THRESHOLD}秒"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo "=========================================="

while true; do
    current_time=$(date +%s)
    idle_gpus=()
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查GPU状态..."
    
    for gpu_id in "${TARGET_GPUS[@]}"; do
        # 获取GPU显存使用量(MB)
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        
        if [ -z "$memory_used" ]; then
            echo "  GPU $gpu_id: 无法获取状态"
            continue
        fi
        
        if [ "$memory_used" -lt "$MIN_MEMORY_MB" ]; then
            # GPU空闲
            if [ -z "${idle_start_time[$gpu_id]}" ]; then
                idle_start_time[$gpu_id]=$current_time
                echo "  GPU $gpu_id: 空闲 (${memory_used}MB) - 开始计时"
            else
                idle_duration=$((current_time - idle_start_time[$gpu_id]))
                echo "  GPU $gpu_id: 空闲 (${memory_used}MB) - 已空闲 ${idle_duration}秒"
                
                if [ "$idle_duration" -ge "$IDLE_THRESHOLD" ]; then
                    idle_gpus+=("$gpu_id")
                fi
            fi
        else
            # GPU在使用
            if [ -n "${idle_start_time[$gpu_id]}" ]; then
                echo "  GPU $gpu_id: 使用中 (${memory_used}MB) - 重置计时"
                unset idle_start_time[$gpu_id]
            else
                echo "  GPU $gpu_id: 使用中 (${memory_used}MB)"
            fi
        fi
    done
    
    echo "  空闲超过${IDLE_THRESHOLD}秒的GPU: ${idle_gpus[*]:-无}"
    
    # 检查是否有足够的空闲GPU
    if [ "${#idle_gpus[@]}" -ge "$REQUIRED_GPUS" ]; then
        # 选择前 REQUIRED_GPUS 个空闲GPU
        selected_gpus=("${idle_gpus[@]:0:$REQUIRED_GPUS}")
        gpu_list=$(IFS=,; echo "${selected_gpus[*]}")
        
        echo ""
        echo "=========================================="
        echo "检测到 ${#idle_gpus[@]} 个GPU空闲超过${IDLE_THRESHOLD}秒"
        echo "选择 GPU: $gpu_list"
        echo "启动训练脚本: $TRAINING_SCRIPT"
        echo "=========================================="
        
        # 修改并运行训练脚本
        export CUDA_VISIBLE_DEVICES="$gpu_list"
        echo "设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        
        # 执行训练脚本
        bash "$TRAINING_SCRIPT"
        
        echo ""
        echo "训练脚本执行完毕，退出监控"
        exit 0
    fi
    
    sleep "$CHECK_INTERVAL"
done