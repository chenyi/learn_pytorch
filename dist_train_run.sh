#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 {single|multi|distributed} [num_gpus|node_rank]"
    echo "Examples:"
    echo "  $0 single            # 单卡训练"
    echo "  $0 multi 2          # 使用2张GPU训练"
    echo "  $0 distributed 0    # 分布式训练，节点0"
    exit 1
fi

MODE=$1
PARAM=${2:-0}  # 第二个参数：multi模式下表示GPU数量，distributed模式下表示node_rank

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# 修改配置文件中的分布式模式
update_config() {
    local mode=$1
    sed -i "s/distributed_mode: .*/distributed_mode: \"$mode\"/" "${SCRIPT_DIR}/dist_train_config.yaml"
}

# 单机单卡模式
run_single_gpu() {
    update_config "single_gpu"
    CUDA_VISIBLE_DEVICES=0 python dist_train.py
}

# 单机多卡模式
run_multi_gpu() {
    local num_gpus=$1
    if [ $num_gpus -gt $AVAILABLE_GPUS ]; then
        echo "Error: Requested $num_gpus GPUs but only $AVAILABLE_GPUS available"
        exit 1
    fi
    
    # 设置可见的GPU数量
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))
    update_config "multi_gpu"
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
    python dist_train.py
}

# 多机多卡模式
run_distributed() {
    local node_rank=$1
    update_config "multi_node"
    export MASTER_ADDR="192.168.1.100"  # 替换为实际的主节点IP
    export MASTER_PORT="12355"
    export WORLD_SIZE=$((AVAILABLE_GPUS * 2))  # 假设有2台机器，每台机器的GPU数量相同
    export NODE_RANK=$node_rank
    
    cd /mnt/data10/  # 添加这行，确保在正确的目录下运行
    python dist_train.py
}

# 根据模式执行相应的命令
case "$MODE" in
    "single")
        run_single_gpu
        ;;
    "multi")
        if [ -z "$PARAM" ]; then
            echo "Please specify number of GPUs for multi-GPU training"
            exit 1
        fi
        run_multi_gpu $PARAM
        ;;
    "distributed")
        run_distributed $PARAM
        ;;
    *)
        echo "Invalid mode. Use: single, multi, or distributed"
        exit 1
        ;;
esac 