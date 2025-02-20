#!/bin/bash

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

# 基础训练参数
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCHS=100
WEIGHT_DECAY=0.0001
PRINT_FREQ=50
DATA_DIR="/mnt/data10/datasets"
SAVE_DIR="/mnt/data10/model_checkpoints"

# 单机单卡模式
run_single_gpu() {
    CUDA_VISIBLE_DEVICES=0 python dist_train.py \
        --distributed_mode single_gpu \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --print_freq $PRINT_FREQ \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR
}

# 单机多卡模式
run_multi_gpu() {
    local num_gpus=$1
    if [ $num_gpus -gt $AVAILABLE_GPUS ]; then
        echo "Error: Requested $num_gpus GPUs but only $AVAILABLE_GPUS available"
        exit 1
    fi
    
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1))) python dist_train.py \
        --distributed_mode multi_gpu \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --print_freq $PRINT_FREQ \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR
}

# 多机多卡模式
run_distributed() {
    local node_rank=$1
    
    # PAI-DLC环境中已经设置：
    # MASTER_ADDR (例如: dlc8014p3oyrw9hi-master-0)
    # MASTER_PORT (例如: 23456)
    # WORLD_SIZE  (总GPU数量)
    # RANK        (当前进程的rank)
    
    echo "Running distributed training with:"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"
    
    python dist_train.py \
        --distributed_mode multi_node \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --print_freq $PRINT_FREQ \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR
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