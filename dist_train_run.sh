#!/bin/bash

# 训练参数说明：
# BATCH_SIZE: 每个GPU上的批次大小
# LEARNING_RATE: 学习率
# EPOCHS: 训练轮数
# WEIGHT_DECAY: 权重衰减系数
# PRINT_FREQ: 打印频率
# DATA_DIR: 数据集路径
# SAVE_DIR: 模型保存路径

# 运行模式：
# single: 单GPU训练
# multi: 单机多卡训练
# distributed: 多机多卡训练

# 环境变量处理：
# LOCAL_RANK: 本地GPU编号
# NUM_NODES: 总节点数
# WORLD_SIZE: 总GPU数量
# RANK: 全局进程编号

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
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-0.001}
EPOCHS=${EPOCHS:-100}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
PRINT_FREQ=${PRINT_FREQ:-50}
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

# 多机多卡/单机多卡模式统一处理
run_distributed() {
    # 设置缺失的环境变量
    export NUM_NODES=${PET_NNODES:-1}
    
    # 确保LOCAL_RANK正确设置
    if [ -z "$LOCAL_RANK" ]; then
        if [ -n "$RANK" ]; then
            # 在每个节点上，LOCAL_RANK应该是0，因为每个节点只有一个GPU
            export LOCAL_RANK=0
        fi
    fi
    
    echo "Running distributed training with:"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"
    echo "LOCAL_RANK: $LOCAL_RANK"
    echo "NUM_NODES: $NUM_NODES"
    echo "PET_NNODES: $PET_NNODES"
    echo "Available GPUs: $AVAILABLE_GPUS"
    
    # 使用本地GPU的序号
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
    
    # 获取当前脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    python ${SCRIPT_DIR}/dist_train.py \
        --distributed_mode multi_node \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --print_freq $PRINT_FREQ \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR
}

# 简化命令处理
case "$MODE" in
    "single")
        run_single_gpu
        ;;
    "multi"|"distributed")
        run_distributed
        ;;
    *)
        echo "Invalid mode. Use: single, multi, or distributed"
        exit 1
        ;;
esac 