#!/bin/bash

# 训练参数说明：
# BATCH_SIZE: 每个GPU上的批次大小
# LEARNING_RATE: 学习率
# EPOCHS: 训练轮数
# PRINT_FREQ: 打印频率
# DATA_DIR: 数据集路径
# SAVE_DIR: 模型保存路径

# 运行模式：
# distributed: 多机多卡训练

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 distributed [node_rank]"
    echo "Examples:"
    echo "  $0 distributed 0    # 分布式训练，节点0"
    exit 1
fi

MODE=$1
PARAM=${2:-0}  # 第二个参数：distributed模式下表示node_rank

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# 基础训练参数
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
EPOCHS=${EPOCHS:-10}
PRINT_FREQ=${PRINT_FREQ:-50}
DATA_DIR="./data"
SAVE_DIR="./checkpoints"

# 分布式模式
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
    echo "Available GPUs: $AVAILABLE_GPUS"
    
    # 使用本地GPU的序号
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
    
    # 获取当前脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    python ${SCRIPT_DIR}/transformer_seq2seq.py \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --print_freq $PRINT_FREQ \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR
}

# 只支持分布式模式
if [ "$MODE" == "distributed" ]; then
    run_distributed
else
    echo "Invalid mode. Use: distributed"
    exit 1
fi 