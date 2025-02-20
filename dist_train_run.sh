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

# 设置工作目录
WORK_DIR="/mnt/data10/chenyi/learn_pytorch"
cd $WORK_DIR

# 单机单卡模式
run_single_gpu() {
    CUDA_VISIBLE_DEVICES=0 python ${WORK_DIR}/dist_train.py \
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
    local node_rank=$1
    
    # 设置缺失的环境变量
    export NUM_NODES=${PET_NNODES:-1}  # 使用PET_NNODES，如果不存在则默认为1
    export LOCAL_RANK=${LOCAL_RANK:-$(($RANK % $AVAILABLE_GPUS))}
    
    echo "Running distributed training with:"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"
    echo "LOCAL_RANK: $LOCAL_RANK"
    echo "NUM_NODES: $NUM_NODES"
    echo "PET_NNODES: $PET_NNODES"
    echo "Available GPUs: $AVAILABLE_GPUS"
    echo "Working Directory: $WORK_DIR"
    
    # 使用本地GPU的序号
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
    
    # 使用完整路径运行Python脚本
    python ${WORK_DIR}/dist_train.py \
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
        run_distributed $PARAM
        ;;
    *)
        echo "Invalid mode. Use: single, multi, or distributed"
        exit 1
        ;;
esac 