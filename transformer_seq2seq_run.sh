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
    echo "Usage: $0 <node_rank>"
    echo "Example:"
    echo "  On first node:  $0 0"
    echo "  On second node: $0 1"
    exit 1
fi

NODE_RANK=$1

# 基础训练参数
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
EPOCHS=${EPOCHS:-10}
PRINT_FREQ=${PRINT_FREQ:-50}
DATA_DIR="/root/code/learn_pytorch/data"
SAVE_DIR="/root/code/learn_pytorch/checkpoints"

# 分布式训练配置
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${NNODES:-2}  # 总节点数
GPUS_PER_NODE=${GPUS_PER_NODE:-4}  # 每个节点的GPU数量
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "Starting distributed training with:"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"

# 使用 torch.distributed.launch 启动训练
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_port=$MASTER_PORT \
    /root/code/learn_pytorch/transformer_seq2seq.py \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --print_freq $PRINT_FREQ \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR 