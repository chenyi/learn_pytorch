import os
import torch
from megatron import get_args
from megatron import initialize_megatron
from megatron import mpu
from megatron.model import GPT2Model
from megatron.training import pretrain_gpt
from megatron.data import build_train_valid_test_datasets
from megatron.data import build_data_loader

def main():
    # 1. 初始化 Megatron
    initialize_megatron()

    # 2. 获取训练参数
    args = get_args()

    # 3. 设置当前设备
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取本地GPU编号
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU

    # 4. 创建模型
    model = GPT2Model(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings
    )

    # 将模型移动到 GPU
    model = model.cuda()

    # 5. 构建数据集
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets()

    # 6. 创建数据加载器
    train_loader = build_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 7. 训练循环
    for epoch in range(args.num_epochs):
        model.train()  # 设置模型为训练模式
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}...")

        for batch_idx, batch in enumerate(train_loader):
            # 获取输入数据
            input_ids = batch['input_ids'].cuda()  # 将输入数据移动到GPU
            attention_mask = batch['attention_mask'].cuda()  # 将注意力掩码移动到GPU

            # 8. 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)

            # 9. 计算损失
            loss = outputs.loss
            loss.backward()  # 反向传播

            # 10. 更新参数
            optimizer.step()
            optimizer.zero_grad()  # 清零梯度

            # 11. 打印训练信息
            if batch_idx % args.print_freq == 0:
                print(f'Epoch: {epoch + 1}/{args.num_epochs}, '
                      f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        print(f"Completed epoch {epoch + 1}/{args.num_epochs}.")

if __name__ == "__main__":
    main() 