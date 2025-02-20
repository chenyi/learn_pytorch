import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse

"""
分布式训练架构说明：

1. 进程模型：
   - 每个GPU对应一个独立的Python进程
   - 使用torch.multiprocessing进行进程管理
   - 每个进程运行相同的代码，但处理不同的数据

2. 通信模式：
   - 使用NCCL后端进行GPU间通信
   - 梯度同步在反向传播后自动进行
   - 使用DistributedDataParallel (DDP)包装模型

3. 数据并行：
   - 使用DistributedSampler划分数据集
   - 每个GPU获取不同的数据批次
   - 所有GPU共享相同的模型架构

4. 同步点：
   - 初始化：等待所有进程加入进程组
   - 训练：每个batch后的梯度同步
   - 评估：所有进程的指标聚合
"""

class ConvNet(nn.Module):
    """
    简单的CNN模型，用于CIFAR10分类
    架构：
    1. 卷积层块1：32通道 -> 64通道 -> MaxPool
    2. 卷积层块2：64通道 -> 128通道 -> MaxPool
    3. 全连接层：128*8*8 -> 512 -> 10
    """
    def __init__(self):
        super().__init__()
        # 第一个卷积块：提取低级特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 批归一化加速训练
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),     # 降采样
            # 第二个卷积块：提取高级特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        # 全连接层：分类
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),     # 防止过拟合
            nn.Linear(512, 10)   # 10个类别
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc_layers(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--print_freq', type=int, default=50)
    
    # 分布式训练模式
    parser.add_argument('--distributed_mode', type=str, default='single_gpu',
                      choices=['single_gpu', 'multi_gpu', 'multi_node'])
    
    # 数据和模型保存路径
    parser.add_argument('--data_dir', type=str, default='/mnt/data10/datasets')
    parser.add_argument('--save_dir', type=str, default='/mnt/data10/model_checkpoints')
    
    return parser.parse_args()

def setup_distributed(rank, world_size, args):
    """设置分布式训练环境"""
    # 确保环境变量已设置
    if not os.environ.get('MASTER_ADDR'):
        os.environ['MASTER_ADDR'] = 'localhost'
    if not os.environ.get('MASTER_PORT'):
        os.environ['MASTER_PORT'] = '12355'
    
    # 在PAI-DLC环境中，使用环境变量中的RANK
    rank = int(os.environ.get('RANK', rank))
    world_size = int(os.environ.get('WORLD_SIZE', world_size))
    
    print(f"Initializing process group: rank={rank}, world_size={world_size}")
    print(f"MASTER_ADDR={os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # 初始化进程组
    dist.init_process_group(backend='nccl',
                          init_method='env://',
                          rank=rank,
                          world_size=world_size)

def cleanup():
    """清理分布式环境，关闭进程组"""
    dist.destroy_process_group()

def get_cifar10_loaders(world_size, rank, args):
    """
    准备CIFAR10数据集的数据加载器
    
    分布式数据加载流程：
    1. 定义数据增强和预处理
    2. 创建训练和测试数据集
    3. 使用DistributedSampler划分数据
    4. 创建DataLoader进行批处理
    
    数据并行示意图：
    进程0 (GPU0) ─→ 数据批次 [0, 4, 8, ...]
    进程1 (GPU1) ─→ 数据批次 [1, 5, 9, ...]
    进程2 (GPU2) ─→ 数据批次 [2, 6, 10, ...]
    进程3 (GPU3) ─→ 数据批次 [3, 7, 11, ...]
    """
    # 数据增强和预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),    # 随机裁剪
        transforms.RandomHorizontalFlip(),       # 随机水平翻转
        transforms.ToTensor(),                   # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # 标准化
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # 测试集只需要标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    # 创建数据集
    data_dir = os.path.join(args.data_dir, 'cifar10')
    os.makedirs(data_dir, exist_ok=True)

    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    # 创建分布式采样器
    # 确保每个进程获取不同的数据子集
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,          # 使用多进程加载数据
        pin_memory=True         # 将数据直接加载到GPU内存
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train(rank, world_size, args):
    """
    主训练函数，每个进程执行一份
    
    训练流程：
    1. 初始化分布式环境
    2. 创建模型和优化器
    3. 加载数据
    4. 训练循环：前向传播→反向传播→梯度同步→参数更新
    5. 定期评估和保存模型
    
    梯度同步示意图：
    GPU0 梯度   GPU1 梯度   GPU2 梯度   GPU3 梯度
       ↓           ↓           ↓           ↓
    ┌─────────────────── AllReduce ────────────────┐
    ↓           ↓           ↓           ↓          │
    平均梯度    平均梯度    平均梯度    平均梯度   │
    ↓           ↓           ↓           ↓          │
    参数更新    参数更新    参数更新    参数更新   │
    └──────────────────────────────────────────────┘
    """
    # 设置分布式环境
    setup_distributed(rank, world_size, args)
    
    # 创建模型
    model = ConvNet().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 获取数据加载器
    train_loader, test_loader = get_cifar10_loaders(world_size, rank, args)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 统计
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()
            
            # 打印训练信息
            if batch_idx % args.print_freq == 0 and rank == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(
                    f"Epoch: {epoch}/{args.epochs-1} | "
                    f"Batch: {batch_idx}/{len(train_loader)-1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Accuracy: {accuracy:.2f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # 评估模型
        model.eval()
        test_loss, test_acc = evaluate(model, test_loader, criterion, rank)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 保存模型和打印评估结果
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%\n")
            
            # 保存最好的模型
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, save_path)
                print(f"Saved best model with accuracy {best_acc:.2f}% to {save_path}")
            
            # 保存最新的模型
            save_path = os.path.join(args.save_dir, 'latest_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
    
    cleanup()

def main():
    args = parse_args()
    
    # 获取本地GPU数量
    local_gpus = torch.cuda.device_count()
    
    # 从环境变量获取分布式信息
    world_size = int(os.environ.get('WORLD_SIZE', local_gpus))
    rank = int(os.environ.get('RANK', 0))
    
    # 计算每个节点应该使用的GPU数量
    gpus_per_node = world_size // int(os.environ.get('NUM_NODES', 1))
    local_rank = rank % gpus_per_node  # 本地GPU编号
    
    if args.distributed_mode in ['multi_gpu', 'multi_node']:
        # 直接运行训练函数，不使用spawn
        train(local_rank, world_size, args)
    else:
        # 单GPU模式
        train(0, 1, args)

if __name__ == "__main__":
    main() 