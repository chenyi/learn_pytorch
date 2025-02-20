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
import yaml

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def setup_distributed(rank, world_size, config):
    """设置分布式训练环境"""
    if config['distributed_mode'] == 'multi_node':
        # 多机多卡模式使用环境变量中的地址
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
    else:
        # 单机模式使用配置文件中的地址
        master_addr = config['master_addr']
        master_port = config['master_port']
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(backend='nccl',
                          init_method='env://',
                          rank=rank,
                          world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def get_cifar10_loaders(world_size, rank, config):
    """准备CIFAR10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    # 设置数据集路径
    data_dir = os.path.join(config['data_dir'], 'cifar10')
    os.makedirs(data_dir, exist_ok=True)

    # 加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    # 加载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    
    test_sampler = torch.utils.data.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
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

def train(rank, world_size, config):
    # 设置分布式环境
    setup_distributed(rank, world_size, config)
    
    # 创建模型
    model = ConvNet().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 获取数据加载器
    train_loader, test_loader = get_cifar10_loaders(world_size, rank, config)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_acc = 0
    for epoch in range(config['epochs']):
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
            if batch_idx % config['print_freq'] == 0 and rank == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(
                    f"Epoch: {epoch}/{config['epochs']-1} | "
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
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                os.makedirs(config['save_dir'], exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, save_path)
                print(f"Saved best model with accuracy {best_acc:.2f}% to {save_path}")
            
            # 保存最新的模型
            save_path = os.path.join(config['save_dir'], 'latest_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dist_train_config.yaml')
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据配置确定使用的GPU数量
    world_size = torch.cuda.device_count()
    if config['distributed_mode'] == 'single_gpu':
        world_size = 1
    
    if world_size > 0:
        mp.spawn(train,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
    else:
        print("No GPU available!")

if __name__ == "__main__":
    main() 