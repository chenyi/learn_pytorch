import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import argparse
import yaml

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

def train(rank, world_size, config):
    # 设置分布式环境
    setup_distributed(rank, world_size, config)
    
    # 创建模型
    model = torch.nn.Linear(10, 2).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset()
    sampler = torch.utils.data.DistributedSampler(dataset,
                                                 num_replicas=world_size,
                                                 rank=rank)
    dataloader = DataLoader(dataset, 
                          batch_size=config['batch_size'],
                          sampler=sampler)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch)
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()
            
            # 只在主进程上打印信息
            if batch_idx % 10 == 0 and rank == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(
                    f"Epoch: {epoch}/{config['epochs']-1} | "
                    f"Batch: {batch_idx}/{len(dataloader)-1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Accuracy: {accuracy:.2f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # 每个epoch结束后打印统计信息
        if rank == 0:  # 只在主进程保存模型
            epoch_loss = total_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            print(f"\nEpoch {epoch} Summary:")
            print(f"Average Loss: {epoch_loss:.4f}")
            print(f"Accuracy: {epoch_acc:.2f}%\n")
            
            # 保存最好的模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                # 确保保存目录存在
                os.makedirs(config['save_dir'], exist_ok=True)
                # 如果是DDP模型，需要保存内部的module
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
            
            # 保存最新的模型
            save_path = os.path.join(config['save_dir'], 'latest_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
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