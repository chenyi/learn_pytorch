import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SimpleNN(nn.Module):
    """
    简单前馈神经网络(FNN)
    特点：
    1. 全连接层：每个神经元与下一层所有神经元相连
    2. 单向传播：信息只向前流动，无反馈连接
    3. 层次结构：输入层 -> 隐藏层 -> 输出层
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        """
        参数:
        - input_size: 输入维度 (28x28=784，MNIST图像大小)
        - hidden_size: 隐藏层神经元数量 (经验值，可调)
        - num_classes: 输出类别数 (0-9共10类)
        """
        super().__init__()
        self.flatten = nn.Flatten()  # 将2D图像(28x28)展平为1D向量(784)
        
        self.layers = nn.Sequential(
            # 输入层->隐藏层
            nn.Linear(input_size, hidden_size),  
            nn.ReLU(),    # ReLU激活函数: f(x)=max(0,x)，引入非线性，防止梯度消失
            nn.Dropout(0.2),  # 随机关闭20%的神经元，防止过拟合
            
            # 隐藏层->隐藏层
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 隐藏层->输出层
            nn.Linear(hidden_size, num_classes)  # 输出10个类别的概率分布
        )
    
    def forward(self, x):
        """
        前向传播
        x: 输入数据 [batch_size, 1, 28, 28]
        返回: 预测结果 [batch_size, 10]
        """
        x = self.flatten(x)  # 展平: [batch_size, 784]
        return self.layers(x)  # 通过网络层: [batch_size, 10]

def load_data(batch_size=64):
    """
    加载MNIST数据集
    
    数据预处理:
    1. ToTensor(): 将图像转换为张量，并归一化到[0,1]
    2. Normalize(): 标准化，使数据分布接近标准正态分布
                   均值(0.1307)和标准差(0.3081)是MNIST的经验值
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 训练集：60000张图像
    train_dataset = datasets.MNIST('./data', 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
    
    # 测试集：10000张图像
    test_dataset = datasets.MNIST('./data', 
                                train=False, 
                                transform=transform)
    
    # DataLoader: 批量加载数据，提高训练效率
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size,  # 每批数据量
                            shuffle=True)  # 随机打乱数据
    
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False)  # 测试集不需要打乱
    
    return train_loader, test_loader

def setup_visualization_dir():
    """
    创建可视化结果保存目录
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_dir = f'visualization_{timestamp}'
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def save_network_structure(model, vis_dir):
    """
    保存网络结构信息到文本文件
    """
    with open(f'{vis_dir}/network_structure.txt', 'w') as f:
        f.write("\n网络结构:\n")
        f.write("-" * 50 + "\n")
        total_params = 0
        for name, param in model.named_parameters():
            f.write(f"层: {name}\n")
            f.write(f"形状: {param.shape}\n")
            f.write(f"参数数量: {param.numel()}\n")
            total_params += param.numel()
            f.write("-" * 30 + "\n")
        f.write(f"总参数数量: {total_params}\n")
        f.write("-" * 50 + "\n")

def visualize_batch_data(train_loader, vis_dir):
    """
    可视化一个批次的数据并保存
    """
    images, labels = next(iter(train_loader))
    
    fig = plt.figure(figsize=(12, 4))
    
    # 1. 显示批次中的图像样本
    ax1 = fig.add_subplot(131)
    grid_size = min(4, images.size(0))
    for i in range(grid_size**2):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Label: {labels[i]}')
    
    # 2. 显示标签分布
    ax2 = fig.add_subplot(132)
    plt.hist(labels.numpy(), bins=10, range=(0,9))
    plt.title('Label Distribution')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    
    # 3. 显示像素值分布
    ax3 = fig.add_subplot(133)
    plt.hist(images.numpy().flatten(), bins=50)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/batch_visualization.png')
    plt.close()

def visualize_training_batch(model, images, labels, outputs, loss, batch_idx, epoch, vis_dir):
    """
    可视化单个训练批次的结果并保存
    """
    if batch_idx % 100 == 0:  # 每100个批次保存一次
        plt.figure(figsize=(12, 4))
        
        # 1. 显示预测结果
        plt.subplot(131)
        _, predicted = outputs.max(1)
        img = images[0].cpu().squeeze()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[0] == labels[0] else 'red'
        plt.title(f'Pred: {predicted[0]}\nTrue: {labels[0]}', color=color)
        plt.axis('off')
        
        # 2. 显示损失值
        plt.subplot(132)
        plt.plot(loss.item(), 'ro')
        plt.title(f'Batch Loss: {loss.item():.4f}')
        
        # 3. 显示预测概率分布
        plt.subplot(133)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        plt.bar(range(10), probs.detach().cpu().numpy())
        plt.title('Prediction Probabilities')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/training_batch_e{epoch}_b{batch_idx}.png')
        plt.close()

def plot_results(model, train_acc, test_acc, test_loader, device, vis_dir):
    """
    绘制最终结果并保存
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. 训练曲线 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_acc, label='Train Accuracy')
    ax1.plot(test_acc, label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Testing Accuracy')
    ax1.legend()
    
    # 2. 预测示例 (左下)
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # 创建子图网格用于显示预测结果
    gs_pred = gs[1, 0].subgridspec(2, 5)
    for idx in range(10):
        ax = fig.add_subplot(gs_pred[idx // 5, idx % 5])
        img = images[idx].cpu().squeeze()
        ax.imshow(img, cmap='gray')
        color = 'green' if predicted[idx] == labels[idx] else 'red'
        ax.set_title(f'Pred: {predicted[idx].item()}\nTrue: {labels[idx].item()}', 
                    color=color)
        ax.axis('off')
    
    # 3. 图像处理过程展示 (右侧)
    img = images[0]  # 使用第一张图片作为示例
    
    # 展平后的向量表示 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    flattened = img.view(-1).cpu()
    ax2.plot(flattened.numpy())
    ax2.set_title('Flattened Vector (784 dimensions)')
    ax2.set_xlabel('Pixel Position')
    ax2.set_ylabel('Pixel Value')
    
    # 像素值分布 (右下)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(flattened.numpy(), bins=50)
    ax3.set_title('Pixel Value Distribution')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/final_results.png')
    plt.close()

def train(model, train_loader, criterion, optimizer, device, epoch, vis_dir):
    """
    训练函数
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 保存可视化结果
        visualize_training_batch(model, data, target, outputs, loss, 
                               batch_idx, epoch, vis_dir)
        
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

def test(model, test_loader, device):
    """
    测试函数：评估模型在测试集上的性能
    """
    model.eval()  # 设置为评估模式（禁用Dropout等）
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    """
    主函数
    """
    # 创建可视化目录
    vis_dir = setup_visualization_dir()
    print(f"可视化结果将保存在: {vis_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    train_loader, test_loader = load_data(batch_size)
    
    # 保存数据集可视化
    visualize_batch_data(train_loader, vis_dir)
    
    model = SimpleNN().to(device)
    save_network_structure(model, vis_dir)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}')
        train_acc = train(model, train_loader, criterion, optimizer, device, epoch, vis_dir)
        test_acc = test(model, test_loader, device)
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
    
    # 保存最终结果
    plot_results(model, train_acc_history, test_acc_history, test_loader, device, vis_dir)

if __name__ == '__main__':
    main()