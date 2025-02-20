import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SimpleCNN(nn.Module):
    """
    简单卷积神经网络(CNN)
    特点：
    1. 卷积层：提取局部特征
    2. 池化层：降维、提取显著特征
    3. 全连接层：分类
    """
    def __init__(self):
        super().__init__()
        # 卷积块1：输入通道1，输出通道16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 28x28 -> 14x14
        )
        
        # 卷积块2：输入通道16，输出通道32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 14x14 -> 7x7
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # 保存中间特征图用于可视化
        self.feature_maps = []
        
        x = self.conv1(x)
        self.feature_maps.append(x.detach().cpu())
        
        x = self.conv2(x)
        self.feature_maps.append(x.detach().cpu())
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def setup_visualization_dir():
    """创建可视化结果保存目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_dir = f'visualization_cnn_{timestamp}'
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def save_network_structure(model, vis_dir):
    """保存网络结构信息到文本文件"""
    with open(f'{vis_dir}/network_structure.txt', 'w') as f:
        f.write("\nCNN网络结构:\n")
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

def visualize_feature_maps(model, images, epoch, batch_idx, vis_dir):
    """可视化特征图"""
    if batch_idx % 100 == 0:  # 每100个批次保存一次
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(131)
        img = images[0].cpu().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示第一个卷积层的特征图
        plt.subplot(132)
        feature_map1 = model.feature_maps[0][0]
        grid_size = min(4, feature_map1.size(0))
        for i in range(grid_size**2):
            plt.subplot(132 + i // grid_size, grid_size, i + 1)
            plt.imshow(feature_map1[i], cmap='viridis')
            plt.axis('off')
            if i == 0:
                plt.title('Conv1 Feature Maps')
        
        # 显示第二个卷积层的特征图
        plt.subplot(133)
        feature_map2 = model.feature_maps[1][0]
        grid_size = min(4, feature_map2.size(0))
        for i in range(grid_size**2):
            plt.subplot(133 + i // grid_size, grid_size, i + 1)
            plt.imshow(feature_map2[i], cmap='viridis')
            plt.axis('off')
            if i == 0:
                plt.title('Conv2 Feature Maps')
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/feature_maps_e{epoch}_b{batch_idx}.png')
        plt.close()

def load_data(batch_size=64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
    
    test_dataset = datasets.MNIST('./data', 
                                train=False, 
                                transform=transform)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)
    
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device, epoch, vis_dir):
    """训练函数"""
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
        
        # 可视化特征图
        visualize_feature_maps(model, data, epoch, batch_idx, vis_dir)
        
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

def test(model, test_loader, device):
    """测试函数"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def plot_results(model, train_acc, test_acc, test_loader, device, vis_dir):
    """绘制最终结果"""
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
    
    # 3. 特征图示例 (右侧)
    feature_maps = model.feature_maps
    
    # Conv1特征图 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    feature_map1 = feature_maps[0][0]
    grid_size = min(4, feature_map1.size(0))
    for i in range(grid_size**2):
        plt.subplot(gs[0, 1] + i // grid_size, grid_size, i + 1)
        plt.imshow(feature_map1[i], cmap='viridis')
        plt.axis('off')
        if i == 0:
            plt.title('Conv1 Feature Maps')
    
    # Conv2特征图 (右下)
    ax3 = fig.add_subplot(gs[1, 1])
    feature_map2 = feature_maps[1][0]
    grid_size = min(4, feature_map2.size(0))
    for i in range(grid_size**2):
        plt.subplot(gs[1, 1] + i // grid_size, grid_size, i + 1)
        plt.imshow(feature_map2[i], cmap='viridis')
        plt.axis('off')
        if i == 0:
            plt.title('Conv2 Feature Maps')
    
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/final_results.png')
    plt.close()

def main():
    """主函数"""
    # 创建可视化目录
    vis_dir = setup_visualization_dir()
    print(f"可视化结果将保存在: {vis_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    train_loader, test_loader = load_data(batch_size)
    
    model = SimpleCNN().to(device)
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