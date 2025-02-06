import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

"""
本教程目的：
1. 理解不同优化器的工作方式和性能差异
2. 学习如何使用DataLoader实现小批量训练
3. 观察优化器对训练过程的影响
4. 理解如何可视化和分析训练结果
"""

# 1. 生成训练数据
def generate_data():
    """
    生成带噪声的训练数据
    目的：创建一个简单的非线性回归问题
    - 使用二次函数 y = x^2 作为基础
    - 添加随机噪声使其更接近实际问题
    """
    # 生成 x 在 [-5, 5] 之间的 1000 个均匀分布的点
    x = torch.linspace(-5, 5, 1000)
    # 生成带高斯噪声的 y = x^2 数据，标准差为3
    y = x**2 + torch.normal(0, 3, size=x.shape)
    # 重塑数据维度为 (n_samples, 1)
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    return x, y

# 2. 定义神经网络模型
class SimpleNet(nn.Module):
    """
    简单的前馈神经网络
    架构：输入层(1) -> 隐藏层(16) -> 隐藏层(16) -> 输出层(1)
    
    目的：
    1. 提供足够的非线性能力来拟合数据
    2. 保持结构简单以便观察优化器的影响
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),    # 第一层：扩展特征维度
            nn.ReLU(),           # 非线性激活
            nn.Linear(16, 16),   # 第二层：保持特征维度
            nn.ReLU(),           # 非线性激活
            nn.Linear(16, 1)     # 输出层：回归到一个值
        )
    
    def forward(self, x):
        return self.net(x)

# 3. 训练函数
def train_model(model, optimizer, train_loader, epochs=1000):
    """
    使用DataLoader进行小批量训练
    
    参数：
    - model: 要训练的神经网络模型
    - optimizer: 使用的优化器
    - train_loader: 数据加载器，提供小批量数据
    - epochs: 训练轮数
    """
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # 每个epoch都会打乱数据顺序
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # 计算每个epoch的平均损失
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    
    return losses

# 4. 主函数：比较不同优化器
def compare_optimizers():
    """
    比较不同优化器的性能
    
    目的：
    1. 展示如何设置和使用不同的优化器
    2. 比较它们的收敛速度和最终性能
    3. 可视化训练过程和结果
    """
    # 生成训练数据
    x_train, y_train = generate_data()
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,          # 小批量大小
        shuffle=True,           # 随机打乱
        num_workers=0,          # 单进程加载
        drop_last=True          # 丢弃不完整的批次
    )
    
    # 定义要比较的优化器
    optimizers = {
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01),
        'SGD with momentum': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.01),
        'RMSprop': lambda params: torch.optim.RMSprop(params, lr=0.01)
    }
    
    # 存储每个优化器的训练结果
    all_losses = {}
    trained_models = {}
    
    # 使用每个优化器训练模型
    for opt_name, opt_class in optimizers.items():
        print(f"\nTraining with {opt_name}")
        model = SimpleNet()
        optimizer = opt_class(model.parameters())
        losses = train_model(model, optimizer, train_loader)
        all_losses[opt_name] = losses
        trained_models[opt_name] = model
    
    # 可视化训练损失
    plt.figure(figsize=(12, 6))
    for opt_name, losses in all_losses.items():
        plt.plot(losses, label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss with Different Optimizers (Mini-batch)')
    plt.legend()
    plt.yscale('log')  # 使用对数尺度更好地显示损失变化
    plt.grid(True)
    plt.savefig('optimizer_comparison_loss.png')
    plt.close()
    
    # 可视化拟合结果
    plt.figure(figsize=(15, 5))
    x_test = torch.sort(x_train, dim=0)[0]
    
    for i, (opt_name, model) in enumerate(trained_models.items(), 1):
        plt.subplot(1, 4, i)
        with torch.no_grad():
            y_pred = model(x_test)
        
        plt.scatter(x_train, y_train, color='blue', alpha=0.1, label='Data')
        plt.plot(x_test, y_pred, color='red', label='Prediction')
        plt.title(f'{opt_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_fit.png')
    plt.close()

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行优化器比较
    compare_optimizers() 