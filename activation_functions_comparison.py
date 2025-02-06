import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建测试数据
# linspace创建一个等间距的一维张量，从-5到5，共200个点
x = torch.linspace(-5, 5, 200)

# 2. 定义不同的激活函数
activations = {
    # ReLU: f(x) = max(0,x)
    # 最常用的激活函数，对负值置0，保持正值不变
    'ReLU': nn.ReLU(),
    
    # LeakyReLU: f(x) = x if x > 0 else 0.1x
    # ReLU的改进版，负值部分有一个小的斜率(0.1)，避免神经元"死亡"
    'LeakyReLU': nn.LeakyReLU(0.1),
    
    # Sigmoid: f(x) = 1 / (1 + e^(-x))
    # 将输入压缩到(0,1)范围，常用于二分类问题
    'Sigmoid': nn.Sigmoid(),
    
    # Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    # 将输入压缩到(-1,1)范围，输出是零中心化的
    'Tanh': nn.Tanh()
}

# 3. 可视化不同激活函数
plt.figure(figsize=(15, 5))  # 创建15x5英寸的图形
for i, (name, func) in enumerate(activations.items(), 1):
    plt.subplot(1, 4, i)  # 创建1行4列的子图
    # 将激活函数应用于输入数据并绘制
    plt.plot(x.numpy(), func(x).numpy())
    plt.title(name)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()  # 自动调整子图间距
plt.savefig('activation_functions.png')
plt.close()

# 4. 创建一个测试网络来比较不同激活函数的效果
class TestNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        # 第一层：输入维度1，输出维度10
        self.layer1 = nn.Linear(1, 10)
        # 可配置的激活函数
        self.activation = activation
        # 第二层：输入维度10，输出维度1
        self.layer2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.layer1(x)        # 线性变换1
        x = self.activation(x)    # 非线性激活
        x = self.layer2(x)        # 线性变换2
        return x

# 5. 训练函数
def train_and_compare(activation_name, activation_func, x_train, y_train, epochs=1000):
    # 创建模型实例
    model = TestNet(activation_func)
    # 使用Adam优化器，学习率0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 使用均方误差损失函数
    criterion = nn.MSELoss()
    # 用于记录训练过程中的损失值
    losses = []
    
    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()        # 清零梯度
        output = model(x_train)      # 前向传播
        loss = criterion(output, y_train)  # 计算损失
        loss.backward()              # 反向传播
        optimizer.step()             # 更新参数
        losses.append(loss.item())   # 记录损失值
        
        # 每200轮打印一次训练状态
        if (epoch + 1) % 200 == 0:
            print(f'{activation_name} - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 6. 准备训练数据
# 创建输入数据：100个从-3到3的等间距点
x_train = torch.linspace(-3, 3, 100).view(-1, 1)
# 创建目标数据：y = sin(x) * 0.5
y_train = torch.sin(x_train) * 0.5

# 7. 比较不同激活函数的训练效果
plt.figure(figsize=(12, 5))

# 对每个激活函数进行训练和绘制
for name, func in activations.items():
    losses = train_and_compare(name, func, x_train, y_train)
    plt.plot(losses, label=name)

plt.title('Training Loss with Different Activation Functions')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('activation_training_comparison.png')
plt.close() 