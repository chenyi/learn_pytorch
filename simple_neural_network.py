import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 创建一个简单的神经网络类
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # 定义第一层（输入层到隐藏层）
        self.layer1 = nn.Linear(input_size, hidden_size)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义第二层（隐藏层到输出层）
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 定义前向传播过程
        x = self.layer1(x)      # 第一层线性变换
        x = self.relu(x)        # 使用ReLU激活函数
        x = self.layer2(x)      # 第二层线性变换
        return x

# 2. 创建训练数据
# 生成一些带有噪声的数据点
x = torch.linspace(-10, 10, 100)  # 创建100个-10到10之间均匀分布的点
x = x.view(-1, 1)  # 改变形状为 [100, 1]
y = x**2 + torch.randn(x.size()) * 50  # y = x² + 随机噪声

# 3. 创建模型实例
model = SimpleNN(input_size=1, hidden_size=10, output_size=1)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # 随机梯度下降优化器

# 5. 训练模型
epochs = 1000
losses = []  # 用于记录损失值

for epoch in range(epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 记录损失
    losses.append(loss.item())
    
    # 每100轮打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. 可视化结果并保存
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制拟合结果
plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y.numpy(), label='Original Data')
plt.plot(x.numpy(), model(x).detach().numpy(), 'r', label='Fitted Line')
plt.title('Regression Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()

# 保存图片到本地文件
plt.savefig('training_results.png')
plt.close()  # 关闭图形，释放内存 