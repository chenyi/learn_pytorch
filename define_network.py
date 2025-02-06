import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络，只有一个隐藏层和一个输出层
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 1)  # 输入层到隐藏层
        self.fc2 = nn.Linear(1, 1)  # 隐藏层到输出层
    
    def forward(self, x):
        h = self.fc1(x)
        h_out = torch.relu(h)  # ReLU 激活函数
        y_pred = self.fc2(h_out)
        return y_pred

# 创建网络实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用梯度下降优化器

# 输入和目标标签
x = torch.tensor([[1.0]])  # 输入数据
y_true = torch.tensor([[2.0]])  # 真实标签

# 训练过程
for epoch in range(1000):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = criterion(y_pred, y_true)

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算当前梯度

    # 更新权重
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 查看训练后的权重
print(f'Final weights: {model.fc1.weight.item()}, {model.fc2.weight.item()}')
