import torch
# 自动求导示例
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"dy/dx = {x.grad}")  # 输出导数值


import torch

# 创建一个需要求导的张量
x = torch.tensor(2.0, requires_grad=True)
# 进行一些运算
y = x ** 2 + 2*x + 1
# 计算导数
y.backward()
# 查看x的梯度
print(f"x = {x}")
print(f"y = {y}")
print(f"dy/dx = {x.grad}")  # 结果应该是 2x + 2 = 6


import torch

# 创建向量
x = torch.tensor([2.0, 3.0], requires_grad=True)
# 进行运算
y = x ** 2
z = y.sum()  # 需要得到标量才能求导
# 计算梯度
z.backward()
print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
print(f"dz/dx = {x.grad}")  # 结果应该是 [4.0, 6.0]

import torch

x = torch.tensor(2.0, requires_grad=True)

# 第一次计算
y = x ** 2
y.backward()
print(f"First gradient: {x.grad}")  # 输出 4.0

# 梯度会累积！
y = x ** 2
y.backward()
print(f"Accumulated gradient: {x.grad}")  # 输出 8.0

# 清零梯度
x.grad.zero_()
y = x ** 2
y.backward()
print(f"After zero_grad: {x.grad}")  # 输出 4.0