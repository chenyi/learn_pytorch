import torch

# 检查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查是否可以使用 CUDA（GPU 支持）
print(f"CUDA is available: {torch.cuda.is_available()}")

# 创建一个简单的张量测试
x = torch.rand(5, 3)
print("Random tensor:")
print(x)
