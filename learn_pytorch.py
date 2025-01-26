import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# 1. Creating tensors of different dimensions
# 0D tensor (scalar)
scalar = torch.tensor(7)
print(f"Scalar tensor: {scalar}")  # tensor(7)

# 1D tensor (vector)
vector = torch.tensor([1, 2, 3])
print(f"Vector tensor: {vector}")  # tensor([1, 2, 3])

# 2D tensor (matrix)
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"Matrix tensor:\n{matrix}")

# 3D tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D tensor:\n{tensor_3d}")

# 2. Common tensor creation methods
zeros = torch.zeros(2, 3)  # Create 2x3 tensor filled with zeros
ones = torch.ones(2, 3)    # Create 2x3 tensor filled with ones
rand = torch.rand(2, 3)    # Create 2x3 tensor with random values
range_tensor = torch.arange(0, 10, 2)  # Create tensor with arithmetic sequence

# 3. Basic tensor attributes
print(f"Tensor shape: {matrix.shape}")      # View shape
print(f"Tensor dimensions: {matrix.dim()}")  # View number of dimensions
print(f"Tensor dtype: {matrix.dtype}")      # View data type

# 4. Tensor operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(f"Addition: {a + b}")
# Or using torch.add
print(f"Addition: {torch.add(a, b)}")

# Multiplication
print(f"Multiplication: {a * b}")  # Element-wise multiplication
print(f"Dot product: {torch.dot(a, b)}")  # Dot product

# 5. Tensor and GPU
if torch.cuda.is_available():
    gpu_tensor = matrix.cuda()  # Move to GPU
    # Or using .to()
    gpu_tensor = matrix.to('cuda')

# print the version of pytorch
print(f"PyTorch version: {torch.__version__}")

# scalar tensor and the dimension of the tensor
scalar = torch.tensor(7)
print(f"Scalar tensor: {scalar}")  # tensor(7)
print(f"The dimension of the scalar tensor: {scalar.ndim}")  # 0

# get the int from the scalar tensor
print(f"The int from the scalar tensor: {scalar.item()}")  # 7

# get a vector tensor
vector = torch.tensor([1, 2, 3])
print(f"The vector tensor: {vector}")  # tensor([1, 2, 3])
print(f"The dimension of the vector tensor: {vector.ndim}")  # 1

# get the shape of the tensor
print(f"The shape of the tensor: {vector.shape}")  # torch.Size([3])

# get the size of the tensor
print(f"The size of the tensor: {vector.size()}")  # torch.Size([3])    


# matrix tensor
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"The matrix tensor: {matrix}")  # tensor([[1, 2], [3, 4]])
print(f"The dimension of the matrix tensor: {matrix.ndim}")  # 2    

# get the shape of the tensor
print(f"The shape of the tensor: {matrix.shape}")  # torch.Size([2, 2])

# get the size of the tensor
print(f"The size of the tensor: {matrix.size()}")  # torch.Size([2, 2])

# get 3 dimension tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"The 3 dimension tensor: {tensor_3d}")  # tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"The dimension of the 3 dimension tensor: {tensor_3d.ndim}")  # 3

# get the shape of the tensor
print(f"The shape of the tensor: {tensor_3d.shape}")  # torch.Size([2, 2, 2])

# get the size of the tensor
print(f"The size of the tensor: {tensor_3d.size()}")  # torch.Size([2, 2, 2])

# get 4 dimension tensor
tensor_4d = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
print(f"The 4 dimension tensor: {tensor_4d}")  # tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
print(f"The dimension of the 4 dimension tensor: {tensor_4d.ndim}")  # 4

# get the shape of the tensor
print(f"The shape of the tensor: {tensor_4d.shape}")  # torch.Size([2, 2, 2, 2])

# get the size of the tensor
print(f"The size of the tensor: {tensor_4d.size()}")  # torch.Size([2, 2, 2, 2])

# create random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(f"The random tensor: {random_tensor}")  # tensor([[0.1234, 0.5678, 0.9012, 0.3456], [0.7890, 0.2345, 0.6789, 0.4567], [0.8901, 0.5678, 0.2345, 0.9012]])

# create a random tensor of size (3, 4, 5)
random_tensor_3d = torch.rand(3, 4, 5)
print(f"The random tensor of size (3, 4, 5): {random_tensor_3d}")
print(f"The dimension of the random tensor of size (3, 4, 5): {random_tensor_3d.ndim}")  # 3

# create a random tensor of simlilar shape of an image tensor
image_tensor = torch.rand(1, 3, 224, 224)
print(f"The random tensor of size (1, 3, 224, 224): {image_tensor}")
print(f"The dimension of the random tensor of size (1, 3, 224, 224): {image_tensor.ndim}")  # 4

# 下载一个示例图片（或者你可以使用本地图片）
url = "https://raw.githubusercontent.com/pytorch/pytorch.github.io/master/assets/images/pytorch-logo.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# 将图片调整为4x4大小
img_resized = img.resize((4, 4))

# 确保图片是RGB模式
img_resized = img_resized.convert('RGB')

# 转换为numpy数组
img_array = np.array(img_resized)
print(f"Numpy array shape: {img_array.shape}")  # 应该是 (4, 4, 3)

# 转换为PyTorch张量 [batch_size, channels, height, width]
img_tensor = torch.from_numpy(img_array).float()
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

# 标准化像素值到0-1范围
img_tensor = img_tensor / 255.0

# 显示原始图片和调整大小后的图片
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_array)
plt.title('4x4 Resized Image')
plt.axis('off')

plt.show()

# 打印张量的形状
print(f"\nTensor shape: {img_tensor.shape}")  # Should be [1, 3, 4, 4]

# 打印每个通道的值
print("\nRed channel (R):")
print(img_tensor[0, 0])  # 第一个通道 - 红色
print("\nGreen channel (G):")
print(img_tensor[0, 1])  # 第二个通道 - 绿色
print("\nBlue channel (B):")
print(img_tensor[0, 2])  # 第三个通道 - 蓝色

# 让我们看看第一个像素的RGB值
first_pixel = img_tensor[0, :, 0, 0]
print("\nFirst pixel RGB values:")
print(f"R: {first_pixel[0]:.3f}")
print(f"G: {first_pixel[1]:.3f}")
print(f"B: {first_pixel[2]:.3f}")

# 可视化每个通道
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_tensor[0, 0].numpy(), cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_tensor[0, 1].numpy(), cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_tensor[0, 2].numpy(), cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.show()

