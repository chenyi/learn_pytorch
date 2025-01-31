import torch

# matrix multiplication 
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
tensor3 = torch.matmul(tensor1, tensor2)
print(f"The tensor after matrix multiplication: {tensor3}")
tensor5 = tensor1 @ tensor2
print(f"The tensor after matrix multiplication: {tensor5}")
tensor4 = tensor1 * tensor2
print(f"The tensor after element-wise multiplication: {tensor4}")

tensor6 = torch.rand(4, 3)
tensor7 = torch.rand(3, 5)
tensor8 = torch.matmul(tensor6, tensor7)
print(f"The tensor after matrix multiplication: {tensor8}")

# example of transpose
tensor9 = torch.rand(4, 3)
tensor10 = tensor9.T
print(f"The tensor after transpose: {tensor10}")




# example of aggregation
tensor11 = torch.rand(4, 3)
tensor12 = tensor11.sum()
print(f"The tensor after aggregation: {tensor12}")

# mean
tensor13 = torch.rand(4, 3)
tensor14 = tensor13.mean()
print(f"The tensor after mean: {tensor14}")


# max and min
tensor15 = torch.rand(4, 3)
tensor16 = tensor15.max()
tensor17 = tensor15.min()
print(f"The tensor after max: {tensor16}")
print(f"The tensor after min: {tensor17}")

## example of reshape
tensor18 = torch.rand(4, 3)
tensor19 = tensor18.reshape(2, 6)
print(f"The tensor after reshape: {tensor19}")

