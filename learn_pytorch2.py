import torch
# create tensor of zeros and ones
zeros = torch.zeros(3, 4)
print(f"The tensor of zeros: {zeros}")

ones = torch.ones(3, 4)
print(f"The tensor of ones: {ones}")

# create a vector of size 3 with values from 0 to 2
vector = torch.tensor([3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)
print(f"The vector of size 3 with values from 0 to 2: {vector}")
# the dtype of the vector is float32
print(f"The dtype of the vector: {vector.dtype}")


# turn float32 to float16
vector = vector.to(torch.float16)
print(f"The dtype of the vector: {vector.dtype}")
# print the device of the vector
print(f"The device of the vector: {vector.device}")

# tensor manipulation
# add 1 to each element of the vector
vector = vector + 1
print(f"The vector after adding 1 to each element: {vector}")

# multiply each element of the vector by 2
vector = vector * 2
print(f"The vector after multiplying each element by 2: {vector}")

# add two tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = tensor1 + tensor2
print(f"The tensor after adding two tensors: {tensor3}")

# multiply two tensors
tensor4 = tensor1 * tensor2
print(f"The tensor after multiplying two tensors: {tensor4}")

# multiply a tensor by a scalar
tensor5 = tensor1 * 2
print(f"The tensor after multiplying a tensor by a scalar: {tensor5}")

# matrix multiplication
tensor6 = torch.matmul(tensor1, tensor2)
print(f"The tensor after matrix multiplication: {tensor6}")

