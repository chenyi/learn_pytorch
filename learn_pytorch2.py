import torch
# create tensor of zeros and ones
zeros = torch.zeros(3, 4)
print(f"The tensor of zeros: {zeros}")

ones = torch.ones(3, 4)
print(f"The tensor of ones: {ones}")

# create a vector of size 3 with values from 0 to 2
vector = torch.tensor([3.0, 6.0, 9.0], dtype=None)
print(f"The vector of size 3 with values from 0 to 2: {vector}")
# the dtype of the vector is float32
print(f"The dtype of the vector: {vector.dtype}")

