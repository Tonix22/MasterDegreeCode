import torch
x = torch.tensor([[[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]],
                 [[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]]])

def vect_diagonals(matrix):
    # Concatenate diagonals into a tensor
    off_diag = torch.cat([torch.diag(matrix, i) for i in range(-1, 2)])
    print(off_diag)


# Create a 4D tensor with dimensions 2x3x4x5
tensor_4d = torch.randn(10, 2, 3, 3)

# Extract a matrix at index (0,1,:,:)
image = 0
channel = 1
matrix = tensor_4d[image, channel, :, :]
# Extract all matrices along the batch dimension
matrices = torch.unbind(tensor_4d, dim=0)

# Iterate through the matrices
for i, matrix in enumerate(matrices):
    print("Matrix {}:".format(i))
    print(matrix[0])
    vect_diagonals(matrix[0])
