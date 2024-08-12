import numpy as np
import torch
from scipy.linalg import qr as scipy_qr

# Using scipy for an accurate reference implementation
def generate_permutation_matrix_scipy(K):
    """
    Generate a permutation matrix for QR decomposition of matrix K using scipy.
    
    Args:
    K (np.array): Channel matrix of shape (M, N).

    Returns:
    P (np.array): Permutation matrix of shape (N, N).
    """
    Q, R, P = scipy_qr(K, pivoting=True)
    permutation_matrix = np.eye(K.shape[1])[:, P]
    return permutation_matrix

# Using PyTorch
def generate_permutation_matrix_torch(K):
    Q, R = torch.linalg.qr(K)
    pivots = -torch.ones(K.shape[1], dtype=torch.int)
    for i in range(R.shape[0]):
        remaining_indices = torch.where(pivots == -1)[0]
        _, max_col_idx = torch.max(torch.abs(R[i, remaining_indices]), 0)
        pivots[remaining_indices[max_col_idx]] = i
    not_pivoted = torch.where(pivots == -1)[0]
    for idx in not_pivoted:
        available_row = torch.where(pivots == -1)[0][0]
        pivots[idx] = available_row
    P = torch.eye(K.shape[1])[pivots]
    return P

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Create a random matrix
K_np = np.random.rand(4, 4)
K_torch = torch.from_numpy(K_np)

# Generate permutation matrices
P_np = generate_permutation_matrix_scipy(K_np)
P_torch = generate_permutation_matrix_torch(K_torch.float())

print("Permutation Matrix using Scipy (numpy-based):")
print(P_np)
print("\nPermutation Matrix using PyTorch:")
print(P_torch.numpy())

# Check if the two permutation matrices are the same
comparison = np.array_equal(P_np, P_torch.numpy())
print("\nAre both permutation matrices identical?", comparison)
