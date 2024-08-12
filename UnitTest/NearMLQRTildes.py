import torch
import numpy as np

def compute_qr_decomposition_with_permutation(H):
    # Assuming H is a PyTorch tensor
    # Sort the columns of H based on their norms
    norms = torch.norm(H, dim=0)
    indices = torch.argsort(norms, descending=True)
    P = torch.eye(H.shape[1])[indices]  # Permutation matrix
    HP = torch.mm(H, P)
    Q, R = torch.linalg.qr(HP)
    return Q, R, P

def compute_v_tilde(Y, H, x, Z_D):
    Q, R, P = compute_qr_decomposition_with_permutation(H)
    # Compute v
    x_permuted = torch.mm(P.T, x)  # Apply permutation to x
    v = torch.mm(Q, torch.mm(R, x_permuted)) + Z_D
    # Compute v_tilde
    v_tilde = torch.mm(Q.T.conj(), v)
    return v_tilde, P

# Example usage
H = torch.randn(100, 100, dtype=torch.cfloat)  # Channel matrix
Y = torch.randn(100, dtype=torch.cfloat)  # Received signal
x = torch.randn(100, dtype=torch.cfloat)  # Transmitted signal
Z_D = torch.randn(100, dtype=torch.cfloat)  # Noise

v_tilde, P = compute_v_tilde(Y, H, x, Z_D)
print("v_tilde:", v_tilde)
print("Permutation matrix P:", P)
