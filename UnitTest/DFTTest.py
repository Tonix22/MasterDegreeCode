import numpy as np

def create_dft_matrix(N):
    # Generate a NxN grid of indices
    k, n = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Compute the DFT matrix elements directly
    F = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
    
    # Conjugate transpose (Hermitian) of the DFT matrix
    F_H = np.exp(2j * np.pi * k * n / N) / np.sqrt(N)
    
    return F, F_H

def test_dft_identity(N):
    F, F_H = create_dft_matrix(N)
    
    # Matrix multiplication of F and F_H
    identity = np.dot(F, F_H)
    print(identity)
    # Check if the result is close to the identity matrix
    return np.allclose(identity, np.eye(N))

# Example usage:
N = 2
F, F_H = create_dft_matrix(N)
#print("DFT Matrix F:\n", F)
#print("\nConjugate Hermitian F_H:\n", F_H)

# Test if F * F_H is the identity matrix
test_result = test_dft_identity(N)
#print("\nIs F * F_H approximately the identity matrix? ", test_result)
