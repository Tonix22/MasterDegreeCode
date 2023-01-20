from scipy.linalg import eig
from scipy.linalg import eigh
import numpy as np
# Original matrix
A = np.random.rand(100, 100)

# Compute the Gram matrix of A
G = np.dot(A, A.T)

# Compute the eigenvectors of the Gram matrix
eigenvalues, eigenvectors = eigh(G)

# Sort the eigenvectors by decreasing eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Select the first k eigenvectors
k = 10
eigenvectors_k = eigenvectors[:, :k]

# Compute the compressed matrix
A_compressed = np.dot(A, eigenvectors_k)
print(A_compressed.shape)