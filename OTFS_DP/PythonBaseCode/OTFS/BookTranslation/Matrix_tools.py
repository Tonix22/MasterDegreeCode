import numpy as np

#Row-column permutation matrix
def Permutation_matrix(N,M):
    P=np.zeros((N*M,N*M));    
    for j in range (0,N):
        for i in range(0,M):
            E = np.zeros((M,N))
            E[i,j]=1
            P[(j-1)*M+1:j*M,(i-1)*N+1:i*N]=E
    return P