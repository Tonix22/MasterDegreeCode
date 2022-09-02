import numpy as np

class Math_toolbox():
    def __init__(self):
        pass

    def GenerateToeplitz(self,h_vect,N):
        toeplitz_matrix = np.zeros((2*N-1,N))
        toeplitz_row = 0
        for i in range(0,N):
            toeplitz_col = 0
            for j in range(i,-1,-1):
                toeplitz_matrix[toeplitz_row][toeplitz_col]=h_vect[j]
                toeplitz_col+=1
            toeplitz_row+=1
            
        for i in range(N-2,-1,-1):
            toeplitz_col = -1*i-1
            offset = toeplitz_col
            for j in range (i,-1,-1):
                toeplitz_matrix[toeplitz_row][toeplitz_col]=h_vect[j+offset]
                toeplitz_col+=1
            toeplitz_row+=1
        
        return toeplitz_matrix
    
    def print_latex_format(self,Matrix):
        print("\\begin{pmatrix}")
        for n in range(0,len(Matrix)):
            for m in range(0,len(Matrix[n])):
                if(m < len(Matrix[n])-1):
                    print(str(Matrix[n][m])+" & ",end='')
                else:
                    print(str(Matrix[n][m]),end='')
            print("\\\\")

        print("\\end{pmatrix}")