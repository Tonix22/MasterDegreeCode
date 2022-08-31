from cmath import pi
from xml.etree.ElementTree import PI
import numpy as np
import pandas as pd
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
from numpy.core.defchararray import greater
from numpy.core.fromnumeric import ptp
from numpy.lib.type_check import real


PLOT_DISTRO_H = False

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

class Sig_Processing:
    def __init__(self):
        self.Reconstruct = None
        self.I           = None
        self.Top_matrix  = None
        self.Diff_var    = 0
        self.RMS         = 0
        self.H           = None
        self.R           = None
        self.tool        = None

    def ReconstructSignal(self,plot=False):
        self.Reconstruct  = self.Top_matrix @ self.I
        if(plot == True):
            self.Plot_realization(self.Reconstruct)
        self.Reconstruct_Err()

    def Reconstruct_Err(self):
        Diff_vect     = self.R - self.Reconstruct
        self.Diff_var = abs(np.var(Diff_vect))
        self.RMS      = sqrt(np.mean(np.square(Diff_vect)))

    def Set_Toeplitz_matrix(self,realization):
        self.Top_matrix = self.tool.GenerateToeplitz(self.H[realization],len(self.H[realization]))

    def Inverse_SVD(self):
        TOP_inv = np.linalg.pinv(self.Top_matrix)
        self.I  = TOP_inv @ self.R

    def PseudoInverse(self):
        realization_t = self.Top_matrix.T
        Corr          = realization_t @ self.Top_matrix
        Res           = realization_t @ self.R
        self.I        = np.linalg.inv(Corr) @ Res

    def Plot_realization(self,realization):
        plt_y = np.linspace(1, len(realization), num=len(realization))
        fig, ax = plt.subplots()
        ax.stem(plt_y,realization, markerfmt=' ')
        plt.ylabel('Amplitud') #set the label for y axis
        plt.xlabel('Sample') #set the label for x-axis
        plt.show()

class Test(Sig_Processing):
    def __init__(self,items):
        self.N = 16
        self.Realizations = items
        
        mu = 0
        sigma = pi/10
        self.H = np.zeros((self.Realizations,self.N))
        
        for M in range (0,self.Realizations):
            self.H[M] = np.random.normal(mu,sigma,self.N)
        
            if PLOT_DISTRO_H:
                count, bins, ignored = plt.hist(self.H[M], self.N, density=True)
                plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *

                    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),

                linewidth=2, color='r')
                plt.show()
        
        
        self.R = np.zeros((2*self.N-1,1))
        self.R[self.N-1] = 1
        columns =[]

        for i in range (0,self.N):
            columns.append('X'+str(i))

        for i in range (0,2*self.N-1):
            columns.append('R'+str(i))
            
        self.df          = pd.DataFrame(columns = columns)
        self.tool        = Math_toolbox()

    def SigErrThreshold(self):
        return (self.RMS < .1 and self.Diff_var < .05)

    def DataFrameAppend(self):
        X_in  = self.R.T @ self.Top_matrix #impulse response
        R_out = self.Reconstruct.T
        db_res = np.hstack((X_in,R_out))
        self.df = self.df.append(pd.DataFrame(db_res, columns=self.df.columns), ignore_index=True)
        
        db_res  = np.hstack((np.ravel(self.I),np.ravel(self.Reconstruct)))
        db_res  = np.append(db_res, (self.RMS,self.Diff_var))
        data_to_append = {}
        for i in range(len(self.df.columns)):
            data_to_append[self.df.columns[i]] = db_res[i]
        self.df = self.df.append(data_to_append, ignore_index = True)
        

    def SaveFrameInCSV(self):
        import os
        if os.path.exists('Realizations.csv'):
            os.remove('Realizations.csv')
        self.df.to_csv('Realizations.csv')


Generator = Test(10000)

for n in range(0,Generator.Realizations):
    #Take realization N and generate it toeplitz matrix
    Generator.Set_Toeplitz_matrix(n)
    # Pseudo Inverse H and isolate for I
    Generator.PseudoInverse() 
    # H*I=R, calculates error from I got by pseudo inverse
    Generator.ReconstructSignal()

    if(Generator.SigErrThreshold()):
        #Generator.Plot_realization(Generator.Reconstruct)
        Generator.DataFrameAppend()
        
#Generator.SaveFrameInCSV()

"""
#0, 0, 0 ,0,0,-6, 2,1,-2,-1,3,4
#12,-4,-2,4,2,-6,-8,0,0, 0 ,0,0
H = np.array([0, 0, 0 ,0,0,-6, 2,1,-2,-1,3,4])
X = np.vstack(np.array([12,-4,-2,4,2,-6,-8,0,0, 0 ,0,0]))
tool = Math_toolbox()
H = tool.GenerateToeplitz(H,len(H))
OUT = H@X
tool.print_latex_format(OUT)
"""