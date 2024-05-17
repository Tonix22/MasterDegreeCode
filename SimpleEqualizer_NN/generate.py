from cmath import pi
from xml.etree.ElementTree import PI
import numpy as np
import pandas as pd
from math import sqrt
from math import pi
from Toeplitz import Math_toolbox
import matplotlib.pyplot as plt
from config import *

from numpy.core.defchararray import greater
from numpy.core.fromnumeric import ptp
from numpy.lib.type_check import real



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
        # 16 Taps
        self.N = 16
        # Number of random realizations
        self.Realizations = items
        # Gaussian noise parameters
        SNR   = -10
        mu    = 0
        sigma = 1/(10**(SNR/10))
        print(sigma)
        
        #Channel vector
        self.H = np.zeros((self.Realizations,self.N))
        
        #Generating Guassian noise for each realization
        for M in range (0,self.Realizations):
            self.H[M] = np.random.normal(mu,sigma,self.N)
        
            if PLOT_DISTRO_H:
                count, bins, ignored = plt.hist(self.H[M], self.N, density=True)
                plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - mu)**2 / (2 * sigma**2)),
                    linewidth=2, color='r')
                plt.show()
        
        #DELTA generation vector calles as R
        self.R = np.zeros((2*self.N-1,1))
        self.R[self.N-1] = 1
        
        #Generate data frame data
        columns =[]
        for i in range (0,self.N):
            columns.append('X'+str(i))

        for i in range (0,2*self.N-1):
            columns.append('R'+str(i))
            
        self.df          = pd.DataFrame(columns = columns)
        
        #Custom math toolbox for Toeplizt matrix
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

for idx in range(0,Generator.Realizations):
    #Take realization N and generate it toeplitz matrix
    Generator.Set_Toeplitz_matrix(idx)
    # Pseudo Inverse H and isolate for I
    Generator.PseudoInverse() 
    # H*I=R, calculates error from I got by pseudo inverse
    Generator.ReconstructSignal()

    #if(Generator.SigErrThreshold()):
    #Generator.Plot_realization(Generator.Reconstruct)
    Generator.Plot_realization(Generator.Top_matrix.T@Generator.R)
    exit()
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