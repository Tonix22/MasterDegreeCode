from mimetypes import init
from random import randint
import numpy as np
from numpy.linalg import norm 
from math import log2
from QAM_mod import QPSK
from Matrix_tools import Permutation_matrix

class Transmiter():
    def __init__(self,N=16,M=64,delta_f=15e3,QAM_symbols=4):
        self.N = N
        self.M = M
        self.P = None
        # normalized DFT matrix
        self.Fn = np.fft.fft(np.eye(N))
        self.Fn = self.Fn/norm(self.Fn)
        #subcarrier spacing
        self.delta_f = delta_f
        #block duration
        self.T = 1/self.delta_f
        #carrier frequency 
        self.fc=4e9
        #OTFS grid delay and Doppler resolution
        self.delay_resolution   = 1/(self.M*self.delta_f)
        self.Doppler_resolution = 1/(self.N*self.T)
        #QAM modulation symbols
        self.mod_size = QAM_symbols
        #Symbols per frame
        self.N_syms_per_frame = self.N*self.M
        #Number of informaton bits in one frame
        N_bits_per_frame = self.N*self.M*log2(self.mod_size)
        #QAM modulation
        self.tx_info_bits = None
        if(self.mod_size == 4):
            self.tx_info_bits = QPSK(N_bits_per_frame)
        else:
            #TODO Generik constelation or Higher QAM
            pass
        self.X = np.reshape(self.tx_info_bits,(M,N))
        #vectorized version
        self.vect_x = np.reshape(self.X.T,(N*M,1))
        
        
    #Delay Doppler transformed 
    #to delay-time with Inveze ZAK transform
    def Inverse_ZAK_method(self):
        X_tilda=self.X*self.Fn
        s = np.reshape(X_tilda,(1,self.N*self.M))
        return s
    #Permutaton matrix implement
    def Permutation_method(self):
        self.P  = Permutation_matrix(self.N,self.M)
        Im = np.eye(self.M)
        s  = self.P@np.kron(Im,self.Fn.T)@self.vect_x
        return s