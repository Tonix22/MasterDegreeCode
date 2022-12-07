from Channel import Channel
from QAM_mod import QAM
import numpy as np
from  math import log2

class RX():
    def __init__(self,constelation,bitstype):
        #Channel Data set is of size 48
        self.bitsframe = int(log2(constelation))
        self.sym_no = 48
        self.total  = 20000
        self.LOS    = Channel()
        self.NLOS   = Channel(LOS=False)
        self.Qsym   = QAM(self.sym_no * self.total,constelation=constelation,cont_type= bitstype) # all symbols per realization
        #Each column is a realization 
        self.Qsym.GroundTruth = np.reshape(self.Qsym.GroundTruth,(self.sym_no,self.total,1))
        self.Qsym.r    = np.reshape(self.Qsym.r,(self.sym_no,self.total,1))
        self.Qsym.bits = np.reshape(self.Qsym.bits,(self.sym_no,self.total,1))
        
        self.H = np.empty((self.sym_no,self.sym_no,self.total), dtype=self.LOS.con_list.dtype)
        self.H[:,:,0::2] = self.LOS.con_list
        self.H[:,:,1::2] = self.NLOS.con_list
        #Collapse with channel
        self.Generate()
    
    def Generate(self):
        #Swtiching verison
        #We mixed LOS and NLOS
        for n in range (0,self.total):
            self.Qsym.r[:,n] = self.H[:,:,n]@self.Qsym.GroundTruth[:,n]             
    
    def AWGN(self,SNR):
        for n in range (0,self.total):
            self.Qsym.r[:,n] += np.sqrt((10**(-SNR/10))/2)*(np.random.randn(self.sym_no,1) + 1j*np.random.randn(self.sym_no,1))
