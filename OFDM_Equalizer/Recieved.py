from turtle import down
from Channel import Channel
from QPSK import QPSK
import numpy as np

class RX():
    def __init__(self):
        #Channel Data set is of size 48
        self.sym_no = 48
        self.total  = 20000
        self.LOS    = Channel()
        self.NLOS   = Channel(LOS=False)
        self.Qsym   = QPSK(self.sym_no * self.total) # all symbols per realization
        #Each column is a realization 
        self.Qsym.GroundTruth = np.reshape(self.Qsym.GroundTruth,(self.sym_no,self.total,1))
        self.Qsym.r    = np.reshape(self.Qsym.r,(self.sym_no,self.total,1))
        self.Qsym.bits = np.reshape(self.Qsym.bits,(self.sym_no,self.total,1))
        
        #Collapse with channel
        self.Generate()
    
    def Generate(self):
        #Swtiching verison
        #We mixed LOS and NLOS
        LOS_cnt  = 0
        NLOS_cnt = 0
        for n in range (0,self.total-1):
            #Get realization
            if(n&1):
                self.Qsym.r[:,n] = self.LOS[LOS_cnt] @ self.Qsym.GroundTruth[:,n]
                LOS_cnt+=1
            else:
                self.Qsym.r[:,n] = self.NLOS[NLOS_cnt] @ self.Qsym.GroundTruth[:,n]
                NLOS_cnt+=1               
    
    def AWGN(self,SNR):
        for n in range (0,self.total-1):
            self.Qsym.r[:,n] = self.Qsym.r[:,n] + np.sqrt(10**(-SNR/20))*(np.random.randn(self.sym_no,1) + 1j*np.random.randn(self.sym_no,1))
