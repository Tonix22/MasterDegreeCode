from turtle import down
from Channel import Channel
from QPSK import QPSK
import numpy as np

class RX():
    def __init__(self):
        #Channel Data set is of size 48
        self.bitsframe = 2
        self.sym_no = 48
        self.total  = 20000
        self.LOS    = Channel()
        self.NLOS   = Channel(LOS=False)
        self.Qsym   = QPSK(self.sym_no * self.total) # all symbols per realization
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

"""
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
# generating 2D matrix of random numbers between 10 and 100
data = RX()

SNR = 20
H   = np.matrix(data.H[:,:,0])
Y   = data.Qsym.r[:,0]

inside = np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))
inv    = np.linalg.inv(inside)@H.H@Y

#diag = np.diagonal(inv,offset=0)
#row_sum = np.sum(inv[0])
#print(diag[0])
#print(row_sum)
hm = sn.heatmap(data = inside.real)

# Heatmap Display
plt.savefig('plots/heatmap_real_inverted.jpg')
"""