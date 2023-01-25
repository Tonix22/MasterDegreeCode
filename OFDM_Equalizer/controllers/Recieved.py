from Channel import Channel
from QAM_mod import QAM
import numpy as np
from  math import log2
from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split

class RX(Dataset):
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
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.power_factor = np.sum(np.abs(self.H)**2)/np.size(self.H) # Power of complex matrix H
    
    def Generate(self):
        #Swtiching verison
        #We mixed LOS and NLOS
        for n in range (0,self.total):
            self.Qsym.r[:,n] = self.H[:,:,n]@self.Qsym.GroundTruth[:,n]             
    
    def AWGN(self,SNR):
        #Assume noise singal is 1
        #Noise power
        Pn = 1 / (10**(SNR/10))
        for n in range (0,self.total):
            #Generate noise
            noise = sqrt(Pn/2)* (np.random.randn(self.sym_no,1) + 1j*np.random.randn(self.sym_no,1))
            Y     = self.H[:,:,n]@self.Qsym.GroundTruth[:,n]
            self.Qsym.r[:,n] = Y+noise
            
    #Data loader functions
    def __len__(self):
        return self.total
    
    #Call AWGN befor call asigment to make sure you have the noise in data
    def __getitem__(self, idx):
        #Normalize tensor
        norm_H       = self.H[:,:,idx]/self.power_factor
        #norm_H        = self.H[:,:,idx]
        #extract both parts
        chann_real   = torch.tensor(norm_H.real).to(self.device).unsqueeze(-1)
        chann_imag   = torch.tensor(norm_H.imag).to(self.device).unsqueeze(-1)        
        #Final tensor (48,48,2) of two channels
        chann_tensor = torch.cat((chann_real, chann_imag), dim=2)
        #tx_tensor    = torch.tensor(self.Qsym.GroundTruth[:,idx]).squeeze().to(torch.complex64).to(self.device)
        tx_tensor    = torch.tensor(self.Qsym.bits[:,idx],dtype=torch.int64).squeeze()
        tx_tensor[0]     = 2 # sos
        tx_tensor[-1]    = 3 # eos
        return chann_tensor,tx_tensor
    
    
class Rx_loader(object):
    def __init__(self,batch_size):
        self.data    = RX(16,"Unit_Pow")
        # Define the split ratios (training, validation, testing)
        train_ratio = 0.6
        val_ratio   = 0.2
        test_ratio  = 0.2
        # Calculate the number of samples in each set
        train_size = int(train_ratio * len(self.data))
        val_size   = int(val_ratio * len(self.data))
        test_size  = len(self.data) - train_size - val_size
        # Split the dataset
        train_set, val_set, test_set = random_split(self.data, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        self.val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
        
