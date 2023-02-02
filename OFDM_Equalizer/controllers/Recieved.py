import os
import sys
from Channel import Channel
from QAM_mod import QAM
import numpy as np
from  math import log2
from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"tools")
sys.path.insert(0, main_path+"conf")
from utils import vector_to_pandas, get_time_string
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP

class RX(Dataset):
    def __init__(self,constelation,bitstype,load):
        #Channel Data set is of size 48
        self.bitsframe = int(log2(constelation))
        self.sym_no = 48
        self.total  = 20000
        self.LOS    = Channel()
        self.NLOS   = Channel(LOS=False)
        self.Qsym   = QAM(self.sym_no * self.total,constelation=constelation,cont_type= bitstype,load_type=load) # all symbols per realization
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
        #load type
        self.load = load # "Complete", "Alphabet"
    
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
        #Channel part
        
        #Normalize tensor
        H_idx       = self.H[:,:,idx]
        #extract both parts
        chann_real   = torch.tensor(H_idx.real).to(torch.float64).to(self.device).unsqueeze(-1)
        chann_imag   = torch.tensor(H_idx.imag).to(torch.float64).to(self.device).unsqueeze(-1)        
        #Final tensor (48,48,2) of two channels
        chann_tensor = torch.cat((chann_real, chann_imag), dim=2)
      
        #Tx part
        tx_tensor = None
        
        if(self.load == "Alphabet"): #natural language procesing
            tx_tensor      = torch.tensor(self.Qsym.bits[:,idx],dtype=torch.int64).squeeze()
            tx_tensor[0]   = 2 # sos
            tx_tensor[-1]  = 3 # eos
        
        if(self.load == "Complete"): #Imaginary and real parts
            tx_tensor = torch.tensor(self.Qsym.GroundTruth[:,idx]).squeeze().to(torch.complex128).to(self.device)
            
        return chann_tensor,tx_tensor
    
    
class Rx_loader(object):
    def __init__(self,batch_size,QAM,load):
        self.data    = RX(QAM,"Unit_Pow",load)
        # Define the split ratios (training, validation, testing)
        train_ratio = 0.6
        val_ratio   = 0.2
        # Calculate the number of samples in each set
        train_size = int(train_ratio * len(self.data))
        val_size   = int(val_ratio * len(self.data))
        test_size  = len(self.data) - train_size - val_size
        # Split the dataset
        # set the seed for reproducibility
        torch.manual_seed(0)
        train_set, val_set, test_set = random_split(self.data, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        self.val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
        self.test_set     = test_set
        self.batch_size   = batch_size
        #Telecom stuff
        self.SNR_db   = 35 #init SNR by default
        #used to calculate BER
        self.BER      = 0
        self.errors   = 0  #Errror per bit
        self.bits_num = 0  # Total number of bits
        self.BER_list = []
    
    #for internal use in the predict section of lightning
    def SNR_calc(self,x_hat,x):
        for n in range(self.batch_size):
            rx     = x_hat[n].cpu().detach().numpy()
            rxbits = self.data.Qsym.Demod(rx)
            rxbits = np.unpackbits(np.expand_dims(rxbits.astype(np.uint8),axis=1),axis=1)
            rxbits = rxbits[:,-self.data.bitsframe:]
            
            txbits = self.data.Qsym.Demod(x[n].cpu().detach().numpy())
            txbits = np.unpackbits(np.expand_dims(txbits.astype(np.uint8),axis=1),axis=1)
            txbits = txbits[:,-self.data.bitsframe:]
            
            self.errors+=np.sum(rxbits!=txbits)
            self.bits_num+=(rxbits.size)
        
        #calculate BER
        self.BER = self.errors/self.bits_num
    
    #this is call by final API  
    def SNR_BER_TEST(self,trainer,csv_name):
        
        for n in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP*-1):
            self.errors   = 0
            self.BER      = 0
            self.bits_num = 0
            self.SNR_db   = n 
            trainer.predict(self)
            self.BER_list.append(self.BER)
            print("SNR:{} BER:{}".format(self.SNR_db,self.BER))
        
        vector_to_pandas("BER_{}.csv".format(csv_name),self.BER_list,path="./BER_csv")
            
    def Get_Y(self,H,x,conj=False):
        Y = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(self.batch_size):
            #0 real,1 imag
            h = torch.complex(H[i,0,:, :],H[i,1,:, :])
            z = h@x[i]
            # Signal Power
            Ps = torch.mean(torch.abs(z)**2)
            # Noise power
            Pn = Ps / (10**(self.SNR_db/10))
            # Generate noise
            noise = torch.sqrt(Pn/2)* (torch.randn(self.data.sym_no).to(self.device) + 1j*torch.randn(self.data.sym_no).to(self.device))
            Y[i] = z+noise
            if(conj == True):
                Y[i] = (h.conj().resolve_conj()).T@Y[i]
             
        return Y
    
    def MSE_X(self,chann,Y):
        x_mse = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            H_H = H.conj().resolve_conj().T
            x_mse[i] = torch.linalg.inv(H_H@H)@H_H@Y[i]
        return x_mse
    
    def LMSE_X(self,chann,Y):
        x_lmse = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            H_H = H.conj().resolve_conj().T
            #Signal power 
            Ps = torch.mean(torch.abs(Y)**2)
            # Noise power
            Pn = Ps / (10**(self.SNR_db/10))
            x_lmse[i] = torch.linalg.inv(H_H@H+torch.eye(48).to(self.device)*Pn)@H_H@Y[i]
        return x_lmse
        
