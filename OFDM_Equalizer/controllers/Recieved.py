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
import crcmod.predefined
import time
from NearMl import near_ml
from scipy.linalg import qr as scipy_qr

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"tools")
sys.path.insert(0, main_path+"conf")
from utils import vector_to_pandas, get_date_string,get_time_string, convert_to_path
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP, GOLDEN_ACTIVE,GOLDEN_SAVE_RESULTS,GOLDEN_DATA_RATIO

class RX(Dataset):
    def __init__(self,constelation,bitstype,load,device):
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
        self.device = device
        self.load = load # "Complete", "Alphabet", "DFT_spreading"
        if self.load == "DFT_spreading":
            self.F,self.F_H   = self.create_dft_matrix(48)
            self.FQ,self.FQH  = self.create_dft_matrix(constelation)
            self.F_Tensor     = torch.tensor(self.F,device=self.device)
            self.F_H_Tensor   = torch.tensor(self.F_H,device=self.device)
        #Collapse with channel
        self.Generate()
        self.power_factor = np.sum(np.abs(self.H)**2)/np.size(self.H) # Power of complex matrix H
        
    def create_dft_matrix(self,N):
        # Generate a NxN grid of indices
        k, n = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        
        # Compute the DFT matrix elements directly
        F = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
        
        # Conjugate transpose (Hermitian) of the DFT matrix
        F_H = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
        
        return F, F_H
    
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
        chann_real = torch.from_numpy(H_idx.real).to(torch.float64)
        chann_imag = torch.from_numpy(H_idx.imag).to(torch.float64)     
        #Final tensor (48,48,2) of two channels
        chann_tensor = torch.cat((chann_real.unsqueeze(-1), chann_imag.unsqueeze(-1)), dim=2)
      
        #Tx part
        tx_tensor = None
        
        if(self.load == "Alphabet"): #natural language procesing
            tx_tensor      = torch.tensor(self.Qsym.bits[:,idx],dtype=torch.int64).squeeze()
            tx_tensor[0]   = 2 # sos
            tx_tensor[-1]  = 3 # eos
        
        if(self.load == "Complete" or self.load == "DFT_spreading"): #Imaginary and real parts
            tx_tensor = torch.tensor(self.Qsym.GroundTruth[:,idx]).squeeze().to(torch.complex128)
            
        return chann_tensor,tx_tensor
    
    
class Rx_loader():
    def __init__(self,batch_size,QAM,load,internaldevice):
        self.data    = RX(QAM,"Unit_Pow",load,internaldevice)
        # Define the split ratios (training, validation, testing)
        train_ratio = 0.6
        val_ratio   = 0.3
        if GOLDEN_ACTIVE :
            print("WARNING GOLDEN ACTIVE GOLDEN AND NOT FOR NN")
            train_ratio = GOLDEN_DATA_RATIO
            val_ratio   = GOLDEN_DATA_RATIO
        # Calculate the number of samples in each set
        train_size = int(train_ratio * len(self.data))
        val_size   = int(val_ratio * len(self.data))
        test_size  = len(self.data) - train_size - val_size
        # Split the dataset
        # set the seed for reproducibility
        torch.manual_seed(0)
        train_set, val_set, test_set = random_split(self.data, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=16)
        self.val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,num_workers=16)
        self.test_loader  = DataLoader(self.data,  batch_size=batch_size, shuffle=False,num_workers=16)
        self.test_set     = test_set
        self.batch_size   = batch_size
        #Telecom stuff
        self.SNR_db   = 35 #init SNR by default
        #used to calculate BER
        self.BER      = 0
        self.errors   = 0  #Errror per bit
        self.bits_num = 0  # Total number of bits
        self.BER_list = []
        self.crc_func = crcmod.predefined.mkCrcFun('crc-16')
        self.bad_block = 0
        self.BLER      = 0 # Block erro rate
        self.total_blocks = 0
        self.BLER_list = []
        #for average time
        self.frame = 0
        self.avg_time = 0
        self.chunk_time = 0
    
    #for internal use in the predict section of lightning
    def BER_cal(self,x_hat,x,norm=False):
        
        for n in range(x_hat.shape[0]):
            rx     = x_hat[n].cpu().detach().numpy()
            rxbits = self.data.Qsym.Demod(rx,norm=norm)
            rxbits = np.unpackbits(np.expand_dims(rxbits.astype(np.uint8),axis=1),axis=1)
            rxbits = rxbits[:,-self.data.bitsframe:]
            
            txbits = self.data.Qsym.Demod(x[n].cpu().detach().numpy(),norm=norm)
            txbits = np.unpackbits(np.expand_dims(txbits.astype(np.uint8),axis=1),axis=1)
            txbits = txbits[:,-self.data.bitsframe:]
            self.errors+=np.sum(rxbits!=txbits)
            self.bits_num+=(rxbits.size)
            
            #Block ERROR RATE
            tx_crc = self.crc_func(txbits.tobytes())
            rx_crc = self.crc_func(rxbits.tobytes())
            # Check if the Blocks are equal
            if(tx_crc != rx_crc):
                self.bad_block +=1
                
        self.total_blocks+= float(x_hat.shape[0])
        #calculate BER
        self.BER  = self.errors/self.bits_num
        self.BLER = self.bad_block/self.total_blocks
            
    
    #this is call by final API  
    def SNR_BER_TEST(self,trainer,path):
        
        for n in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP*-1):
            self.errors       = 0
            self.BER          = 0
            self.bad_block    = 0
            self.BLER         = 0
            self.bits_num     = 0
            self.total_blocks = 0
            self.SNR_db       = n 
            
            trainer.predict(self)
            
            self.BLER_list.append(self.BLER)
            self.BER_list.append(self.BER)
            if(self.SNR_db == 5):
                print("SNR:{} BER:{} BLER:{} avg_time:{:.2e}".format(self.SNR_db,self.BER,self.BLER,self.avg_time))
            else:
                print("SNR:{} BER:{} BLER:{}".format(self.SNR_db,self.BER,self.BLER))
                
        if (GOLDEN_SAVE_RESULTS) :
            
            pathBER  = convert_to_path(path+"_BER_"+get_date_string())
            pathBLER = convert_to_path(path+"_BLER_"+get_date_string())
            
            vector_to_pandas("{}_{}.csv".format(self.data.load,get_time_string()),self.BER_list,  path = pathBER)
            vector_to_pandas("{}_{}.csv".format(self.data.load,get_time_string()),self.BLER_list, path = pathBLER)
        
            
    def Get_Y(self,H,x,conj=False,noise_activ = True):
        Y = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(self.batch_size):
            #0 real,1 imag
            h = torch.complex(H[i,0,:, :],H[i,1,:, :])
            Y[i] = h@x[i]
            if(noise_activ == True):
                z = Y[i]
                # Signal Power
                Ps = torch.mean(torch.abs(z)**2)
                # Noise power
                Pn = Ps / (10**(self.SNR_db/10))
                # Generate noise
                noise = torch.sqrt(Pn/2)* (torch.randn(self.data.sym_no).to(self.device) + 1j*torch.randn(self.data.sym_no).to(self.device))
                
                if self.data.load == "DFT_spreading":
                    Y[i] = self.data.F_H_Tensor@(z+noise) # u = Gs+w
                else:
                    Y[i] = z+noise
            
            if(conj == True):
                Y[i] = (h.conj().resolve_conj()).T@Y[i]
             
        return Y
    
    def MSE_X(self,chann,Y):
        x_mse = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            if self.data.load == "DFT_spreading":
                H = self.data.F_H_Tensor@H@self.data.F_Tensor
            H_H = H.conj().resolve_conj().T
            x_mse[i] = torch.linalg.inv(H_H@H)@H_H@Y[i]
            
            if self.data.load == "DFT_spreading":
                x_mse[i] = self.data.F_Tensor@x_mse[i]
        return x_mse
    
    def LMSE_X(self,chann,Y):
        x_lmse = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            
            if self.data.load == "DFT_spreading":
                H = self.data.F_H_Tensor@H@self.data.F_Tensor
            #Signal power 
            Ps = torch.mean(torch.abs(Y)**2)
            # Noise power
            Pn = Ps / (10**(self.SNR_db/10))
            H_H = H.conj().resolve_conj().T
            x_lmse[i] = torch.linalg.inv(H_H@H+torch.eye(48).to(self.device)*Pn)@H_H@Y[i]
            
            if self.data.load == "DFT_spreading":
                x_lmse[i] = self.data.F_Tensor@x_lmse[i]
        return x_lmse
    
    
    def OSIC_X(self,chann,Y):
        conste = self.data.Qsym.QAM_N_arr
        index  = np.arange(48)
        x_osic = torch.zeros((self.batch_size,48),dtype=torch.complex128).to(self.device)
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            
            if self.data.load == "DFT_spreading":
                H = self.data.F_H_Tensor@H@self.data.F_Tensor

            x_osic[i] = self.osic_detection(H,Y[i])
            
            if self.data.load == "DFT_spreading":
                x_osic[i] = self.data.F_Tensor@x_osic[i]
            
        return x_osic

    def osic_detection(self,H, y):
        dtype = H.dtype
        device = H.device
        H = H.to(device, dtype)
        y = y.to(device, dtype)
        
        # Step 1: QR decomposition of H
        Q, R = torch.linalg.qr(H)
        
        # Step 2: Transform the received vector using the transpose conjugate of Q
        v_tilde = torch.matmul(Q.conj().transpose(-2, -1), y)
        
        # Step 3: Initialize the estimated signal vector
        N = H.size(1)
        x_est = torch.zeros(N, dtype=dtype, device=device)
        
        # Step 4: Backward substitution for interference cancellation
        for i in reversed(range(N)):
            sum_Rx = torch.matmul(R[i, i+1:N], x_est[i+1:N])
            x_est[i] = (v_tilde[i] - sum_Rx) / R[i, i]
        
        return x_est

    def NML_X(self,chann,Y):
        conste = self.data.Qsym.QAM_N_arr_tensor.cpu().numpy()[:, np.newaxis]
        P      = np.arange(1, 49)[:, np.newaxis]
        x_nml  = np.zeros((self.batch_size,48),dtype = np.complex128)
        
        for i in range(chann.shape[0]):
            H = torch.complex(chann[i,0,:, :],chann[i,1,:, :])
            
            if self.data.load == "DFT_spreading":
                G    = (self.data.F_H_Tensor@H@self.data.F_Tensor)
                Q, R = torch.linalg.qr(G)
                v_tilde = torch.matmul(Q.conj().transpose(-2, -1), Y[i])
                x_nml[i] = near_ml(v_tilde.cpu().numpy()[:, np.newaxis],
                                   R.cpu().numpy(),
                                   conste,
                                   P)
                
            else:   
                x_nml[i] = near_ml(Y[i].cpu().numpy()[:, np.newaxis],
                                   H.cpu().numpy(),
                                   conste,
                                   P)
             
            if self.data.load == "DFT_spreading":
                x_nml[i] = self.data.F@x_nml[i]
            
        return torch.from_numpy(x_nml).to(self.device)

    def OSIC_Det(self,Y, H, conste, index):
        rows, cols = H.shape
        dim = len(conste)

        x_hat = torch.zeros(rows, dtype=torch.complex128)
        simbolEstimate  = torch.zeros(rows, dtype=torch.complex128)
        distances = torch.zeros(dim)

        for k in range(cols):
            i = cols - (k + 1)
            # Y over diagonal
            a_est = Y[i] / H[i, i]
            # Measure distance between estim and constelation points
            distances = torch.abs(a_est - conste)**2
            # Return indeces from lowest to highest
            pos = torch.argsort(distances)
            simbolEstimate[i] = conste[pos[0]]
            # Remove interference with symbol estimate times i-th column
            Y -= simbolEstimate[i] * H[:, i]
            x_hat[i] = simbolEstimate[i]            

        return x_hat
    
    def Chann_diag(self,chann):
        # Extract the diagonal of each matrix in the tensor
        diagonals = []
        H = torch.complex(chann[:,0,:, :],chann[:,1,:, :])
        for i in range(H.shape[0]):
            matrix = H[i]
            diagonal = torch.diagonal(matrix)
            diagonals.append(diagonal)

        # Stack the diagonals into a new tensor
        diagonal_tensor = torch.stack(diagonals)
        return diagonal_tensor
    
    def Chann_diag_complex(self,chann):
        # Extract the diagonal of each matrix in the tensor
        diagonals = []
        for i in range(chann.shape[0]):
            matrix   = chann[i]
            diagonal = torch.diagonal(matrix)
            diagonals.append(diagonal)

        # Stack the diagonals into a new tensor
        diagonal_tensor = torch.stack(diagonals)
        return diagonal_tensor
    
    def ZERO_X(self,chann,Y):
        real_part = torch.diagonal(chann[:, 0, :, :], dim1=-2, dim2=-1)  # Shape: [100, 48]
        imag_part = torch.diagonal(chann[:, 1, :, :], dim1=-2, dim2=-1)  # Shape: [100, 48]
        complex_diagonal = torch.complex(real_part, imag_part)  # Shape: [100, 48]
        x_hat = Y/complex_diagonal #ZERO FORCING equalizer
        return x_hat
    
        #90% Confidence level
    def filter_z_score(self,data, threshold=1.645):
        # Calculate the z-score for each data point in the batch
        z_scores = (data - torch.mean(data, dim=1, keepdim=True)) / torch.std(data, dim=1, keepdim=True)

        # Identify the outlier data points
        outlier_mask = torch.abs(z_scores) > threshold

        # Compute the number of data points don't pass the z-score threshold
        valid_count = torch.sum(~outlier_mask, dim=1)

        indices     = torch.nonzero(torch.eq(valid_count, 48)).squeeze()
        # If batch there are not odiagonal_tensor
        # Filter out the invalid sequences from the batch
        valid_data = data[indices]
        

        return valid_data, indices

    def filter_z_score_matrix(self,batch_tensor, threshold=1.645):
        diagonal_tensor = self.Chann_diag(batch_tensor)    
        
        # Compute the mean and standard deviation of each batch
        mean = torch.mean(diagonal_tensor, dim=1, keepdim=True)
        std  = torch.std(diagonal_tensor, dim=1, keepdim=True)
        
        # Compute the z-score for each element in the tensor
        zscore_tensor = (diagonal_tensor - mean) / std.clamp(min=1e-6)
        
        # Create a mask indicating which elements have a z-score below the threshold
        outlier_mask = zscore_tensor.abs() > threshold
        
        # Compute the number of data points don't pass the z-score threshold
        valid_count = torch.sum(~outlier_mask, dim=1)

        indices     = torch.nonzero(torch.eq(valid_count, 48)).squeeze()
        # If batch there are not odiagonal_tensor
        # Filter out the invalid sequences from the batch
        valid_data = batch_tensor[indices]
        
        # Return the filtered tensor
        return valid_data,indices
            
    def filter_chann_diag_z_score(self,batch_tensor, threshold=9):
        # Compute the mean and standard deviation of each batch
        mean = torch.mean(batch_tensor, dim=[1, 2], keepdim=True)
        std = torch.std(batch_tensor, dim=[1, 2], keepdim=True)
        
        # Compute the z-score for each element in the tensor
        zscore_tensor   = (batch_tensor - mean) / std.clamp(min=1e-6)
        diagonal_tensor = self.Chann_diag_complex(zscore_tensor)  
    
        # Create a mask indicating which elements have a z-score below the threshold
        zscore_mask = diagonal_tensor.abs() < threshold
        
        # Compute the number of data points don't pass the z-score threshold
        valid_count = torch.sum(zscore_mask, dim=1)

        indices     = torch.nonzero(torch.eq(valid_count, 48)).squeeze()
        # If batch there are not odiagonal_tensor
        # Filter out the invalid sequences from the batch
        valid_data = batch_tensor[indices]
        
        # Return the filtered tensor
        return valid_data,indices
    
    def start_clock(self):
                
        if(self.SNR_db <= 15):
            self.frame = 0
            self.start_time = time.time() #init time stamp

    def stop_clock(self,blocks):
        if(self.SNR_db <= 15):
            end_time = time.time() # stop watch
            execution_time = (end_time - self.start_time)/blocks #exec 
            self.chunk_time += execution_time
            self.frame +=1
            self.avg_time = self.chunk_time/self.frame