from cmath import pi
from NN_lab import NetLabs
import torch
import torch.nn as nn
import numpy as np
from Constants import *
import pandas as pd
from  tqdm import tqdm
#import GPUtil
import matplotlib.pyplot as plot

import os 
import sys

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
test_path = main_path+"Test"

sys.path.insert(0, main_path+"tools")
sys.path.insert(0, main_path+"conf")

from utils import vector_to_pandas
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP
from utils import get_time_string

class TestNet(NetLabs):
    def __init__(self,path = None,pth_real=None,pth_imag=None,loss_type=MSE):
        super().__init__(loss_type,GOLDEN_BEST_SNR,GOLDEN_WORST_SNR,step=GOLDEN_STEP)
        self.real_imag = None
        #model loading
        if(loss_type == MSE_COMPLETE):
            self.model = self.Generate_Network_Model()
            self.model.load_state_dict(torch.load(path))
        
        if(loss_type == MSE):
            self.model_real = None
            self.model_imag = None
            self.real_imag  = REAL
            self.model_real = self.Generate_Network_Model()
            self.model_imag = self.Generate_Network_Model()
            self.model_real.load_state_dict(torch.load(pth_real))
            self.model_imag.load_state_dict(torch.load(pth_imag))
            
        #ground truth
        if(self.loss_type == CROSSENTROPY):
            self.gt = self.Get_ground_truth(self.data.Qsym.bits)
        elif(self.loss_type == MSE_COMPLETE):
            self.gt = torch.tensor(self.data.Qsym.GroundTruth).to(self.device)
        else:
            self.real_imag = BOTH
            self.gt = self.Get_ground_truth(self.data.Qsym.GroundTruth)
    
    #************************
    #*******TESTING**********
    #************************
    def TestQAM(self):
        self.loss_type=CROSSENTROPY
        
        df = pd.DataFrame()
        BER    = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,-1*self.step):
            losses = []
            self.r = self.Generate_SNR(SNR,BOTH)
            loop   = tqdm(range(0,self.data.total),desc="Progress")
            frames = self.data.total-0 #self.training_data
            errors = 0
            
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                #Ground truth are constelation bits
                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                Y      = torch.from_numpy(txbits).cuda()
                
                pred = self.model(X[0:self.data.sym_no].float(),X[self.data.sym_no:].float()) 

                loss = self.criterion(pred,Y.float())
                losses.append(loss.cpu().detach().numpy())
                #BER
                rxbits = pred.cpu().detach().numpy()
                rxbits = rxbits*3
                #rxbits = np.around(rxbits)
                rxbits = rxbits.astype(np.uint8)
                #errors += ((txbits^rxbits)&1).sum()+((txbits^rxbits)&2>>1).sum()
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    #print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            Ber_DF = errors/((self.data.bitsframe*self.data.sym_no)*frames)
            df["SNR{}".format(SNR)]= Ber_DF
            losses.clear()
            
            #Calculate BER
            BER.append(Ber_DF)
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[BOTH],get_time_string())
        df.to_csv('{}/Test_BER_{}.csv'.format(test_path,formating), header=True, index=False)
        vector_to_pandas("BER_{}.csv".format(formating),BER)
    
    def Test(self):
        df = pd.DataFrame()
        BER    = []
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,-1*self.step):
            self.r = self.Generate_SNR(SNR,BOTH)
            loop   = tqdm(range(0,self.data.total),desc="Progress")
            errors = 0
            frames = self.data.total # self.training_data
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # ground 
                
                if(self.real_imag == BOTH):
                    pred_real = self.model_real(X[0:self.data.sym_no])
                    pred_imag = self.model_imag(X[self.data.sym_no:])
                
                #BER
                if(self.loss_type == MSE):
                    #demodulate prediction data
                    real   = pred_real.cpu().detach().numpy()
                    imag   = pred_imag.cpu().detach().numpy()
                    res    = real + 1j*imag
                    rxbits = self.data.Qsym.Demod(res)
                    
                if(self.loss_type == MSE_COMPLETE):
                    real = X[0:self.data.sym_no]
                    imag = X[self.data.sym_no:]
                    pred = self.model(real,imag)
                    pred = pred.cpu().detach().numpy()
                    pred = pred[:,0]+1j*pred[:,1]
                    rxbits = self.data.Qsym.Demod(pred)
                    

                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    #print(GPUtil.showUtilization())
                    
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,BOTH,get_time_string())
        vector_to_pandas("BER_{}.csv".format(formating),BER)
        
        
class TestNet_Angle_Phase(NetLabs):
    def __init__(self,pth_angle,pth_mag,loss_type=MSE):
        super().__init__(loss_type,GOLDEN_BEST_SNR,GOLDEN_WORST_SNR,step=GOLDEN_STEP,real_imag = ANGLE)
        #Phase
        self.model_angle = self.Generate_Network_Model()
        self.model_angle.load_state_dict(torch.load(pth_angle))
        #ABS
        self.real_imag = ABS
        if(self.data.Qsym.constelation !=4):
            self.model_mag   = self.Generate_Network_Model()
            self.model_mag.load_state_dict(torch.load(pth_mag))
        
    def Test(self):
        df  = pd.DataFrame()
        BER = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,-1*self.step):
            #losses = []
            
            self.r_abs   = self.Generate_SNR(SNR,ABS)
            self.r_angle = self.Generate_SNR(SNR,ANGLE)
            
            loop   = tqdm(range(self.training_data,self.data.total),desc="Progress")
            errors = 0
            #passed = 0
            frames = self.data.total-self.training_data # self.training_data
            for i in loop:
                
                X_ang  = torch.squeeze(self.r_angle[:,i],1)  # input
                if(self.data.Qsym.constelation !=4):
                    X_abs  = torch.squeeze(self.r_abs[:,i],1) 
                #Y_angle  = torch.squeeze(self.gt_angle[:,i],1) # ground thruth
                
                pred_ang = self.model_angle(X_ang)
                if(self.data.Qsym.constelation !=4):
                    pred_abs = self.model_mag(X_abs)
                                
                #loss_ang = self.criterion(pred_ang,Y_angle.float())
                #loss_abs = self.criterion(pred_abs,Y_mag.float())
                
                #loss = (loss_ang + loss_abs)/2

                #losses.append(loss_ang.cpu().detach().numpy())
                

                pred_ang = pred_ang*pi
                theta    = pred_ang.cpu().detach().numpy()
                if(self.data.Qsym.constelation !=4):
                    radius   = pred_abs.cpu().detach().numpy()
                else:
                    radius = .7
                    
                res      = radius*np.exp(1j*theta)
                rxbits   = self.data.Qsym.Demod(res)

                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    #print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            #df["SNR{}".format(SNR)]= losses
            #losses.clear()
            
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[BOTH],get_time_string())
        #df.to_csv('{}/Testing_Loss_{}.csv'.format(test_path,formating), header=True, index=False)
        
        vector_to_pandas("BER_{}.csv".format(formating),BER)

class TestNet_BCE(NetLabs):
    def __init__(self,pth):
        loss_type = BCE
        super().__init__(loss_type,GOLDEN_BEST_SNR,GOLDEN_WORST_SNR,step=GOLDEN_STEP,real_imag = FOUR)
        self.model = self.Generate_Network_Model()
        self.model.load_state_dict(torch.load(pth))
        self.gt = torch.tensor(self.data.Qsym.bits,device = self.device,dtype=torch.float64)
    
    def hammingWeight(self, n):
      n = str(bin(n))
      one_count = 0
      for i in n:
         if i == "1":
            one_count+=1
      return one_count
     
    def Test(self):
        BER = []
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,-1*self.step):
            self.r = self.Generate_SNR(SNR,self.real_imag)
            
            loop   = tqdm(range(int(self.data.total*.5),self.data.total),desc="Progress")
            errors = 0
            frames = int(self.data.total*.5)
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i])
                Y  = torch.squeeze(self.gt[:,i])
                for j in range(0,self.data.sym_no):
                    rxbit = self.model(X[j])
                    rxbit = int(torch.argmax(rxbit))
                    txbit = int(Y[j])
                    errors+=self.hammingWeight(rxbit^txbit)
                    #if(rxbit^txbit != 0):
                    #    print(str(rxbit)+" -> "+str(txbit))         
                    
                if(i % 50 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
            
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[FOUR],get_time_string())
        vector_to_pandas("BER_{}.csv".format(formating),BER)
        
class TestNet_COMPLEX(NetLabs):
    def __init__(self,pth):
        loss_type = MSE
        super().__init__(loss_type,GOLDEN_BEST_SNR,GOLDEN_WORST_SNR,step=GOLDEN_STEP,real_imag = COMPLEX)
        self.model = self.Generate_Network_Model()
        self.model.load_state_dict(torch.load(pth))
        self.gt = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        
    def Test(self):
        BER = []
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,-1*self.step):
            self.r = self.Generate_SNR(SNR,self.real_imag)
            loop   = tqdm(range(int(self.data.total*.8),self.data.total),desc="Progress")
            errors = 0
            frames = self.data.total*.2
            for i in loop:
                X  = torch.squeeze(self.r[:,i])
                Y  = torch.squeeze(self.gt[:,i]).cpu().detach().numpy()
                pred   = self.model(X).cpu().detach().numpy()
                rxbits = self.data.Qsym.Demod(pred)
                txbits = self.data.Qsym.Demod(Y)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                #Status bar and monitor  
                if(i % 100 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
            
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
            
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[COMPLEX],get_time_string())
        vector_to_pandas("BER_{}.csv".format(formating),BER)