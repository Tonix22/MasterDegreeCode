from cmath import pi
from NN_lab import NetLabs
import torch
import numpy as np
from Constants import *
import pandas as pd
from  tqdm import tqdm
import os 
import sys

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
report_path = main_path+"reports"
model_path  = main_path+"models"

sys.path.insert(0, main_path+"tools")

from utils import get_date_string


class TrainNet(NetLabs):
    def __init__(self,real_imag = REAL,loss_type=MSE,toggle = False,best_snr = 60,worst_snr = 5,step=-1):
        super().__init__(loss_type,best_snr,worst_snr,toggle=toggle,step=step,real_imag = real_imag)
        print("device = {}".format(self.device))
        #NN model
        self.model = None 
       
        #ground truth
        if(self.loss_type == CROSSENTROPY):
            self.gt = self.Get_ground_truth(self.data.Qsym.bits)
        elif(self.loss_type == MSE_COMPLETE):
            self.gt = torch.tensor(self.data.Qsym.GroundTruth).to(self.device)
        elif(self.loss_type == BCE):
            self.gt = torch.tensor(self.data.Qsym.bits,device = self.device,dtype=torch.float64)
        else:
            self.gt = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        #Set up NN
        self.model = self.Generate_Network_Model()
       
    def LMSE_Ground_Truth(self,i,SNR):
        Y = self.data.Qsym.r[:,i]
        H = np.matrix(self.data.H[:,:,i])
        LMMSE = np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))@H.H@Y
        if(self.real_imag == COMPLEX):
            return torch.from_numpy(LMMSE).to(self.device)
        if(self.real_imag == REAL):
            return torch.from_numpy(LMMSE.real).to(self.device)
        if(self.real_imag == IMAG):
            return torch.from_numpy(LMMSE.imag).to(self.device)
        if(self.real_imag == ANGLE):
            return torch.from_numpy(np.angle(LMMSE)/pi).to(self.device)
        if(self.real_imag == ABS):
            return torch.from_numpy(np.abs(LMMSE)).to(self.device)
        if(self.real_imag == INV):
            channels = np.concatenate((LMMSE.real,LMMSE.imag),axis=0)
            channels = np.reshape(channels,(2,self.data.sym_no))
            return torch.from_numpy(channels).to(self.device)
    
    def TrainBCE(self,epochs=3):
        df   = pd.DataFrame()
        pred = None
        toogle_iter = 1
        size = 1<<self.data.bitsframe
        symbol = torch.zeros(size,dtype=torch.float64).to(self.device)
        if(self.toggle == True):
            toogle_iter = 2
        for it in range(0,toogle_iter):# 2 if toggle required
            for SNR in range(self.BEST_SNR,self.WORST_SNR-1,self.step):
                for epochs_range in range(0,epochs):
                    losses = []
                    self.r = self.Generate_SNR(SNR,self.real_imag)
                    loop  = tqdm(range(0,3000),desc="Progress")
                    for i in loop:
                        X  = torch.squeeze(self.r[:,i])  # input TODO
                        Y  = torch.squeeze(self.gt[:,i])
                        avg_loss = 0
                        for j in range(0,self.data.sym_no):
                            symbol[int(Y[j])]=1
                            pred = self.model(X[j])
                            loss = self.criterion(pred,symbol)
                            avg_loss+=loss
                            #Clear gradient
                            self.optimizer.zero_grad()
                            # Backpropagation
                            loss.backward()
                            # Update Gradient
                            self.optimizer.step()
                            symbol[int(Y[j])]=0
                        
                        avg_loss = avg_loss/self.data.sym_no
                        #Record the Average loss
                        losses.append(avg_loss.cpu().detach().numpy())   
                        #Status bar and monitor    
                        if(i % 50 == 0):
                            loop.set_description(f"SNR [{SNR}] EPOCH[{epochs_range}] [{real_imag_str[self.real_imag]}]]")
                            loop.set_postfix(loss=avg_loss.cpu().detach().numpy())
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[self.real_imag],get_date_string())
        df.to_csv('{}/Train_{}.csv'.format(report_path,formating), header=True, index=False)
        
        self.SaveModel("PTH",formating)

                                         
    def TrainMSE(self,epochs=3):
        df   = pd.DataFrame()
        pred = None
        toogle_iter = 1
        if(self.toggle == True):
            toogle_iter = 2
        
        for it in range(0,toogle_iter):# 2 if toggle required
            for SNR in range(self.BEST_SNR,self.WORST_SNR-1,self.step):
                for epochs_range in range(0,epochs):
                    losses = []
                    self.r = self.Generate_SNR(SNR,self.real_imag)
                    #loop is the progress bar
                    loop  = tqdm(range(0,self.training_data),desc="Progress")
                    for i in loop:
                        X  = torch.squeeze(self.r[:,i],1)  # input
                        Y  = None
                        if(self.toggle == True):
                            if(epochs_range%2 == 0):
                                Y  = torch.squeeze(self.gt[:,i],1)
                            else:
                                Y  = torch.squeeze(self.LMSE_Ground_Truth(i,SNR),1)
                        else:    
                            #Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                            Y   = torch.squeeze(self.LMSE_Ground_Truth(i,SNR),1) 
                            
                        if(self.loss_type == MSE_COMPLETE):
                            real = X[0:self.data.sym_no]
                            imag = X[self.data.sym_no:]
                            pred = self.model(real,imag)
                            Y    = torch.squeeze(self.gt[:,i],1)
                            loss = self.criterion(pred,Y)
                            
                        else:
                            pred = self.model(X)
                            loss = self.criterion(pred,Y)
                            
                            
                        #Record the Average loss
                        losses.append(loss.cpu().detach().numpy())
                        #Clear gradient
                        self.optimizer.zero_grad()
                        # Backpropagation
                        loss.backward()
                        # Update Gradient
                        self.optimizer.step()
                        
                        #Status bar and monitor    
                        if(i % 1000 == 0):
                            loop.set_description(f"SNR [{SNR}] EPOCH[{epochs_range}] [{real_imag_str[self.real_imag]}]]")
                            loop.set_postfix(loss=loss.cpu().detach().numpy())
                            #print(GPUtil.showUtilization())
                    
                    df["SNR{}".format(SNR)]= losses
                    losses.clear()
            
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[self.real_imag],get_date_string())
        df.to_csv('{}/Train_{}.csv'.format(report_path,formating), header=True, index=False)
        
        self.SaveModel("PTH",formating)
        
                    
    def SaveModel(self,format,formating):
        if(format == "PTH"):
            torch.save(self.model.state_dict(),"{}/OFDM_Eq_{}.pth".format(model_path,formating))
        if(format == "ONNX"):
            torch.onnx.export(self.model,self.r[:,0].float(),"{}/OFDM_Eq_{}.onnx".format(model_path,formating), export_params=True,opset_version=10)
    
    def TraiINV(self,epochs=3):
        df   = pd.DataFrame()
        pred = None
        toogle_iter = 1
        if(self.toggle == True):
            toogle_iter = 2
            
        torch.set_default_dtype(torch.float64)
        for it in range(0,toogle_iter):# 2 if toggle required
            for SNR in range(self.BEST_SNR,self.WORST_SNR,self.step):
                for epochs_range in range(0,epochs):
                    losses = []                
                    #loop is the progress bar
                    loop  = tqdm(range(0,self.training_data),desc="Progress")
                    for i in loop:
                        #GT  = torch.squeeze(self.LMSE_Ground_Truth(i,SNR),1)
                        GT   = torch.squeeze(self.gt[:,:,i])
                        
                        #input parameters
                        self.data.AWGN(SNR)
                        Y   = self.data.Qsym.r[:,i]
                        H   = np.matrix(self.data.H[:,:,i])
                        P   = ((H.H@H+np.eye(self.data.sym_no)*(10**(-SNR/10))))
                        #separate H in 2 channels
                        P_chann = np.concatenate((P.real,P.imag),axis=0)
                        P_chann = P_chann.A1
                        P_chann = np.reshape(P_chann,(2,self.data.sym_no,self.data.sym_no))
                        #separate H.H@Y
                        RightSide = H.H@Y
                        RightSide = np.concatenate((RightSide.real,RightSide.imag),axis=0)
                        RightSide = RightSide.A1
                        RightSide = np.reshape(RightSide,(2,self.data.sym_no))
                        
                        #Convert to torch CUDA vector
                        P_chann   = torch.from_numpy(P_chann).to(self.device)
                        RightSide = torch.from_numpy(RightSide).to(self.device)
        
                        inverse = self.model(P_chann)
                        #(a+bi)(c+di)
                        #(ac-bd)+i(ad+bc)
                        a = inverse[0]
                        b = inverse[1]
                        c = RightSide[0]#real
                        d = RightSide[1]#imag
                        
                        X_hat = torch.zeros(2,self.data.sym_no).to(self.device)
                        X_hat[0]   = a@c-b@d
                        X_hat[1]   = a@d+b@c
                        
                        loss = self.criterion(X_hat,GT)
                             
                        #Record the Average loss
                        losses.append(loss.cpu().detach().numpy())
                        #Clear gradient
                        self.optimizer.zero_grad()
                        # Backpropagation
                        loss.backward()
                        # Update Gradient
                        self.optimizer.step()
                        
                        #Status bar and monitor    
                        if(i % 50 == 0):
                            loop.set_description(f"SNR [{SNR}] EPOCH[{epochs_range}] [{real_imag_str[self.real_imag]}]]")
                            loop.set_postfix(loss=loss.cpu().detach().numpy())
                            #print(GPUtil.showUtilization())
                    
                    df["SNR{}".format(SNR)]= losses
                    losses.clear()
            
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[self.real_imag],get_date_string())
        df.to_csv('{}/Train_{}.csv'.format(model_path,formating), header=True, index=False)
        
        self.SaveModel("PTH",formating)