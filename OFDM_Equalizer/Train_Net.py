from cmath import pi
from NN_lab import NetLabs
import torch
import torch.nn as nn
import numpy as np
from Constants import *
import pandas as pd
from  tqdm import tqdm

class TrainNet(NetLabs):
    def __init__(self,real_imag = REAL,loss_type=MSE,toggle = False,best_snr = 60,worst_snr = 5,step=-1):
        super().__init__(loss_type,best_snr,worst_snr,toggle=toggle,step=step)
        #NN model
        self.model = None 
        #Constant Paramters
        self.real_imag = real_imag
       
        #ground truth
        if(self.loss_type == CROSSENTROPY or self.loss_type == MSE_COMPLETE):
            self.gt = self.Get_ground_truth(self.data.Qsym.bits)
        else:
            self.gt = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        #Set up NN
        self.model = self.Generate_Network_Model()
       
    def LMSE_Ground_Truth(self,i,SNR):
        Y = self.data.Qsym.r[:,i]
        H = np.matrix(self.data.H[:,:,i])
        LMMSE = np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))@H.H@Y
        if(self.real_imag == REAL):
            return torch.from_numpy(LMMSE.real).to(self.device)
        if(self.real_imag == IMAG):
            return torch.from_numpy(LMMSE.imag).to(self.device)
        if(self.real_imag == ANGLE):
            return torch.from_numpy(np.angle(LMMSE)/pi).to(self.device)
        if(self.real_imag == ABS):
            return torch.from_numpy(np.abs(LMMSE)).to(self.device)
            
                                            
    def TrainMSE(self,epochs=3):
        df   = pd.DataFrame()
        pred = None
        toogle_iter = 1
        if(self.toggle == True):
            toogle_iter = 2
        
        for it in range(0,toogle_iter):# 2 if toggle required
            for SNR in range(self.BEST_SNR,self.WORST_SNR,self.step):
                for epochs_range in range(0,epochs):
                    losses = []
                    self.r = self.Generate_SNR(SNR,self.real_imag)
                    #loop is the progress bar
                    loop  = tqdm(range(0,self.training_data),desc="Progress")
                    for i in loop:
                        X  = torch.squeeze(self.r[:,i],1)  # input
                        Y  = None
                        if(self.toggle == True):
                            if(it %2 == 1):
                                Y  = torch.squeeze(self.gt[:,i],1)
                            else:
                                Y  = torch.squeeze(self.LMSE_Ground_Truth(i,SNR),1)
                        else:    
                            Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                            
                        if(self.loss_type == MSE_COMPLETE):
                            a = X[0:self.data.sym_no].float()
                            b = X[self.data.sym_no:].float()
                            pred = self.model(a,b)
                            Y    = Y/3 #normalize constelation points
                            loss = self.criterion(pred,Y.float())
                        else:
                            pred = self.model(X.float())
                            loss = self.criterion(pred,Y.float())
                            
                            
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
            
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[self.real_imag],self.get_time_string())
        df.to_csv('reports/Train_{}.csv'.format(formating), header=True, index=False)
        
        self.SaveModel("PTH",formating)
        
                    
    def SaveModel(self,format,formating):
        if(format == "PTH"):
            torch.save(self.model.state_dict(),"models/OFDM_Eq_{}.pth".format(formating))
        if(format == "ONNX"):
            torch.onnx.export(self.model,self.r[:,0].float(),"models/MSE_net.onnx", export_params=True,opset_version=10)
        