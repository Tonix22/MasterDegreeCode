import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from   Recieved import RX
import GPUtil
from   tqdm import tqdm
from   Networks import LinearNet
import pandas as pd
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter

class NetLabs():
    def __init__(self,loss_type="MSE",worst_snr = 5):
        #Constant Paramters
        self.BEST_SNR  = 60
        self.WORST_SNR = worst_snr
        #Data set read
        self.data = RX()
        
        #Data numbers
        self.N = self.data.sym_no
        
        #ground truth
        self.Get_ground_truth(loss_type)
        
        #Set up NN
        self.Generate_Network_Model(loss_type)
        
        #Training data lenght is only 80%
        self.training_data = int(self.data.total*.8)
        
    def Get_ground_truth(self,criteria):
        if(criteria == "MSE"):
            self.gt_real = torch.tensor(self.data.Qsym.GroundTruth.real,device  = torch.device('cuda'),dtype=torch.float64)
            self.gt_imag = torch.tensor(self.data.Qsym.GroundTruth.imag,device  = torch.device('cuda'),dtype=torch.float64)
            self.gt = torch.cat((self.gt_real,self.gt_imag),0)
            
        if(criteria == "Entropy"):
            pass
            #self.gt_real
            #self.gt_imag
            #self.gt
        
        del self.gt_real
        del self.gt_imag
        torch.cuda.empty_cache()


    def Generate_Network_Model(self,criteria):
        #MODEL COFING
        if(criteria == "MSE"):
            self.model  = LinearNet(input_size=2*self.N, hidden_size=4*self.N, num_classes=2*self.N)
        
        if(criteria == "Entropy"):
            self.model  = LinearNet(input_size=2*self.N, hidden_size=4*self.N, num_classes=self.data.sym_no)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=.001,eps=.001)
        
        if(criteria == "MSE"):
            #criteria based on MSE
            self.criterion = nn.MSELoss()
        if(criteria == "Entropy"):
            self.criterion = nn.CrossEntropyLoss()
            
    
    def Trainning(self):
        df = pd.DataFrame()
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-1):
            losses = []
            self.Generate_SNR(self.data,SNR)
            #loop is the progress bar
            loop  = tqdm(range(0,self.training_data),desc="Progress")
            for i in loop:     
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # groung thruth
                        
                # Compute prediction and loss
                pred = self.model(X.float())
                loss = self.criterion(pred,Y.float())
                
                #Record the Average loss
                losses.append(torch.mean(loss).cpu().detach().numpy())
                #Clear gradient
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update Gradient
                self.optimizer.step()
                
                #Status bar and monitor    
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    print(GPUtil.showUtilization())
            
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
        df.to_csv('reports/Train_Loss_SNR_{}.csv'.format(self.get_time_string()), header=True, index=False)
        self.SaveModel("PTH")
        
                    
    #TODO BER/SNR PLOT
    def Testing(self,pth):
        df = pd.DataFrame()
        
        self.model.load_state_dict(torch.load(pth))
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-1):
            losses = []
            
            self.Generate_SNR(self.data,SNR)
            loop  = tqdm(range(self.training_data,self.data.total),desc="Progress")
            
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                
                pred = self.model(X.float())
                loss = self.criterion(pred,Y.float())
                
                losses.append(torch.mean(loss).cpu().detach().numpy())
                
                #Status bar and monitor  
                if(i % 100 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    print(GPUtil.showUtilization())
            
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
        df.to_csv('reports/Train_Loss_SNR_{}.csv'.format(self.get_time_string()), header=True, index=False)
                    
        
    def SaveModel(self,format):
        if(format == "PTH"):
            torch.save(self.model.state_dict(),"models/OFDM_Equalizer{}.pth".format(self.get_time_string()))
        if(format == "ONNX"):
            torch.onnx.export(self.model,self.r[:,0].float(),"models/MSE_net.onnx", export_params=True,opset_version=10)

    def Generate_SNR(self,data,SNR):
        data.AWGN(SNR)
        r_real = torch.tensor(data.Qsym.r.real,device  = torch.device('cuda'),dtype=torch.float64)
        r_imag = torch.tensor(data.Qsym.r.imag,device  = torch.device('cuda'),dtype=torch.float64)
        self.r = torch.cat((r_real,r_imag),0)
        del r_real
        del r_imag
        torch.cuda.empty_cache()
        
    def get_time_string(self):
        current_time = datetime.now()
        day  = current_time.day
        mon  = current_time.month
        year = current_time.year
        hr   = current_time.time().hour
        mn   = current_time.time().minute
        return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)
    
    

