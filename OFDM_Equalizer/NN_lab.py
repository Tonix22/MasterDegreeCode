from operator import index
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from   Recieved import RX
import GPUtil
from   tqdm import tqdm
from   Networks import LinearNet,Chann_EQ_Net
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plot
#from torch.utils.tensorboard import SummaryWriter

class NetLabs():
    def __init__(self,loss_type="MSE",best_snr = 60,worst_snr = 5):
        #Constant Paramters
        self.BEST_SNR  = best_snr
        self.WORST_SNR = worst_snr
        self.loss_type = loss_type
        #Data set read
        self.data   = RX()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #Data numbers
        self.N = self.data.sym_no
        
        #ground truth
        self.Get_ground_truth()
        
        #Set up NN
        self.Generate_Network_Model()
        
        #Training data lenght is only 80%
        self.training_data = int(self.data.total*.8)
        
    def Get_ground_truth(self):
        if(self.loss_type == "MSE"):
            self.gt_real = torch.tensor(self.data.Qsym.GroundTruth.real,device = self.device,dtype=torch.float64)
            self.gt_imag = torch.tensor(self.data.Qsym.GroundTruth.imag,device = self.device,dtype=torch.float64)
            self.gt      = torch.cat((self.gt_real,self.gt_imag),0)
            del self.gt_real
            del self.gt_imag
            torch.cuda.empty_cache()
            
        if(self.loss_type == "Entropy"):
            self.gt = torch.tensor(self.data.Qsym.bits,device = self.device,dtype=torch.float64)
    

    def Generate_Network_Model(self):
        #MODEL COFING
        if(self.loss_type == "MSE"):
            self.model  = LinearNet(input_size=2*self.N, hidden_size=2*self.N, num_classes=2*self.N)
            #self.model = Chann_EQ_Net(input_size=2*self.N, num_classes=2*self.N)
            
        if(self.loss_type == "Entropy"):
            self.model  = LinearNet(input_size=2*self.N, hidden_size=4*self.N, num_classes=self.data.sym_no)
            
        self.model  = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=.005,eps=.005)
        
        if(self.loss_type == "MSE"):
            #criteria based on MSE
            self.criterion = nn.MSELoss()
        if(self.loss_type == "Entropy"):
            self.criterion = nn.CrossEntropyLoss()
            
    #************************
    #*******TRAINNING********
    #************************
    
    def Trainning(self):
        df = pd.DataFrame()
        for epochs in range(0,2):
            for SNR in range(self.BEST_SNR,self.WORST_SNR,-1):
                losses = []
                self.Generate_SNR(self.data,SNR)
                #loop is the progress bar
                loop  = tqdm(range(0,self.training_data),desc="Progress")
                for i in loop:     
                    X  = torch.squeeze(self.r[:,i],1)  # input
                    Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                    # Compute prediction and loss
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
                    if(i % 500 == 0):
                        loop.set_description(f"SNR [{SNR}] EPOCH[{epochs}]")
                        loop.set_postfix(loss=loss.cpu().detach().numpy())
                        print(GPUtil.showUtilization())
                
                df["SNR{}".format(SNR)]= losses
                losses.clear()
            
        df.to_csv('reports/Train_Loss_SNR_{}.csv'.format(self.get_time_string()), header=True, index=False)
        self.SaveModel("PTH")
        
                    
    #************************
    #*******TESTING**********
    #************************
    def Testing(self,pth):
        df = pd.DataFrame()
        
        self.model.load_state_dict(torch.load(pth))
        BER    = []
        frames = self.data.total-self.training_data
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-1):
            losses = []
            
            self.Generate_SNR(self.data,SNR)
            loop   = tqdm(range(self.training_data,self.data.total),desc="Progress")
            errors = 0
            
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                
                pred = self.model(X.float())
                loss = self.criterion(pred,Y.float())
                losses.append(loss.cpu().detach().numpy())
                
                #BER
                if(self.loss_type == "MSE"):
                    #demodulate prediction data
                    split  = torch.tensor_split(pred.cpu().detach(),2)
                    real   = split[0].numpy()
                    imag   = split[1].numpy()
                    res    = real + 1j*imag
                    rxbits = self.data.Qsym.Demod(res)
                else:
                    rxbits = pred.cpu().detach()

                
                errors+=np.unpackbits((self.data.Qsym.bits[:,i]^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        
        str_time = self.get_time_string()
        df.to_csv('reports/Testing_Loss_SNR{}.csv'.format(str_time), header=True, index=False)
        
        indexValues = np.arange(self.WORST_SNR,self.BEST_SNR,)
        BER = np.asarray(BER)
        plot.grid(True, which ="both")
        plot.semilogy(indexValues,BER)
        plot.title('SNR and BER')
        # Give x axis label for the semilogy plot
        plot.xlabel('SNR')
        # Give y axis label for the semilogy plot
        plot.ylabel('BER')
        plot.savefig('plots/Test_BER_SNR{}.png'.format(str_time))
           
           
        
    def SaveModel(self,format):
        if(format == "PTH"):
            torch.save(self.model.state_dict(),"models/OFDM_Equalizer{}.pth".format(self.get_time_string()))
        if(format == "ONNX"):
            torch.onnx.export(self.model,self.r[:,0].float(),"models/MSE_net.onnx", export_params=True,opset_version=10)

    def Generate_SNR(self,data,SNR):
        data.AWGN(SNR)
        #right side of equalizer
        Entry = np.empty((self.data.sym_no,self.data.total,1),dtype=self.data.Qsym.r.dtype)
        for i in range(0,self.data.total):
            Y = data.Qsym.r[:,i]
            H = np.matrix(data.H[:,:,i])
            Entry[:,i]=H.H@Y
                
        r_real = torch.tensor(Entry.real,device  = torch.device('cuda'),dtype=torch.float64)
        r_imag = torch.tensor(Entry.imag,device  = torch.device('cuda'),dtype=torch.float64)
        self.r = torch.cat((r_real,r_imag),0)
        del r_real
        del r_imag
        del Entry
        torch.cuda.empty_cache()
        
    def get_time_string(self):
        current_time = datetime.now()
        day  = current_time.day
        mon  = current_time.month
        year = current_time.year
        hr   = current_time.time().hour
        mn   = current_time.time().minute
        return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)
    
    

