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
class NetLabs(object):
    def __init__(self, loss_type="MSE",best_snr = 60,worst_snr = 5):
        self.BEST_SNR  = best_snr
        self.WORST_SNR = worst_snr
        self.loss_type = loss_type
        #Data set read
        self.data   = RX()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Data numbers
        self.N = self.data.sym_no
        #Training data lenght is only 80%
        self.training_data = int(self.data.total*.8)
        
    def get_time_string(self):
        current_time = datetime.now()
        day  = current_time.day
        mon  = current_time.month
        year = current_time.year
        hr   = current_time.time().hour
        mn   = current_time.time().minute
        return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)
        
    def Generate_Network_Model(self):
        NN = None
        #MODEL COFING
        if(self.loss_type == "MSE"):
            NN  = LinearNet(input_size=self.N, hidden_size=2*self.N, num_classes=self.N)
            #self.model = Chann_EQ_Net(input_size=2*self.N, num_classes=2*self.N)
            
        if(self.loss_type == "Entropy"):
            NN  = LinearNet(input_size=2*self.N, hidden_size=4*self.N, num_classes=self.data.sym_no)
            
        NN = NN.to(self.device)
        
        self.optimizer = optim.Adam(NN.parameters(),lr=.001,eps=.001)
        
        if(self.loss_type == "MSE"):
            #criteria based on MSE
            self.criterion = nn.MSELoss()
        if(self.loss_type == "Entropy"):
            self.criterion = nn.CrossEntropyLoss()
        
        return NN        

    def Generate_SNR(self,SNR,real_imag):
        r = None
        self.data.AWGN(SNR)
        #right side of equalizer
        Entry = np.empty((self.data.sym_no,self.data.total,1),dtype=self.data.Qsym.r.dtype)
        for i in range(0,self.data.total):
            Y = self.data.Qsym.r[:,i]
            H = np.matrix(self.data.H[:,:,i])
            Entry[:,i]=H.H@Y
        
        r_real = torch.tensor(Entry.real,device  = torch.device('cuda'),dtype=torch.float64)
        r_imag = torch.tensor(Entry.imag,device  = torch.device('cuda'),dtype=torch.float64)
        if(real_imag == "real"):      
           r = r_real
        if(real_imag == "imag"):
            r = r_imag
        if(real_imag == "both"):
            r = torch.cat((r_real,r_imag),0)
            
        del Entry
        torch.cuda.empty_cache()
        
        return r
    

class TrainNet(NetLabs):
    def __init__(self,real_imag = "real",loss_type="MSE",best_snr = 60,worst_snr = 5):
        super().__init__(loss_type,best_snr,worst_snr)
        #NN model
        self.model = None 
        #Constant Paramters
        self.real_imag = real_imag
       
        #ground truth
        self.Get_ground_truth()
        
        #Set up NN
        self.model = self.Generate_Network_Model()
        
        
    def Get_ground_truth(self):
        if(self.loss_type == "MSE"):
            if(self.real_imag == "real"):
                self.gt = torch.tensor(self.data.Qsym.GroundTruth.real,device = self.device,dtype=torch.float64)
            if(self.real_imag == "imag"):
                self.gt = torch.tensor(self.data.Qsym.GroundTruth.imag,device = self.device,dtype=torch.float64)
            
        if(self.loss_type == "Entropy"):
            self.gt = torch.tensor(self.data.Qsym.bits,device = self.device,dtype=torch.float64)
    
        torch.cuda.empty_cache()
            
    def Train(self,epochs=3):
        df = pd.DataFrame()
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-5):
            for epochs_range in range(0,epochs):
                losses = []
                self.r = self.Generate_SNR(SNR,self.real_imag)
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
                    if(i % 1000 == 0):
                        loop.set_description(f"SNR [{SNR}] EPOCH[{epochs_range}]")
                        loop.set_postfix(loss=loss.cpu().detach().numpy())
                        #print(GPUtil.showUtilization())
                
                df["SNR{}".format(SNR)]= losses
                losses.clear()
            
        df.to_csv('reports/Train_Loss_SNR_{}_{}.csv'.format(self.real_imag,self.get_time_string()), header=True, index=False)
        self.SaveModel("PTH")
        
                    
    def SaveModel(self,format):
        if(format == "PTH"):
            torch.save(self.model.state_dict(),"models/OFDM_Equalizer{}.pth".format(self.get_time_string()))
        if(format == "ONNX"):
            torch.onnx.export(self.model,self.r[:,0].float(),"models/MSE_net.onnx", export_params=True,opset_version=10)
        
    
class TestNet(NetLabs):
    def __init__(self,pth_real,pth_imag):
        super().__init__()
        self.model_real = None
        self.model_imag = None
        self.model_real = self.Generate_Network_Model()
        self.model_imag = self.Generate_Network_Model()
        self.model_real.load_state_dict(torch.load(pth_real))
        self.model_imag.load_state_dict(torch.load(pth_imag))
        self.Get_ground_truth()
        
    def Get_ground_truth(self):
        self.gt_real = torch.tensor(self.data.Qsym.GroundTruth.real,device = self.device,dtype=torch.float64)
        self.gt_imag = torch.tensor(self.data.Qsym.GroundTruth.imag,device = self.device,dtype=torch.float64)
        self.gt      = torch.cat((self.gt_real,self.gt_imag),0)
        del self.gt_real
        del self.gt_imag
        torch.cuda.empty_cache()
        
    #************************
    #*******TESTING**********
    #************************
    def Test(self):
        df = pd.DataFrame()
        BER    = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-1):
            losses = []
            self.r = self.Generate_SNR(SNR,"both")
            loop   = tqdm(range(self.training_data,self.data.total),desc="Progress")
            errors = 0
            
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                
                pred_real = self.model_real(X[0:self.data.sym_no].float())
                pred_imag = self.model_imag(X[self.data.sym_no:].float())
                
                loss_real = self.criterion(pred_real,Y[0:self.data.sym_no].float())
                loss_imag = self.criterion(pred_imag,Y[self.data.sym_no:].float())
                
                loss = (loss_real + loss_imag)/2

                losses.append(loss.cpu().detach().numpy())
                
                #BER
                if(self.loss_type == "MSE"):
                    #demodulate prediction data
                    real   = pred_real.cpu().detach().numpy()
                    imag   = pred_imag.cpu().detach().numpy()
                    res    = real + 1j*imag
                    rxbits = self.data.Qsym.Demod(res)

                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*self.data.total))
                    print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*self.data.total))
        
        
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