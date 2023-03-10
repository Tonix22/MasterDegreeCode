import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string
from Scatter_plot_results import ComparePlot

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"MagNet")
sys.path.insert(0, main_path+"PhaseNet")

from MagNet import MagEqualizer
from PhaseNet import PhaseEqualizer

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
INPUT_SIZE = 48
MAG_PATH   = '/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/MagNet/lightning_logs/version_30/checkpoints/epoch=9-step=12000.ckpt'
ANGLE_PATH = '/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/models/16QAM/version_138/checkpoints/epoch=1-step=2400.ckpt'

class PolarMixed(pl.LightningModule,Rx_loader):
    def __init__(self):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        self.output_size = 48
        self.mag_net   = MagEqualizer(INPUT_SIZE,120)
        self.angle_net = PhaseEqualizer(INPUT_SIZE,240)
        #load 
        state = torch.load(MAG_PATH,map_location=torch.device('cpu'))['state_dict']
        new_state_dict = {}
        for key, value in state.items():
            new_key = key.replace('mag_net.mag_net.', 'mag_net.')
            new_state_dict[new_key] = value
            
        self.mag_net.load_state_dict(new_state_dict)
        
        state = torch.load(ANGLE_PATH,map_location=torch.device('cpu'))['state_dict']
        new_state_dict = {}
        for key, value in state.items():
            new_key = key.replace('angle_net.angle_net.', 'angle_net.')
            new_state_dict[new_key] = value
        
        self.angle_net.load_state_dict(new_state_dict)
        
      
    def forward(self,abs,phase):        
        return self.mag_net(abs),self.angle_net(phase)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters())
    
    def common_step(self,batch):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        
        Y = self.Get_Y(chann,x,conj=True,noise_activ=True)
        valid_data, valid_indices   = self.filter_z_score(Y)
        
        if valid_data.numel() != 0:
            Y     = valid_data
            x     = x[valid_indices]
            chann = chann[valid_indices]
            
            if valid_data.dim() == 1:
                Y = torch.unsqueeze(Y, 0)
                x = torch.unsqueeze(x,0)
                chann = torch.unsqueeze(chann,0)
            
            self.start_clock() #start time eval ***************
            
            # ------------ Source Data Preprocesing ------------
            # ANGLE
            Y_ang        = (torch.angle(Y)) / (2 * torch.pi)
            # Maginutd recieved zero forcing
            Y            = self.ZERO_X(chann,Y)
            Y_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
            Y            = Y / Y_abs_factor
            Y_abs        = Y.abs()
            
            # model eval
            output_abs,output_angle = self(Y_abs,Y_ang)
            output_angle            = output_angle*(2 * torch.pi)
            
            x_hat     = torch.polar(output_abs,output_angle)
            
            self.stop_clock(int(Y.shape[0])) #Stop time eval ***************
            
            # ------------ Target Data Preprocesing------------
            #Normalize target
            
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            target         = x/tgt_abs_factor
                
            self.BER_cal(x_hat,target,norm=True) 
            
        return 0
    
    def predict_step(self, batch, batch_idx):
        loss = 0
        if(batch_idx < 400):
            loss = self.common_step(batch)
        return loss 
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=True)
    Cn = PolarMixed()
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PolarMixed",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    