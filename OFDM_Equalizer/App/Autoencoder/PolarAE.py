import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from Common.Conv_real import Encoder
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import get_time_string


#Hyperparameters
BATCHSIZE  = 10
QAM        = 4

class PolarAE(pl.LightningModule,Rx_loader):
    def __init__(self,encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")        
        self.Encoder_angle = Encoder(encoded_space_dim)
        self.Encoder_abs   = Encoder(encoded_space_dim)
        checkpoint_angle   = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/Angle_AE/lightning_logs/version_15/checkpoints/epoch=420-step=505200.ckpt')
        checkpoint_abs     = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/Mag_AE/lightning_logs/version_3/checkpoints/epoch=74-step=90000.ckpt')
        #state dict
        self.load_encoder(self.Encoder_angle,checkpoint_angle)
        self.load_encoder(self.Encoder_abs,checkpoint_abs)
        print("h")

        
    def load_encoder(self,model,checkpoint):
        state_dict_angle   = checkpoint['state_dict']
        # Create a new state dictionary that only includes the keys for the encoder
        encoder_state_dict = {}
        for key, value in state_dict_angle.items():
            if key.startswith('encoder.encode'):
                encoder_state_dict[key.replace('encoder.', '')] = value

        # Load the encoder state dictionary into the new model
        model.load_state_dict(encoder_state_dict)
        
    def forward(self, x):        
        return x
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters())    
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 100):
            # training_step defines the train loop. It is independent of forward
            chann, x = batch
            #chann preparation
            chann = chann.permute(0,3,1,2)
            # ------------ Generate Y ------------
            Y        = self.Get_Y(chann,x,conj=False,noise_activ=True)
            # ------------ Source Data ------------
            # Normalize the first channel by dividing by max_abs
            chann         = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
            max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
            chann         = chann / max_magnitude
            chann         = chann.unsqueeze(dim=1)
            
            # ------------ Target Data ANGLE------------
            chann_ang = (torch.angle(chann)+torch.pi)/(2*torch.pi) #input
            
            # encoder
            angle = self.Encoder_angle(chann_ang)
            mag   = self.Encoder_abs(torch.abs(chann))
            # denormalize
            angle = (angle*(2*torch.pi))-torch.pi
            mag   = mag*torch.squeeze(max_magnitude)
            #build complex channel
            z = torch.polar(mag,angle)
            
            x_hat = Y/z #ZERO FORCING equalizer
        
            self.SNR_calc(x_hat,x,norm=False) 
            
        return 0 
            
    
    def predict_dataloader(self):
        return self.test_loader
        
        
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=2)],enable_checkpointing=False)
    PolAE   = PolarAE(48)
    #name of output log file 
    formating = "Test_(PolarAE_QAM_{})_{}".format(QAM,get_time_string())
    PolAE.SNR_BER_TEST(trainer,formating)