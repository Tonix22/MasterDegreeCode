import torch
from FXPA import FXP_Linear,HardTanh,FXP_Sequential,fxpa_build
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
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string
from Scatter_plot_results import ComparePlot

#Hyperparameters
BATCHSIZE  = 100
QAM        = 16


class PolarNet(pl.LightningModule,Rx_loader):
    def __init__(self):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        checkpoint_file = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarNet/models/16QAM/version_80/checkpoints/epoch=2-step=900.ckpt',\
                            map_location=torch.device('cpu'))
        state_dict      = checkpoint_file['state_dict']
        # get an iterator for state_dict items
        iter_state_dict = iter(state_dict.items())
        layers= []
        # loop through the dictionary
        while True:
            try:
                # get two consecutive items
                w = next(iter_state_dict)
                b = next(iter_state_dict)
                linear = FXP_Linear(w[1],b[1])
                layers.append(linear)
                print(w[0], b[0])
            except StopIteration:
                # reached the end of the dictionary
                break
        
        self.mag_net   = [layers[0],HardTanh(),layers[1],layers[2],HardTanh(),layers[3]]
        self.mag_net   = FXP_Sequential(self.mag_net)
        self.angle_net = [layers[4],HardTanh(),layers[5],layers[6],HardTanh(),layers[7]]
        self.angle_net = FXP_Sequential(self.angle_net)
        
  
    def forward(self,abs,phase):
        mag = self.mag_net(abs)
        phase = self.angle_net(phase)
        return mag,phase
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters())
        
    def common_step(self,batch):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
            
        Y        = self.Get_Y(chann,x,conj=True,noise_activ=True)
        
        #Filter batches that are not outliers, borring batches
        valid_data, valid_indices   = self.filter_z_score(Y)
        if valid_data.numel() != 0:
            Y = valid_data
            x = x[valid_indices]
            
            if valid_data.dim() == 1:
                Y = torch.unsqueeze(Y, 0)
                x = torch.unsqueeze(x,0)
            
            # ------------ Source Data Preprocesing ------------
            #normalize factor, normalize by batch
            Y_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
            Y            = Y / Y_abs_factor
            Y_abs        = Y.abs()
            Y_ang        = (torch.angle(Y)) / (2 * np.pi)
        
            # ------------ Target Data Preprocesing------------
            #Normalize target
            
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            target         = x/tgt_abs_factor
            target_abs     = target.abs()
            target_ang     = target.angle()
            
            # model eval
            Y_abs = fxpa_build(Y_abs.numpy())
            Y_ang = fxpa_build(Y_ang.numpy())
            
            
            output_abs_float   = torch.zeros(Y.shape, dtype=torch.float32)
            output_angle_float = torch.zeros(Y.shape, dtype=torch.float32)
            
            for n in range(Y.shape[0]):
                output_abs,output_angle = self(Y_abs[n],Y_ang[n])
                output_angle            = output_angle*(2 * np.pi)
                
                for i in range(output_abs.shape[0]):
                    output_abs_float[n][i]   = output_abs[i]._N.astype(float)
                    output_angle_float[n][i] = output_angle[i]._N.astype(float)
            
            #Predict
            x_hat     = torch.polar(output_abs_float,output_angle_float)
            #ComparePlot(target,x_hat)
            self.SNR_calc(x_hat,target,norm=True) 
        
    
    def validation_epoch_end(self, outputs):
        # Concatenate the tensors in outputs
        losses = [x['val_loss'] for x in outputs]
        concatenated = torch.cat([l.view(-1) for l in losses])
        # Compute the average loss
        avg_loss = concatenated.mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        loss = 0
        if(batch_idx < 1):
            loss = self.common_step(batch)
        return loss 
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=True)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/MobileNet/lightning_logs/version_49/checkpoints/epoch=8-step=2700.ckpt')
    Cn = PolarNet()    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PolarNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    



    
"""
Net struct

nn.Linear(input_size, hidden_size,bias=True),
nn.Hardtanh(),
nn.Linear(hidden_size, hidden_size*int(2),bias=True),
nn.Linear(hidden_size*int(2), hidden_size,bias=True),
nn.Hardtanh(),
nn.Linear(hidden_size, input_size,bias=True)
"""
#Polar_net
