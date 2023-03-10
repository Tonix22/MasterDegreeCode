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

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
NUM_EPOCHS = 10
#
CONJ = 1
ZERO = 2
DENOISE = 3 
MODE = CONJ


#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 120

#Optimizer
LEARNING_RATE = 5e-5

class MagEqualizer(nn.Module):
    
    def __init__(self,input_size, hidden_size):
        super(MagEqualizer,self).__init__()
        self.mag_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(2),bias=True),
            nn.Linear(hidden_size*int(2), hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, input_size,bias=True)
        ).double()

    def forward(self,abs): 
        return self.mag_net(abs)

class MagNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        self.output_size = 48
        self.mag_net = MagEqualizer(input_size,hidden_size)
        self.loss_f = nn.MSELoss()
        self.mode   = MODE
      
    def forward(self,abs):        
        return self.mag_net(abs)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
    
    def common_step(self,batch,predict=False):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        
        if(predict == False):
            self.SNR_db = 25
            
        Y        = self.Get_Y(chann,x,conj=False,noise_activ=True)
        Y        = self.ZERO_X(chann,Y)
            
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
        
            # ------------ Target Data Preprocesing------------
            #Normalize target
            
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            target         = x/tgt_abs_factor
            target_abs     = target.abs()
            
            # model eval
            output_abs = self(Y_abs)
            
            #output = torch.polar(output_abs,target.angle())
            # Complex build
            loss = self.loss_f(target_abs,output_abs)
            #loss = .5*self.loss_f(target.real,output.real)+.5*self.loss_f(target.imag,output.imag)
            
            if(predict == True):
                x_hat     = torch.polar(output_abs,target.angle())
                #ComparePlot(target,x_hat)
                
                self.BER_cal(x_hat,target,norm=True) 
            
        else: # block back propagation
            loss = torch.tensor([0.0],requires_grad=True).to(torch.float64).to(self.device)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        
        self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
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
        if(batch_idx < 400):
            loss = self.common_step(batch,predict=True)
        return loss 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=False, max_epochs=NUM_EPOCHS,
                resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/MagNet/lightning_logs/version_30/checkpoints/epoch=9-step=12000.ckpt')
    Cn = MagNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"MagNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    