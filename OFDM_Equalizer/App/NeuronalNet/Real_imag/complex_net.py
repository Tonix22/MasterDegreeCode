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
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexConvTranspose2d,ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from torch.nn.functional import tanh ,hardtanh

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string

#Hyperparameters
BATCHSIZE  = 20
QAM        = 16
NUM_EPOCHS = 150
#
LAST_LIST   = 250
CONJ_ACTIVE = True
#If use x or x_mse
GROUND_TRUTH_SOFT = False # fts
NOISE = False
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 72 

#Optimizer
LEARNING_RATE = .001
EPSILON = .01

def complex_tanh(input):
    return torch.tanh(input.real).type(torch.complex64)+1j*torch.tanh(input.imag).type(torch.complex64)


class HardTahn_complex(nn.Module):

    def __init__(self):
        super(HardTahn_complex, self).__init__()
        self.hard_real = nn.Hardtanh()
        self.hard_imag = nn.Hardtanh()

    def forward(self, input):
        return   self.hard_real(input.real).type(torch.float64)+1j*self.hard_imag(input.imag).type(torch.float64)

class ComplexNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.complex_net = nn.Sequential(
            ComplexLinear(input_size, hidden_size),
            HardTahn_complex(),
            ComplexLinear(hidden_size, hidden_size*int(1.5)),
            HardTahn_complex(),
            ComplexLinear(hidden_size*int(1.5), hidden_size),
            HardTahn_complex(),
            ComplexLinear(hidden_size, input_size)
        ).double()
        #self.loss_f = nn.MSELoss()
        self.loss_f = self.Complex_MSE
        
    def Complex_MSE(self,output,target):
        return torch.sum(torch.abs((target-output)))
        
    def forward(self,x):        
        return self.complex_net(x)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.0007,eps=.007)
        
    def training_step(self, batch, batch_idx):
        #self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationAnglePhaseNet
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE)
        
        # ------------ Source Data ------------
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        src        = Y / src_abs_factor
        
        # ------------ Target Data ------------
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
        tgt            = x/tgt_abs_factor
        
        # ------------ Model Eval ------------
        output = self(src)
        
        #loss func
        loss  = self.loss_f(output,tgt)
        
        #self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        #self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationAnglePhaseNet
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE)
        
        # ------------ Source Data ------------
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        src        = Y / src_abs_factor
        
        # ------------ Target Data ------------
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
        tgt            = x/tgt_abs_factor
        
        # ------------ Model Eval ------------
        output = self(src)
        
        #loss func
        loss  = self.loss_f(output,tgt)
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 200):
            chann, x = batch
            chann    = chann.permute(0,3,1,2)
            Y        = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=True)
            
            
            # ------------ Source Data ------------
            #normalize factor, normalize by batch
            src_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
            src            = Y / src_abs_factor
            
            # ------------ Target Data ------------
            #Normalize target 
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            tgt            = x/tgt_abs_factor
            # ------------ Model Eval ------------
            #normalize factor, normalize by batch            
           
            out = self(src)
            # ------------ Predict Loss ------------
            # ------------ Concatenate ------------
            loss  = self.loss_f(out,tgt)
            # ------------ Denormalize ------------
            #denormalize angle          
            # ------------ Polar trans ------------
            #transform output to polar
            x_hat    = out
            x_t      = tgt
            
            x_hat = x_hat.cpu().to(torch.float32)
            x_t   = x_t.cpu().to(torch.float32)
            #torch polar polar(abs: Tensor, angle: Tensor)
           
            self.SNR_calc(x_hat,x_t,norm=True) 
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=True, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/Real_imag/lightning_logs/version_12/checkpoints/epoch=149-step=90000.ckpt')
    Cn = ComplexNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"ComplexNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    