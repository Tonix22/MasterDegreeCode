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

#Hyperparameters
BATCHSIZE  = 50
QAM        = 16
NUM_EPOCHS = 50
LEARNING_RATE = .0001
#
LAST_LIST   = 250
CONJ_ACTIVE = True
#If use x or x_mse
GROUND_TRUTH_SOFT = False # fts
NOISE = False
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 120



def complex_tanh(input):
    return torch.tanh(input.real).type(torch.complex64)+1j*torch.tanh(input.imag).type(torch.complex64)

def apply_complex(fr, fi, input, dtype = torch.complex128):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

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
            ComplexLinear(hidden_size, hidden_size*int(2)),
            ComplexLinear(hidden_size*int(2), hidden_size),
            HardTahn_complex(),
            ComplexLinear(hidden_size, input_size)
        ).double()
        self.loss_f = self.Complex_MSE
        #self.loss_f = self.Polar_MSE
        
    def Complex_MSE(self,output,target):
        return torch.mean(torch.abs((target-output)))

    def Polar_MSE(self,output,target):
        return .5*torch.mean((torch.square(torch.log(torch.abs(target)/torch.abs(output)))\
            +torch.square(torch.angle(target)-torch.angle(output))))
        
    def forward(self,x):        
        return self.complex_net(x)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
    
    def common_step(self,batch,predict=False):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        
        if(predict == False):
            self.SNR_db = 40
            
        Y  = self.Get_Y(chann,x,conj=True,noise_activ=True)
                
        #Filter batches that are not outliers, borring batches
        valid_data, valid_indices   = self.filter_z_score(Y)
        if valid_data.numel() != 0:
            Y = valid_data
            x = x[valid_indices]
            
            if valid_data.dim() == 1:
                Y = torch.unsqueeze(Y, 0)
                x = torch.unsqueeze(x,0)
            
            self.start_clock() #start time eval ***************
            
            # ------------ Source Data Preprocesing ------------
            #normalize factor, normalize by batch
            Y_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
            Y            = Y / Y_abs_factor
        
            # ------------ Target Data Preprocesing------------
            #Normalize target
            
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            target         = x/tgt_abs_factor
            
            # model eval
            x_hat = self(Y)
            
            self.stop_clock(int(Y.shape[0])) #Stop time eval ***************
            
            # Complex build
            loss  = self.loss_f(x_hat,target)
            
            if(predict == True):
                self.BER_cal(x_hat,target,norm=True) 
            
        else: # block back propagation
            loss = torch.tensor([0.0],requires_grad=True).to(torch.float64).to(self.device)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        
        #self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 200):
            self.common_step(batch,predict=True)
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/Real_imag/lightning_logs/version_41/checkpoints/epoch=199-step=48000.ckpt')
    Cn = ComplexNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"ComplexNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    