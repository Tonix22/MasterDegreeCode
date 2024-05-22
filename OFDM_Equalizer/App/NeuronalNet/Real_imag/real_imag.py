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
from utils import vector_to_pandas, get_date_string

#Hyperparameters
BATCHSIZE  = 20
QAM        = 4
NUM_EPOCHS = 50
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

class RealImag(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.real_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            nn.Hardtanh()
        ).double()
        self.imag_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            nn.Hardtanh()
        ).double()
        
        self.loss_f = nn.MSELoss()
        
    def distance_loss(self,x_out,y_out,x_tgt,y_tgt):
        return torch.mean(torch.sqrt((x_out-x_tgt)**2+(y_out-y_tgt)**2))
        
        
    def forward(self,abs,phase):        
        return self.real_net(abs),self.imag_net(phase)
    
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
        out_real,out_imag = self(src.real,src.imag)
        # ------------ Concatenate ------------
        output = torch.cat((torch.unsqueeze(out_real,2),torch.unsqueeze(out_imag,2)),dim=2)
        target = torch.cat((torch.unsqueeze(tgt.real,2),torch.unsqueeze(tgt.imag,2)),dim=2)
        
        #loss func
        loss  = self.loss_f(output,target)
        
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
        tgt        = x/tgt_abs_factor
        
        # ------------ Model Eval ------------
        out_real,out_imag = self(src.real,src.imag)
        # ------------ Concatenate ------------
        output = torch.cat((torch.unsqueeze(out_real,2),torch.unsqueeze(out_imag,2)),dim=2)
        target = torch.cat((torch.unsqueeze(tgt.real,2),torch.unsqueeze(tgt.imag,2)),dim=2)
        
        #loss func
        loss  = self.loss_f(output,target)
        
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 50):
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
           
            out_real,out_imag = self(src.real,src.imag)
            # ------------ Predict Loss ------------
            # ------------ Concatenate ------------
            output = torch.cat((torch.unsqueeze(out_real,2),torch.unsqueeze(out_imag,2)),dim=2)
            target = torch.cat((torch.unsqueeze(tgt.real,2),torch.unsqueeze(tgt.imag,2)),dim=2)
            loss  = self.loss_f(output,target)
            # ------------ Denormalize ------------
            #denormalize angle          
            # ------------ Polar trans ------------
            #transform output to polar
            x_hat    = torch.complex(out_real,out_imag)
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
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/Real_imag/lightning_logs/version_6/checkpoints/epoch=99-step=60000.ckpt')
    Cn = RealImag(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"RealImag",get_date_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    