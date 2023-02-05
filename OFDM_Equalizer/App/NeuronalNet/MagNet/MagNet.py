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
BATCHSIZE  = 20
QAM        = 16
NUM_EPOCHS = 70
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

class AngleMagnitud(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.mag_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(2),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(2), input_size,bias=True),
            nn.Hardtanh()
        ).double()
        self.angle_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(2),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(2), input_size,bias=True),
            nn.Hardtanh(),
        ).double()
        
        self.loss_f = nn.MSELoss()
        
    def forward(self,abs,phase):        
        return self.mag_net(abs),self.angle_net(phase)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.0007,eps=.007)
        
    def training_step(self, batch, batch_idx):
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationAnglePhaseNet
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE)
        
        # ------------ Source Data ------------
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        src_abs        = Y / src_abs_factor
        src_abs        = src_abs.abs()
        src_angle_factor = (torch.angle(Y) + np.pi) / (2 * np.pi)
        
        # ------------ Target Data ------------
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
        tgt_abs        = x/tgt_abs_factor
        tgt_abs        = tgt_abs.abs()    
        tgt_angle_factor = (torch.angle(x) + np.pi) / (2 * np.pi)
        
        #model eval
        output_abs,output_angle = self(src_abs,src_angle_factor)
        output = torch.cat((torch.unsqueeze(output_abs,2),torch.unsqueeze(output_angle,2)),dim=2)
        target = torch.cat((torch.unsqueeze(tgt_abs,2),torch.unsqueeze(tgt_angle_factor,2)),dim=2)
        #loss func
        loss  = self.loss_f(output,target)
        
        self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationAnglePhaseNet
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE)
        
        # ------------ Source Data ------------
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        src_abs        = Y / src_abs_factor
        src_abs        = src_abs.abs()
        src_angle_factor = (torch.angle(Y) + np.pi) / (2 * np.pi)
        
        # ------------ Target Data ------------
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
        tgt_abs        = x/tgt_abs_factor
        tgt_abs        = tgt_abs.abs()    
        tgt_angle_factor = (torch.angle(x) + np.pi) / (2 * np.pi)
        
        #model eval
        output_abs,output_angle = self(src_abs,src_angle_factor)
        output = torch.cat((torch.unsqueeze(output_abs,2),torch.unsqueeze(output_angle,2)),dim=2)
        target = torch.cat((torch.unsqueeze(tgt_abs,2),torch.unsqueeze(tgt_angle_factor,2)),dim=2)
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
            src_abs        = Y / src_abs_factor
            src_abs        = src_abs.abs()
            src_angle_factor = (torch.angle(Y) + np.pi) / (2 * np.pi)
            
            # ------------ Target Data ------------
            #Normalize target 
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            tgt_abs        = x/tgt_abs_factor
            tgt_abs        = tgt_abs.abs()
            tgt_angle_factor = (torch.angle(x) + np.pi) / (2 * np.pi)
            
            # ------------ Model Eval ------------
            #normalize factor, normalize by batch            
            output_abs,output_angle = self(src_abs,src_angle_factor)
            # ------------ Predict Loss ------------
            output = torch.cat((torch.unsqueeze(output_abs,2),torch.unsqueeze(output_angle,2)),dim=2)
            target = torch.cat((torch.unsqueeze(tgt_abs,2),torch.unsqueeze(tgt_angle_factor,2)),dim=2)
            loss   = self.loss_f(output,target)
            # ------------ Denormalize ------------
            #denormalize angle
            output_angle     = output_angle*(2 * torch.pi)-torch.pi
            x_angle          = tgt_angle_factor*(2 * torch.pi)-torch.pi
            
            
            # ------------ Polar trans ------------
            #transform output to polar
            x_hat    = torch.polar(output_abs,output_angle)
            x_t      = torch.polar(tgt_abs,x_angle)
            
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
    
    trainer = Trainer(fast_dev_run=False,accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=40)],auto_lr_find=False, max_epochs=NUM_EPOCHS,
                resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/MagNet/lightning_logs/version_211/checkpoints/epoch=70-step=85200.ckpt')
    Cn = AngleMagnitud(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"AnglePhaseNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    