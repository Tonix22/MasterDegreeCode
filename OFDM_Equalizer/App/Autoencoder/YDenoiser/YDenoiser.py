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
NUM_EPOCHS = 30
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 96 #48*1.5
#Optimizer
LEARNING_RATE = .001
EPSILON = .01

class YDenoiser(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        # Complex loss
        self.loss_f = nn.MSELoss()
        self.mag_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(2.0),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(2.0), input_size,bias=True),
            nn.Hardtanh()
        ).double()
        self.angle_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(2.0),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(2.0), input_size,bias=True),
            nn.Hardtanh(),
        ).double()
        
        self.loss_f = nn.MSELoss()
    
    def forward(self,abs,phase):        
        return self.mag_net(abs),self.angle_net(phase)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.001)
    
    def normalize(self,vect):
        abs_factor = torch.max(torch.abs(vect),dim=1, keepdim=True)[0]
        norm       = vect / abs_factor
        abs        = torch.abs(norm)
        angle      = (torch.angle(vect) + np.pi) / (2 * np.pi)
        return abs,angle
    
    def norm_x(sefl,vect):
        abs_factor = torch.max(torch.abs(vect),dim=1, keepdim=True)[0]
        norm       = vect / abs_factor
        return norm
    
    def denorm_ang(self,angle):
        return (angle*(2 * np.pi))- np.pi
    
    def build_polar(self,abs,ang):
        return torch.polar(abs,ang)
    
    def common_step(self,batch,predict = False):
        #SNR CALC
        if(predict == False):
            i = self.current_epoch
            self.SNR_db = 45 - 5 * (i % 5)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #Multiply X by the channel
        Y_noise = self.Get_Y(chann,x,conj=False,noise_activ=True)
        Y_clean = self.Get_Y(chann,x,conj=False,noise_activ=False)
        #norm data
        Y_n_abs,Y_n_ang = self.normalize(Y_noise)
        Y_c_abs,Y_c_ang = self.normalize(Y_clean)
        # Model
        Y_hat_abs,Y_hat_ang  = self(Y_n_abs,Y_n_ang)
        # Loss
        output = torch.cat((torch.unsqueeze(Y_hat_abs,2),torch.unsqueeze(Y_hat_ang,2)),dim=2)
        target = torch.cat((torch.unsqueeze(Y_c_abs,2),torch.unsqueeze(Y_c_ang,2)),dim=2)
        loss   = self.loss_f(output,target)
        
        if(predict == True):
            # Denorm output vector
            Y_hat_ang = self.denorm_ang(Y_hat_ang)
            # Polar
            y_hat  = self.build_polar(Y_hat_abs,Y_hat_ang)
            x_hat  = self.ZERO_X(chann,y_hat)
            x      = self.norm_x(x)
            self.SNR_calc(x_hat,x,norm=True)
        
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
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        loss = 0
        if(batch_idx < 100):
            loss = self.common_step(batch,predict = True)
        return loss
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/YDenoiser/lightning_logs/version_11/checkpoints/epoch=19-step=12000.ckpt')
    Dn = YDenoiser(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Dn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"YDenoiser",get_time_string())
    Dn.SNR_BER_TEST(trainer,formating)
    

    