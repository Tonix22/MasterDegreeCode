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
BATCHSIZE  = 10
QAM        = 16
NUM_EPOCHS = 20
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 72 #48*1.5
#Optimizer
LEARNING_RATE = .001
EPSILON = .01

class GridTransformer(pl.LightningModule,Rx_loader):
    def __init__(self):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        step   = 1/QAM
        self.x_axis = torch.arange(-1, 1+step, step)
        self.y_axis = torch.arange(-1, 1+step,step)
        self.loss_f = nn.MSELoss()
        
    def forward(self,abs):        
        return self.mag_net(abs)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.0007,eps=.007)
    
    def grid_token(self,data):
        mask    = (self.x_axis[:-1] <= data) & (data <= self.x_axis[1:])
        point_x = mask.nonzero().squeeze()
        #Y point
        mask    = (self.x_axis[:-1] <= data) & (data <= self.x_axis[1:])
        point_y = mask.nonzero().squeeze()
        
      
    def training_step(self, batch, batch_idx):
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationGridTransformer
        Y     = self.Get_Y(chann,x,conj=True)
        
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y.abs()),dim=1, keepdim=True)[0]
        src_abs        = Y.abs() / src_abs_factor
        
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x.abs()),dim=1, keepdim=True)[0]
        tgt_abs        = x.abs()/tgt_abs_factor
        
        #model eval
        output_abs = self(src_abs)
        #loss func
        loss  = self.loss_f(output_abs,tgt_abs)
        
        self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #chann preparationGridTransformer
        Y     = self.Get_Y(chann,x,conj=True)
        
        #normalize factor, normalize by batch
        src_abs_factor = torch.max(torch.abs(Y.abs()),dim=1, keepdim=True)[0]
        src_abs        = Y.abs() / src_abs_factor
        
        #Normalize target 
        tgt_abs_factor = torch.max(torch.abs(x.abs()),dim=1, keepdim=True)[0]
        tgt_abs        = x.abs()/tgt_abs_factor
        
        #model eval
        output_abs = self(src_abs)
        #loss func
        loss  = self.loss_f(output_abs,tgt_abs)
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 100):
            chann, x = batch
            chann    = chann.permute(0,3,1,2)
            Y        = self.Get_Y(chann,x,conj=True)
            
            #normalize factor, normalize by batch
            src_abs_factor = torch.max(torch.abs(Y.abs()),dim=1, keepdim=True)[0]
            src_abs        = Y.abs() / src_abs_factor
            
            output_abs = self(src_abs)
            #de normalize
            output_abs  = output_abs*src_abs_factor
            
            #torch polar polar(abs: Tensor, angle: Tensor)
            x_hat    = torch.polar(output_abs,x.angle()).cpu().to(torch.float32)
            self.SNR_calc(x_hat,x) 
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/GridTransformer/lightning_logs/version_6/checkpoints/epoch=9-step=12000.ckpt')
    Cn = GridTransformer(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"GridTransformer",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    