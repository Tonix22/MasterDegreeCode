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
QAM        = 4
NUM_EPOCHS = 21
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 72 #48*1.5
#Optimizer
LEARNING_RATE = .001
EPSILON = .01
CONJ = False


class PhaseNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.angle_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            nn.Hardtanh(),
        ).double()    
        self.loss_f = nn.MSELoss()
    
    
    def forward(self,ang):        
        return self.angle_net(ang)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.0007)
    
    def common_step(self,batch,predict = False):
        if(predict == False):
            i = self.current_epoch
            self.SNR_db = 45 - 5 * (i % 8)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #Multiply X by the channel            
        if(predict == True):
            Y     = self.Get_Y(chann,x,conj=CONJ,noise_activ=True)
        else:
            Y     = self.Get_Y(chann,x,conj=CONJ,noise_activ=False)
        
        if(CONJ == False):
            Y     = self.ZERO_X(chann,Y)
        
        #normalize angle
        src_ang = (torch.angle(Y) + np.pi) / (2 * np.pi)
        #Normalize target 
        tgt_ang = (torch.angle(x) + np.pi) / (2 * np.pi)
        
        #model eval
        output_ang = self(src_ang)
        #loss func
        loss  = self.loss_f(output_ang,tgt_ang)
        
        if(predict == True):
            #torch polar polar(abs: Tensor, angle: Tensor)
            output_ang = output_ang*(2 * np.pi)-np.pi
            x_hat    = torch.polar(torch.ones(output_ang.shape).to(torch.float64)*.7,output_ang)
            self.SNR_calc(x_hat,x)
        
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
        loss = 0
        if(batch_idx < 200):
            loss = self.common_step(batch,predict = True)
        return loss 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=False, max_epochs=NUM_EPOCHS,
                resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/lightning_logs/version_8/checkpoints/epoch=20-step=25200.ckpt')
    Cn = PhaseNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PhaseNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    