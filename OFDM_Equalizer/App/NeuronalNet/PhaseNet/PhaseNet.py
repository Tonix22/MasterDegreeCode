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
import torch.optim.lr_scheduler as lr_scheduler

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
NUM_EPOCHS = 2
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 240 #48*1.5
#Optimizer
LEARNING_RATE = 8e-5 # 16 QAM 8e-5, 4QAM 4e-5
CONJ = True

class PhaseEqualizer(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(PhaseEqualizer,self).__init__()
        self.angle_net = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(3),bias=True),
            nn.Linear(hidden_size*int(3), hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, input_size,bias=True)
        ).double()   

    def forward(self,abs): 
        return self.angle_net(abs)

class PhaseNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.angle_net = PhaseEqualizer(input_size,hidden_size)
        self.loss_f    = nn.MSELoss()
        #self.loss_f = self.distanceLoss
        
    def distanceLoss(self,real_target,imag_target,out_real,out_imag):
        return torch.mean(torch.sqrt(torch.pow(real_target-out_real,2)+torch.pow(imag_target-out_imag,2)))
    
    def forward(self,ang):
        x = self.angle_net(ang)
        return x
    
    def configure_optimizers(self):
        start = LEARNING_RATE
        optimizer = torch.optim.Adam(self.parameters(),lr=start)  
        return [optimizer]
    
    
    def common_step(self,batch,predict = False):
        if(predict == False):
            self.SNR_db = 40
            #i = self.current_epoch
            #self.SNR_db = 35 - 5 * (i % 5)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #Multiply X by the channel            
        if(predict == True):
            Y     = self.Get_Y(chann,x,conj=True,noise_activ=True)
        else:
            Y     = self.Get_Y(chann,x,conj=True,noise_activ=True)
        
        if(CONJ == False):
            Y     = self.ZERO_X(chann,Y)
        
        #Filter batches that are not outliers, borring batches
        valid_data, valid_indices   = self.filter_z_score(Y)
        if valid_data.numel() != 0:
            #normalize angle
            Y = valid_data
            src_ang = (torch.angle(Y)) / (2 * np.pi)
            
            #model eval
            out_ang  = self(src_ang)
            out_real = torch.cos(out_ang*torch.pi*2)
            out_imag = torch.sin(out_ang*torch.pi*2)
            # Unitary magnitud imaginary, only matters angle
            x = x[valid_indices]
            target     = torch.polar(torch.ones(x.shape).to(torch.float64).to(self.device),torch.angle(x))    

            #output = torch.stack((out_real,out_imag),dim=-1)
            #tgt    = torch.stack((target.real,target.imag),dim=-1)
            
            #loss func
            #loss  = self.loss_f(target.real,target.imag,out_real,out_imag)
            #loss  = self.loss_f(tgt,output)
            loss = .5*self.loss_f(target.real,out_real)+.5*self.loss_f(target.imag,out_imag)
        
            if(predict == True):
                #torch polar polar(abs: Tensor, angle: Tensor)
                x_hat    = torch.complex(out_real,out_imag)
            
                #ComparePlot(x,x_hat)
                self.SNR_calc(x_hat,target)
        
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
        # Concatenate the tensors in outputs
        losses = [x['val_loss'] for x in outputs]
        concatenated = torch.cat([l.view(-1) for l in losses])
        # Compute the average loss
        avg_loss = concatenated.mean()
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
    
    trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/lightning_logs/version_91/checkpoints/epoch=3-step=4800.ckpt')
    Cn = PhaseNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PhaseNet",get_time_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    