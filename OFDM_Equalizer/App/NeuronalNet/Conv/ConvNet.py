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
from Recieved import RX,Rx_loader

#Hyperparameters
BATCHSIZE  = 10
#from 35 SNR to 5 takes 30 EPOCHS, so calculate epochs caount in this
NUM_EPOCHS = 300

GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2

class ConvNN(pl.LightningModule,Rx_loader):
    def __init__(self):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,16,"Complete")
        self.loss_f = self.Complex_MSE_polar
        
        #input enconder
        self.enconder = nn.Sequential(
            nn.Conv2d(2, 4, 3, stride=1, padding=1), #(4, 48, 48)
            nn.MaxPool2d(4,stride=1),#(4, 45, 45)
            nn.Hardtanh(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),#(8, 23, 23)
            nn.Hardtanh(),
            nn.MaxPool2d(4,stride=1),#(8, 20, 20)
            nn.Conv2d(8, 16, 3, stride=2, padding=1),#(16, 10, 10)
            nn.Hardtanh(),
            nn.MaxPool2d(4,stride=1),
            nn.Hardtanh(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #(32, 3, 3)
            nn.Flatten(start_dim=1),
            nn.Linear(288, 96),
            nn.Hardtanh(),
            nn.Unflatten(dim=1, unflattened_size=(2, 48)) # 2 channels of lenght 48
        )
        
        #input is 1,48
        self.H_x_mix = nn.Sequential
        (
            nn.LayerNorm(),
            nn.Conv1d(4, 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(2, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.radius_linear = nn.Linear(48,48)
        self.angle_linear  = nn.Linear(48,48)
        
        #Communications stuff
        self.SNR_db   = 35
        self.errors   = 0
        self.bits_num = 0
        self.snr_db_values = [(150,40), (180,30), (200,20), (220,10),(240,5)]
        self.BER = 0
        
    def forward(self,H,y):
        H      = self.enconder(H) #Batch,Chann,Lenght
        y_real = y.real 
        y_imag = y.imag
        concat = torch.cat((H,y_real,y_imag),dim=1)
        cross  = self.H_x_mix(concat)
        abs    = self.radius_linear(cross)
        angle  = self.angle_linear(cross)*2*torch.pi
        IQ     = torch.polar(abs,angle)
        return IQ
        
    def Complex_MSE_polar(self,output,target):
        return torch.sum(torch.pow((torch.log(target.abs()/output.abs()),2))+torch.pow(target.angle()-output.angle(),2))
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
        
    def Get_Y(self,H,x):
        Y = torch.zeros((BATCHSIZE,48),dtype=torch.complex128).to(self.device)
        for i in range(BATCHSIZE):
            #0 real,1 imag
            z = torch.complex(H[i,0,:, :],H[i,1,:, :])@x[i]
            # Signal Power
            Ps = torch.mean(torch.abs(z)**2)
            # Noise power
            Pn = Ps / (10**(self.SNR_db/10))
            # Generate noise
            noise = torch.sqrt(Pn/2)* (np.random.randn(self.sym_no,1) + 1j*np.random.randn(self.sym_no,1))
            Y[i] = z+noise
            Y[i] = Y[i]/torch.max(torch.abs(Y[i])) #normalize magnitud
        return Y
    
    def training_step(self, batch, batch_idx):
        self.SNR_db = (self.current_epoch%30+5)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        Y     = self.Get_Y(chann,x)
        #auto encoder
        x_hat = self(chann,Y) #model eval
        loss        = self.loss_f(x_hat,x)
        
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        Y     = self.Get_Y(chann,x)
        #auto encoder
        x_hat = self(chann,Y) #model eval
        loss  = self.loss_f(x_hat,x)
        
        self.log("val_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 100):
            chann, x = batch
            chann    = chann.permute(0,3,1,2)
            Y        = self.Get_Y(chann,x)
            x_hat    = self(chann,Y)
            for n in range(BATCHSIZE):
                rx     = x_hat[n].cpu().detach().numpy()
                rxbits = self.data.Qsym.Demod(rx)
                txbits = self.data.Qsym.Demod(x.cpu().detach().numpy())
                self.errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                self.bits+=(self.data.bitsframe*self.data.sym_no)
            
            #calculate BER
            self.BER = self.errors/self.bits
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=True, max_epochs=NUM_EPOCHS)
    Cn = ConvNN()
    trainer.fit(Cn)
    
    """
    for n in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP*-1):
        Cn.error  = 0
        Cn.BER    = 0
        Cn.SNR_db = n
        trainer.predict(tf)
        print("SNR:{} BER:{}".format(tf.SNR_db,tf.BER), file=open('BER_SNR.txt', 'a'))
    """