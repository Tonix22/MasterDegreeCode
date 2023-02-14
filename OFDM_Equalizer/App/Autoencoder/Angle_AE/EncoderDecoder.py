import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
main_path   = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string

common_path = os.path.dirname(os.path.abspath(__file__))+"/../Common"
sys.path.insert(0, common_path)
from Conv_real import Encoder,Decoder
from PlotChannel import plot_channel

#Hyperparameters
BATCHSIZE  = 10
NUM_EPOCHS = 500
QAM        = 16
TARGET     = "normal"

class Angle_AE(pl.LightningModule,Rx_loader):
    
    def __init__(self, encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete") #Rx_loader constructor
        #self.loss_f = torch.nn.MSELoss()
        self.loss_f     = torch.nn.HuberLoss()
        self.encoder    = Encoder(encoded_space_dim)
        self.decoder    = Decoder(encoded_space_dim)
        self.target     = TARGET
        
    def forward(self, chann):
        z = self.encoder(chann)
        decoded_image = self.decoder(z)
        return z,decoded_image
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def common_step(self,batch):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Source Data ------------
        # Normalize the first channel by dividing by max_abs
        chann         = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
        max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
        chann         = chann / max_magnitude
        chann         = chann.unsqueeze(dim=1)
        
        # ------------ Target Data ------------
        if(self.target == "inv"):
            chann_tgt  = torch.linalg.inv(chann)
            chann_tgt = (torch.angle(chann_tgt)+torch.pi)/(2*torch.pi)
        else:
            chann_tgt = (torch.angle(chann)+torch.pi)/(2*torch.pi)
        
        
        chann = (torch.angle(chann)+torch.pi)/(2*torch.pi) #input
        #auto encoder
        z,chann_hat = self(chann)
        #loss eval
        loss        = self.loss_f(chann_hat,chann_tgt)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        return {'test_loss': loss}
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    trainer = Trainer(accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=False, max_epochs=NUM_EPOCHS,
                resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/Angle_AE/lightning_logs/version_14/checkpoints/epoch=299-step=360000.ckpt')
    model   = Angle_AE(48)
    trainer.fit(model)
    
    plot_channel(model,"angle")