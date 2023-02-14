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
from Networks import Encoder,Decoder # Complex autoencoder network
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning) 
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string

#Hyperparameters
BATCHSIZE  = 10
NUM_EPOCHS = 60
QAM        = 16
TARGET     = "normal"

class ComplexAE(pl.LightningModule,Rx_loader):
    
    def __init__(self, encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete") #Rx_loader constructor
        #self.loss_f = torch.nn.MSELoss()
        #self.loss_f    = torch.nn.HuberLoss()
        self.loss_f    = self.Complex_MSE
        self.encoder   = Encoder(encoded_space_dim)
        self.decoder   = Decoder(encoded_space_dim)
        self.target    = TARGET
    
    def forward(self,chann):
        z = self.encoder(chann)
        decoded_image = self.decoder(z)
        return z,decoded_image
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr= 0.0001,weight_decay=1e-5,eps=.005)
    
    def Complex_MSE(self,output,target):
        return torch.sum((target-output).abs())
    
    def training_step(self, batch, batch_idx):
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
            chann_tgt  = torch.linalg.inv(chann).unsqueeze(dim=1)
        else:
            chann_tgt = chann
        
        #auto encoder
        z,chann_hat   = self(chann)
        #loss eval
        loss        = self.loss_f(chann_hat,chann_tgt)
        
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
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
            chann_tgt  = torch.linalg.inv(chann).unsqueeze(dim=1)
        else:
            chann_tgt = chann
        
        #auto encoder
        z,chann_hat   = self(chann)
        #loss eval
        loss        = self.loss_f(chann_hat,chann_tgt)
        
        
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def plot_channel(self):

        idx = 10
        H_idx        = self.data.H[:,:,idx]
        
        chann_real   = torch.tensor(H_idx.real).to(torch.float64).to(self.device)
        chann_imag   = torch.tensor(H_idx.imag).to(torch.float64).to(self.device)     
        chann = torch.complex(chann_real,chann_imag)
        
        
        max_magnitude = torch.max(torch.abs(chann),dim=0, keepdim=True)[0]
        chann         = chann / max_magnitude
        chann         = chann.unsqueeze(dim=0).unsqueeze(dim=0)
        z,chann_hat   = self(chann)
        
        chann = chann.squeeze().detach().numpy()
        chann_hat = chann_hat.squeeze().detach().numpy()
        
        fig =  plt.figure()
        ax1  = plt.subplot2grid((1, 2), (0, 0))
        ax2  = plt.subplot2grid((1, 2), (0, 1))
        
        im1 = ax1.imshow(np.abs(chann), cmap='viridis')
        im2 = ax2.imshow(np.abs(chann_hat), cmap='viridis')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar2 = fig.colorbar(im2, ax=ax2)
        ax1.set_title("Chann")
        ax2.set_title("Chann Reconstruction")
        
        
        #plt.show()
        plt.savefig("./Channels_phase.png")
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 100):
            # training_step defines the train loop. It is independent of forward
            chann, x = batch
            #chann preparation
            chann = chann.permute(0,3,1,2)
            copy_chann    = chann.clone()
            # ------------ Generate Y ------------
            Y        = self.Get_Y(chann,x,conj=False,noise_activ=True)
            
            # ------------ Source Data ------------
            # Normalize the first channel by dividing by max_abs
            chann         = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
            max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
            chann         = chann / max_magnitude
            chann         = chann.unsqueeze(dim=1)
            
            # ------------ Target Data ------------
            if(self.target == "inv"):
                chann_tgt  = torch.linalg.inv(chann).unsqueeze(dim=1)
            else:
                chann_tgt = chann
            
            #auto encoder
            z,chann_hat   = self(chann)
            #loss eval
            loss        = self.loss_f(chann_hat,chann_tgt)
            
            if(self.target == "inv"):
                # ------------ Multiply by inverse ------------
                x_hat = torch.einsum('bij,bi->bj', chann_hat, Y)
            else:
                z = z*torch.squeeze(max_magnitude)
                x_hat = Y/z #ZERO FORCING equalizer
                #x_hat  = self.MSE_X(copy_chann,Y)
            
            self.SNR_calc(x_hat,x,norm=False) 
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    trainer = Trainer(accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=False, max_epochs=NUM_EPOCHS,
                      resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/ComplexAE/lightning_logs/version_1/checkpoints/epoch=59-step=72000.ckpt')
    model   = ComplexAE(48)
    trainer.fit(model)
    #model.plot_channel()
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"GridTransformer",get_time_string())
    model.SNR_BER_TEST(trainer,formating)