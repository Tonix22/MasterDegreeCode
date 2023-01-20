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

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX,Rx_loader

#Hyperparameters
BATCHSIZE  = 10
NUM_EPOCHS = 500

#Calculate dimension withs this tool 
#https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()
        
        self.conv_section = nn.Sequential(    
            #(2,48,48)   -> (4, 48, 48)
            nn.Conv2d(2, 4, 3, stride=1, padding=1),
            #(4, 48, 48) -> (4, 45, 45)
            nn.MaxPool2d(4,stride=1),
            nn.BatchNorm2d(4),
            #(4, 45, 45) -> (8, 23, 23)
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            #(8, 23, 23) -> (8, 20, 20)
            nn.MaxPool2d(4,stride=1),
            nn.ReLU(),
            #(8,20,20)   -> (16,10,10)
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #(16,10,10)  -> (32,4,4)
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
        )
        self.linear_section = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4*4*32, 128),
            nn.ReLU()
        )
        
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N  = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = self.conv_section(x)
        x = self.linear_section(x)
        
        #first order moment
        mu    =  self.linear2(x)
        #second order moment
        sigma = torch.exp(self.linear3(x))
        #reprametrization trick
        z = mu + sigma*self.N.sample(mu.shape)
        #internal kullback liber divergence
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        return z
    
#Nmist experiment with d = 4
class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 4 * 4 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))

        self.decoder_conv = nn.Sequential(
            #(32,4,4)  -> (16,11,11)
            nn.ConvTranspose2d(32, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardtanh(),
            #(16,11,11)  -> (8,24,24)
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=0),
            nn.Hardtanh(),
            #(8,24,24) -> (4,48,48)
            nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0),
            #(4,48,48) -> (2,48,48)
            nn.ConvTranspose2d(4, 2, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
class VariationalAutoencoder(pl.LightningModule,Rx_loader):
    def __init__(self, latent_dims):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,NUM_EPOCHS) #Rx_loader constructor
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return z,self.decoder(z)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(), lr=1e-4,eps=1e-3,weight_decay=1e-4)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #auto encoder
        z,chann_hat = self(chann)
        loss = ((chann - chann_hat)**2).sum() + self.encoder.kl
        
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #auto encoder
        z,chann_hat = self(chann)
        loss = ((chann - chann_hat)**2).sum() + self.encoder.kl
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
if __name__ == '__main__':
    trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=100)],auto_lr_find=True, max_epochs=NUM_EPOCHS)
    #MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
    model   = VariationalAutoencoder(48)
    #trainer.fit(model)
    #checkpoint = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder_VAE/lightning_logs/version_1/checkpoints/epoch=499-step=600000.ckpt')
    #print(checkpoint.keys())