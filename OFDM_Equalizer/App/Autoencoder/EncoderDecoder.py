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
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string

#Hyperparameters
BATCHSIZE  = 10
NUM_EPOCHS = 37
QAM        = 4

class AutoEncoder(nn.Module):
    def __init__(self,encoded_space_dim):
        super(AutoEncoder, self).__init__()
        #Networks
        self.enconder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1), #(4, 48, 48)
            nn.MaxPool2d(4,stride=1),#(4, 45, 45)
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),#(8, 23, 23)
            nn.Tanh(),
            nn.MaxPool2d(4,stride=1),#(8, 20, 20)
            nn.Conv2d(8, 16, 3, stride=2, padding=1),#(16, 10, 10)
            nn.Tanh(),
            nn.MaxPool2d(4,stride=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #(32, 3, 3)
            nn.Flatten(start_dim=1),
            nn.Linear(288, 128),
            nn.Linear(128, encoded_space_dim)
        ).double()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.Hardtanh(),
            nn.Linear(128, 288),
            nn.Hardtanh(),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(16,12, 4, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(8, 6, 3, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.Hardtanh(),
            nn.ConvTranspose2d(6, 4, 3, stride=1, padding=0),
            nn.ConvTranspose2d(4, 1, 3, stride=1, padding=0)
        ).double()
        
    def forward(self, x):
        z   = self.enconder(x)
        out = self.decoder(z)
        return out


class AutoencoderNN(pl.LightningModule,Rx_loader):
    
    def __init__(self, encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete") #Rx_loader constructor
        #self.loss_f = torch.nn.MSELoss()
        self.loss_f    = torch.nn.HuberLoss()
        self.abs_net   = AutoEncoder(encoded_space_dim)
        self.angle_net = AutoEncoder(encoded_space_dim)

        
    def forward(self, abs,angle):
        return self.abs_net(abs),self.angle_net(angle)
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Source Data ------------
        # Normalize the first channel by dividing by max_abs
        chann      = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
        #normalize abs
        chann      = chann*673.631 # factor to get inverse with abs max of 1
        chann_abs  = torch.abs(chann).unsqueeze(dim=1)
        #normalize angle
        chann_ang  = (torch.angle(chann) + torch.pi) / (2 * torch.pi)
        chann_ang  = chann_ang.unsqueeze(dim=1)
        
        # ------------ Target Data ------------
        chann_inv  = torch.linalg.inv(chann)
        tgt_ang    = (torch.angle(chann_inv)+ torch.pi)/ (2 * torch.pi)
        chan_tgt   = torch.stack((torch.abs(chann_inv), tgt_ang), dim=1)
        
        #auto encoder
        abs_hat,angle_hat   = self(chann_abs,chann_ang)
        chann_hat = torch.cat((abs_hat, angle_hat), dim=1)
        #loss eval
        loss        = self.loss_f(chann_hat,chan_tgt)
        
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Source Data ------------
        # Normalize the first channel by dividing by max_abs
        chann      = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
        #normalize abs
        chann      = chann*673.631 # factor to get inverse with abs max of 1
        chann_abs  = torch.abs(chann).unsqueeze(dim=1)
        #normalize angle
        chann_ang  = (torch.angle(chann) + torch.pi) / (2 * torch.pi)
        chann_ang  = chann_ang.unsqueeze(dim=1)
        
        # ------------ Target Data ------------
        chann_inv  = torch.linalg.inv(chann)
        tgt_ang    = (torch.angle(chann_inv)+ torch.pi)/ (2 * torch.pi)
        chan_tgt   = torch.stack((torch.abs(chann_inv), tgt_ang), dim=1)
        
        #auto encoder
        abs_hat,angle_hat   = self(chann_abs,chann_ang)
        chann_hat = torch.cat((abs_hat, angle_hat), dim=1)
        #loss eval
        loss        = self.loss_f(chann_hat,chan_tgt)
        
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 400):
            # training_step defines the train loop. It is independent of forward
            chann, x = batch
            #chann preparation
            chann = chann.permute(0,3,1,2)
            #torch.linalg.inv(chann[0,:,:]*abs_factor[0])
            # ------------ Generate Y ------------
            Y        = self.Get_Y(chann,x,conj=False,noise_activ=False)
            
            # ------------ Source Data ------------
            # Normalize the first channel by dividing by max_abs
            chann      = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
            chann      = chann*673.631
            chann_abs  = torch.abs(chann).unsqueeze(dim=1)
            
            #normalize angle
            chann_ang  = (torch.angle(chann) + torch.pi) / (2 * torch.pi)
            chann_ang  = chann_ang.unsqueeze(dim=1)
            # ------------ Model Eval ------------
            abs_hat,ang_hat = self(chann_abs,chann_ang)
            #denormalize
            abs_hat = abs_hat*673.631
            ang_hat = (ang_hat*(2 * torch.pi))-torch.pi
            # ------------ Denormalize angle -----------
            chann_hat = torch.polar(abs_hat.squeeze(),ang_hat.squeeze())
            # ------------ Multiply by inverse ------------
            x_hat = torch.einsum('bij,bi->bj', chann_hat, Y)
            
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
                resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/lightning_logs/version_12/checkpoints/epoch=36-step=44400.ckpt')
    model   = AutoencoderNN(96)
    #checkpoint_file = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/lightning_logs/Autoencoderckpt/checkpoints/epoch=999-step=1200000.ckpt')
    #print(checkpoint_file.keys())
    #model.load_state_dict(checkpoint_file['state_dict'])
    trainer.fit(model)
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PolarNet",get_time_string())
    model.SNR_BER_TEST(trainer,formating)
    
    