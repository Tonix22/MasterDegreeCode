import torchvision.models as models
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
NUM_EPOCHS = 100
QAM        = 16
TARGET     = "normal"

class PredesignedModel(nn.Module):
    def __init__(self, encoded_space_dim):
        super(PredesignedModel,self).__init__()
        # Load the pre-trained ResNet50 model
        self.model  = models.resnet18(weights=None)
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        # Modify this to the desired output size
        self.model.fc = nn.Linear(num_features, encoded_space_dim)
        # Replace the first convolutional layer
        num_input_channels = 1  # Modify this to the number of input channels
        new_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = new_conv1
        
        # Convert all modules to float64
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or \
               isinstance(module, nn.Linear) or \
               isinstance(module, nn.BatchNorm2d):
                module.weight.data = module.weight.data.double()
                if(hasattr(module.bias, 'data') == True):
                    module.bias.data   = module.bias.data.double()
                if(hasattr(module, 'running_mean') == True):    
                    if module.running_mean is not None:
                        module.running_mean = module.running_mean.double()
                if(hasattr(module, 'running_var') == True):  
                    if module.running_var is not None:
                        module.running_var = module.running_var.double()

    def forward(self,data):
        z = self.model(data)
        return z

class ZeroForcing(pl.LightningModule,Rx_loader):
    
    def __init__(self, encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete") #Rx_loader constructor
        # Loss
        self.loss_f    = self.Complex_MSE
        self.abs_net   = PredesignedModel(encoded_space_dim)
        self.angle_net = PredesignedModel(encoded_space_dim)
        self.eq        = None
        
    def Complex_MSE(self,output,target):
        return torch.sum((target-output).abs())    
        
    def forward(self, mag,angle):
        m = self.abs_net(mag)
        a = self.angle_net(angle)
        return m,a
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def common_step(self,batch,predict=False):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Generate Y ------------
        Y     = self.Get_Y(chann,x,conj=False,noise_activ=False)
        
        # ------------ Source Data ------------
        # Normalize the first channel by dividing by max_abs
        chann         = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
        max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
        chann         = chann / max_magnitude
        chann         = chann.unsqueeze(dim=1)
        
        # ------------ Target Data ------------
        magnitud = torch.abs(chann)
        angle    = (torch.angle(chann)+torch.pi)/(2*torch.pi) #input
        #auto encoder
        m,a = self(magnitud,angle)
        denominator  = torch.polar(m,a)
        zero_forcing = Y/denominator
        #loss eval
        loss        = self.loss_f(x,zero_forcing)
        if(predict):
            self.eq = zero_forcing.clone()
            self.SNR_calc(self.eq,x,norm=False)
            
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
    
    def predict_dataloader(self):
        return self.test_loader
    
    def predict_step(self, batch, batch_idx):
        loss = 0
        if(batch_idx < 100):
            loss = self.common_step(batch,predict=True)
        return loss
        
if __name__ == '__main__':
    trainer = Trainer(accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/Angle_AE/lightning_logs/version_14/checkpoints/epoch=299-step=360000.ckpt')
    model   = ZeroForcing(48)
    trainer.fit(model)
    #formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"ZeroForceResNet",get_time_string())
    #model.SNR_BER_TEST(trainer,formating)