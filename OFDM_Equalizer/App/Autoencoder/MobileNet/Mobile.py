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
NUM_EPOCHS = 30
QAM        = 16
TARGET     = "normal"

class PredesignedModel(nn.Module):
    def __init__(self, encoded_space_dim):
        super(PredesignedModel,self).__init__()
        # Load the pre-trained ResNet50 model
        self.model  = models.mobilenet_v3_small(weights=None)

        # Replace the final fully connected layer
        num_features = self.model.classifier[3].in_features
        # Modify this to the desired output size
        self.model.classifier[3] = nn.Linear(num_features, encoded_space_dim)
        
        # Replace the first convolutional layer
        num_input_channels = 1  # Modify this to the number of input channels
        new_conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.features[0][0] = new_conv1

        # Convert model to double
        self.model  = self.model.double() # Convert the root to double
        self.convert_to_double(self.model) # recursive convert
            
    def convert_to_double(self,module):
        for child_module in module.children():
            self.convert_to_double(child_module)
        if hasattr(module, 'float'):
            module.float()
        if hasattr(module, 'double'):
            module.double()

    def forward(self,data):
        z = self.model(data)
        return z

class ZeroForcing(pl.LightningModule,Rx_loader):
    
    def __init__(self, encoded_space_dim):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete") #Rx_loader constructor
        # Loss
        self.loss_f    = nn.MSELoss()
        self.abs_net   = PredesignedModel(encoded_space_dim)
        self.angle_net = PredesignedModel(encoded_space_dim)
        
        self.final_merge_abs = nn.Sequential(
            nn.Linear(encoded_space_dim*3,int(encoded_space_dim*2)),
            nn.Hardtanh(),
            nn.Linear(int(encoded_space_dim*2),encoded_space_dim)
        ).double()
        self.final_merge_ang = nn.Sequential(
            nn.Linear(encoded_space_dim*3,int(encoded_space_dim*2)),
            nn.Hardtanh(),
            nn.Linear(int(encoded_space_dim*2),encoded_space_dim),
        ).double()
        
    def forward(self, mag,angle):
        m = self.abs_net(mag)
        #m   = torch.cat((m_h,y_mag),dim=1)
        #m   = self.final_merge_abs(m)
        
        a = self.angle_net(angle)
        #a   = torch.cat((a_h,y_ang),dim=1)
        #a   = self.final_merge_abs(a)
        
        return m,a
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0001)
    
    def common_step(self,batch,predict=False):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Generate Y ------------
        self.SNR_db = self.SNR_db if predict else 40
        Y     = self.Get_Y(chann,x,conj=False,noise_activ=True)
        # Normalize Y
        #Y     = Y/torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        
        #Filter batches that are not outliers, borring batches
        valid_data, valid_indices   = self.filter_z_score_matrix(chann)
        if valid_data.numel() != 0:
            # Filtered data
            chann = valid_data
            Y     = Y[valid_indices]
            x     = x[valid_indices]
            
            # Only on batch case
            if valid_data.dim() == 1:
                Y     = torch.unsqueeze(Y, 0)
                chann = torch.unsqueeze(chann,0)
                x     = torch.unsqueeze(x,0)
                
            # ------------ Normalize Chann ------------
            # Normalize the first channel by dividing by max_abs
            chann         = torch.complex(chann[:, 0, :, :],chann[:, 1, :, :])
            max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
            chann         = chann / max_magnitude
            chann         = chann.unsqueeze(dim=1)
            
            max_magnitude = max_magnitude.squeeze()
            # ------------ Normalize x ------------
            x              = x/max_magnitude
            # ------------ Normalize y ------------
            Y = Y/max_magnitude
            
            # ------------ Input Data ------------
            magnitud = torch.abs(chann)
            angle    = (torch.angle(chann)+torch.pi)/(2*torch.pi)
            
            # ------------ Model  ------------
            m,a = self(magnitud,angle)
            
            #zero_forcing  = torch.polar(m,a)
            zero_forcing = Y/x
            target_abs = zero_forcing.abs()
            target_ang = (zero_forcing.angle()+torch.pi)/(2*torch.pi)
                
            #loss eval
            #loss        = self.loss_f(x,zero_forcing)
            loss = .5*self.loss_f(m,target_abs)+ \
                   .5*self.loss_f(a,target_ang)
            
            if(predict):
                zero_forcing = zero_forcing*max_magnitude
                x = x*max_magnitude
                self.SNR_calc(zero_forcing,x,norm=True)
                
        else: # block back propagation
            loss = torch.tensor([0.0],requires_grad=True).to(torch.float64).to(self.device)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        # Concatenate the tensors in outputs
        losses = [x['val_loss'] for x in outputs]
        concatenated = torch.cat([l.view(-1) for l in losses])
        # Compute the average loss
        avg_loss = concatenated.mean()
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
        if(batch_idx < 50):
            loss = self.common_step(batch,predict=True)
        return loss
        
if __name__ == '__main__':
   
    trainer = Trainer(accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/MobileNet/lightning_logs/version_4/checkpoints/epoch=39-step=48000.ckpt')
    model   = ZeroForcing(48)
    trainer.fit(model)
    #formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"ZeroForceMobileNet",get_time_string())
    #model.SNR_BER_TEST(trainer,formating)