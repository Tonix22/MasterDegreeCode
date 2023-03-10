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
from Scatter_plot_results import ComparePlot

#Hyperparameters
BATCHSIZE     = 50
NUM_EPOCHS    = 50
LEARNING_RATE = 0.0001
QAM        = 16

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
        return torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
    
    def common_step(self,batch,predict=False):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        # ------------ Generate Y ------------
        self.SNR_db = self.SNR_db if predict else 30
        Y     = self.Get_Y(chann,x,conj=True,noise_activ=True)
        # Normalize Y
        #Y     = Y/torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        
        #Filter batches that are not outliers, borring batches
        chann         = torch.complex(chann[:,0,:, :],chann[:,1,:, :])
        
        # Log scale abs
        chann_abs       = chann.abs()
        max_magnitude   = (chann_abs.max() - chann_abs.min())
        chann_abs       = (chann_abs - chann_abs.min()) / max_magnitude
        # Filter magnitud by score
        valid_data, valid_indices   = self.filter_chann_diag_z_score(chann_abs)
        
        if valid_data.numel() != 0:
            # ------------ Input Data ------------
            magnitud = torch.unsqueeze(valid_data,1).abs()
            angle    = torch.unsqueeze((torch.angle(chann[valid_indices])+torch.pi)/(2*torch.pi),1)
            # Target
            Y     = Y[valid_indices]
            x     = x[valid_indices]
            
            # Only on batch case
            if magnitud.dim() == 3:
                Y     = torch.unsqueeze(Y, 0)
                chann = torch.unsqueeze(chann,0)
                x     = torch.unsqueeze(x,0)
                            
            # ------------ Normalize x ------------
            tgt_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            x         = x/tgt_abs_factor
            # ------------ Normalize y ------------
            Y_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
            Y            = Y / Y_abs_factor
            
            # ------------ Model  ------------NUM_EPOCHS
            m,a = self(magnitud,angle)
                
            a = (a*torch.pi*2)-torch.pi # normalize angle
            
            #Sum Phase of Y with angle estimate and save into a polar representation of radius 1
            ang_real = torch.cos(a+Y.angle())
            ang_imag = torch.sin(a+Y.angle())
            
            #compare angle complex values
            target_angle = torch.polar(torch.ones(x.shape).to(torch.float64).to(self.device),torch.angle(x))   
            
            loss = .5*self.loss_f(m*Y.abs(),x.abs())+ \
                   .25*self.loss_f(target_angle.real,ang_real)+.25*self.loss_f(target_angle.imag,ang_imag)
            
            if(predict):
                # ------------ Network output------------
                # Denormalize
                Y_ang = Y.angle()+a
                Y_abs = Y.abs()*m
                # ------------ Zero Forcing------------
                x_hat = torch.polar(Y_abs,Y_ang)
                self.BER_cal(x_hat,x,norm=True)
                
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
        if(batch_idx < 40):
            loss = self.common_step(batch,predict=True)
        return loss
        
if __name__ == '__main__':
   
    trainer = Trainer(accelerator='cpu',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/MobileNet/models/version_77/checkpoints/epoch=399-step=96000.ckpt')
    model   = ZeroForcing(48)
    print(model)
    trainer.fit(model)
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"ZeroForceMobileNet",get_time_string())
    model.SNR_BER_TEST(trainer,formating)