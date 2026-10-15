import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
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
from utils import vector_to_pandas, get_date_string
from Scatter_plot_results import ComparePlot
import torch.optim.lr_scheduler as lr_scheduler

#Hyperparameters
BATCHSIZE  = 100
QAM        = 4
NUM_EPOCHS = 10
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 240 #48*1.5
#Optimizer
LEARNING_RATE = 1e-3 # 16 QAM 8e-5, 4QAM 4e-5
CONJ = True

class BiLSTMDetection(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2):
        super(BiLSTMDetection, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, 2)  # Output: real + imag for each symbol

    def forward(self, x):
        batch_size = x.size(0)
        num_symbols = x.size(1) // 2
        x = x.view(batch_size, 2, num_symbols).transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        output = self.fc(lstm_out)
        # Stack real over imag to shape [batch, num_symbols*2]
        output_stacked = output.reshape(batch_size, -1)
        return output_stacked

class PhaseNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete",torch.device("cuda"))
        
        self.angle_net = BiLSTMDetection()
        self.loss_f    = nn.MSELoss()
        #self.loss_f = self.distanceLoss
        
    def distanceLoss(self,real_target,imag_target,out_real,out_imag):
        return F.mse_loss(real_target,out_real)+F.mse_loss(imag_target,out_imag)
    
    def forward(self,ang):
        x = self.angle_net(ang)
        return x
    
    def configure_optimizers(self):
        start = LEARNING_RATE
        optimizer = torch.optim.Adam(self.parameters(),lr=start)  
        return [optimizer]
    
    
    def common_step(self,batch,predict = False):
        self.SNR_db = 45
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #Multiply X by the channel
        Y     = self.Get_Y(chann,x,conj=False,noise_activ=True)
        Y     = self.ZERO_X(chann,Y)
        
        #Filter batches that are not outliers, borring batches
        if(predict == False):
            valid_data, valid_indices   = self.filter_z_score(Y,threshold=3.0)
        else:
            valid_data, valid_indices   = self.filter_z_score(Y,threshold=3.0)
            
        if valid_data.numel() != 0:
            #normalize angle
            Y = valid_data
            # Separate real and imaginary parts
            real_part = Y.real  # Shape: (batch, feature)
            imag_part = Y.imag  # Shape: (batch, feature)

            # Stack real and imaginary parts along the last dimension
            X_zf_stacked = torch.cat((real_part, imag_part), dim=1).to(torch.float32)  # Shape: (batch, feature * 2)

            self.start_clock() #start time eval
            out_stacked = self(X_zf_stacked)  # [batch, 96]
            target_real = x[valid_indices].real.float()
            target_imag = x[valid_indices].imag.float()
            target_stacked = torch.cat([target_real, target_imag], dim=1).to(torch.float32)  # [batch, 96]

            loss = self.loss_f(out_stacked, target_stacked)

    
        if predict:
            # Separar de nuevo la predicción stackeada
            out_real = out_stacked[:, :out_stacked.shape[1]//2]
            out_imag = out_stacked[:, out_stacked.shape[1]//2:]
            
            # Formar complejos a partir de predicción
            x_hat = torch.complex(out_real, out_imag)
            
            # Calcular BER comparando predicción vs ground truth
            self.BER_cal(x_hat, x[valid_indices])
        
        else: # block back propagation
            loss = torch.rand(1, requires_grad=True, dtype=torch.float64, device=self.device)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        #self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('t_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #tensorboard logs
        self.log('SNR',self.SNR_db,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss':loss}
    
    def predict_step(self, batch, batch_idx):
        loss = 0
        loss = self.common_step(batch,predict = True)
        return loss 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    #trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=2)], max_epochs=NUM_EPOCHS)
    logger = TensorBoardLogger("tb_logs", name=f"PhaseNet{BATCHSIZE}_{NUM_EPOCHS}")
    trainer = Trainer(logger=logger, fast_dev_run=False, accelerator='gpu',
                    callbacks=[TQDMProgressBar(refresh_rate=10)], 
                    max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/lightning_logs/version_91/checkpoints/epoch=3-step=4800.ckpt')
    Cn = PhaseNet(INPUT_SIZE,HIDDEN_SIZE)
    trainer.fit(Cn)#,ckpt_path='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/tb_logs/PhaseNet100_30/version_7/checkpoints/epoch=29-step=3600.ckpt')
    
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"PhaseNet",get_date_string())
    Cn.SNR_BER_TEST(trainer,formating)
    

    