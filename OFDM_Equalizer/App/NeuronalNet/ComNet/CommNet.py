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
NUM_EPOCHS = 30
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 240 #48*1.5
#Optimizer
LEARNING_RATE = 1e-4 # 16 QAM 8e-5, 4QAM 4e-5
CONJ = True

class BiLSTMDetection(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(BiLSTMDetection, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, 2)  # Output: real + imag for each symbol

    def forward(self, x):
        # x shape: [batch, N*2] where N is the number of complex symbols
        batch_size = x.size(0)
        num_symbols = x.size(1) // 2  # infer number of complex symbols

        # Reshape from [batch, N*2] to [batch, N, 2] where [:, :, 0] = real, [:, :, 1] = imag
        x = x.view(batch_size, 2, num_symbols).transpose(1, 2)  # -> [batch, N, 2]

        # BiLSTM expects input [batch, seq_len, input_size] = [batch, N, 2]
        lstm_out, _ = self.bilstm(x)  # [batch, N, hidden*2]
        output = self.fc(lstm_out)    # [batch, N, 2]

        # Separate real and imag again
        real = output[:, :, 0]  # [batch, N]
        imag = output[:, :, 1]  # [batch, N]

        return real, imag

class PhaseNet(pl.LightningModule,Rx_loader):
    def __init__(self, input_size, hidden_size):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        
        self.angle_net = BiLSTMDetection()
        #self.loss_f    = nn.MSELoss()
        self.loss_f = self.distanceLoss
        
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
        
        #change different noises for different epoch
        if(predict == False):
            self.SNR_db = 40 - 5 * (self.current_epoch % 2)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #Multiply X by the channel
        Y     = self.Get_Y(chann,x,conj=True,noise_activ=True)
        
        #Filter batches that are not outliers, borring batches
        if(predict == False):
            valid_data, valid_indices   = self.filter_z_score(Y,threshold=3.0)
        else:
            valid_data, valid_indices   = self.filter_z_score(Y,threshold=3.0)
            
        if valid_data.numel() != 0:
            #normalize angle
            Y = valid_data
            X_zf = Y/chann 

            self.start_clock() #start time eval
            #model eval
            out_real, out_imag  = self(X_zf)

            #model End
            self.stop_clock(int(Y.shape[0])) #end time eval
            
            loss = self.loss_f(x[valid_indices].real,x[valid_indices].imag,out_real,out_imag)
    
            #torch polar polar(abs: Tensor, angle: Tensor)
            if(predict):
                x_hat    = torch.complex(out_real,out_imag)
                #ComparePlot(x,x_hat)
                self.BER_cal(x_hat,x[valid_indices])
        
        else: # block back propagation
            loss = torch.tensor([0.0],requires_grad=True).to(torch.float64).to(self.device)
        
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
    

    