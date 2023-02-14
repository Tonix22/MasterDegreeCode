import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import get_time_string

#Hyperparameters
BATCHSIZE  = 100
QAM        = 16
ESTIM      = "ZERO"

class Golden(pl.LightningModule,Rx_loader):
    def __init__(self,mode="MSE"):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")        
        self.mode   = mode
        
        if(self.mode == "MSE"):
            self.estimator = self.MSE_X
        elif(self.mode == "LMSE"):
            self.estimator = self.LMSE_X
        elif(self.mode == "ZERO"):
            self.estimator = self.ZERO_X
        
    def forward(self, x):        
        return x
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters())    
    
    def predict_step(self, batch, batch_idx):
        chann, x = batch
        chann    = chann.permute(0,3,1,2)
        Y        = self.Get_Y(chann,x)
        x_hat    = self.estimator(chann,Y)
        self.SNR_calc(x_hat,x)
        return 0
            
    def predict_dataloader(self):
        return self.test_loader
        
        
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=2)],enable_checkpointing=False)
    Gold    = Golden(ESTIM)
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,Gold.mode,get_time_string())
    Gold.SNR_BER_TEST(trainer,formating)