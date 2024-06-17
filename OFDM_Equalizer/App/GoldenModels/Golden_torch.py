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
import sys

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import get_date_string, convert_to_path

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
ESTIM      = "OSIC"
METHOD     = "DFT_spreading" # Complete DFT_spreading

class Golden(pl.LightningModule,Rx_loader):
    def __init__(self,mode="MSE",method = METHOD,qam = QAM,batchSize = BATCHSIZE):
        internaldevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,method,internaldevice)   
        self.mode   = mode
        
        if(self.mode == "MSE"):
            self.estimator = self.MSE_X
        elif(self.mode == "LMSE"):
            self.estimator = self.LMSE_X
        elif(self.mode == "ZERO"):
            self.estimator = self.ZERO_X
        elif(self.mode == "OSIC"):
            self.estimator = self.OSIC_X
        elif(self.mode == "NML"):
            self.estimator = self.NML_X
        
    def forward(self, x):        
        return x
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters())    
    
    def predict_step(self, batch, batch_idx):
        chann, x = batch
        chann    = chann.permute(0,3,1,2)
        Y        = self.Get_Y(chann,x)

        self.start_clock()
        
        x_hat    = self.estimator(chann,Y)
        
        self.stop_clock(BATCHSIZE)
        
        self.BER_cal(x_hat,x) #Actually calculate BER
        return 0
            
    def predict_dataloader(self):
        return self.test_loader
        
        
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=2)],enable_checkpointing=False)
    
    modes   = ["NML"]
    methods = ["Complete","DFT_spreading"]
    
    for currentMode in modes:
        for currenMehod in methods:
            Gold   = Golden(mode=currentMode,method=currenMehod)
            pathPreamble = "Test_Golden_{}QAM_{}".format(QAM,currentMode)
            Gold.SNR_BER_TEST(trainer,pathPreamble)