import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
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

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"MagNet")
sys.path.insert(0, main_path+"PhaseNet")

from MagNet import MagEqualizer
from PhaseNet import PhaseEqualizer

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
INPUT_SIZE = 48
curr_path  = os.path.dirname(os.path.abspath(__file__))
MAG_PATH   = curr_path+'/models/Abs.ckpt'
ANGLE_PATH = curr_path+'/models/Phase.ckpt'

class PolarPreprocess():
    def __init__(self):
        self.mag_net   = MagEqualizer(INPUT_SIZE,120)
        self.angle_net = PhaseEqualizer(INPUT_SIZE,240)
        #load 
        state = torch.load(MAG_PATH)['state_dict']
        new_state_dict = {}
        for key, value in state.items():
            new_key = key.replace('mag_net.mag_net.', 'mag_net.')
            new_state_dict[new_key] = value
            
        self.mag_net.load_state_dict(new_state_dict)
        self.mag_net.eval()
        
        state = torch.load(ANGLE_PATH)['state_dict']
        new_state_dict = {}
        for key, value in state.items():
            new_key = key.replace('angle_net.angle_net.', 'angle_net.')
            new_state_dict[new_key] = value
        
        self.angle_net.load_state_dict(new_state_dict)
        self.angle_net.eval()
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            self.mag_net.cuda()
            self.angle_net.cuda()
        
    def forward(self,abs,phase):        
        return self.mag_net(abs),self.angle_net(phase)
    
    def PrepareData(self,chann,Y,class_caller):
        
        # ------------ Source Data Preprocesing ------------
        # ANGLE
        Y_ang        = (torch.angle(Y)) / (2 * torch.pi)
        # Maginutd recieved zero forcing
        Y            = class_caller.ZERO_X(chann,Y)
        Y_abs_factor = torch.max(torch.abs(Y),dim=1, keepdim=True)[0]
        Y            = Y / Y_abs_factor
        Y_abs        = Y.abs()
        
        # model eval
        output_abs,output_angle = self.forward(Y_abs,Y_ang)
        output_angle            = output_angle*(2 * torch.pi)
        
        return torch.polar(output_abs,output_angle)


    