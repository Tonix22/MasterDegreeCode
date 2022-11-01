from operator import index
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from   Recieved import RX
from   tqdm import tqdm
from   Networks import QAMDemod,Chann_EQ_Net,Linear_concat,LinearNet
import pandas as pd
from datetime import datetime

from math import pi
from config import *

from Constants import *


#from torch.utils.tensorboard import SummaryWriter
class NetLabs(object):
    def __init__(self, loss_type=MSE,best_snr = 60,worst_snr = 5,toggle=False,step = -1):
        self.BEST_SNR  = best_snr
        self.WORST_SNR = worst_snr
        self.step      = step
        self.loss_type = loss_type
        #Data set read
        self.data   = RX()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Data numbers
        self.N = self.data.sym_no
        #Training data lenght is only 80%
        self.training_data = int(self.data.total*.8)
        self.toggle = toggle
        
    def get_time_string(self):
        current_time = datetime.now()
        day  = current_time.day
        mon  = current_time.month
        year = current_time.year
        hr   = current_time.time().hour
        mn   = current_time.time().minute
        return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)
    
    def BER_LOS(self,output,target):
        #discrete = output.type('torch.CharTensor').to(self.device)
        #target   = target.type('torch.CharTensor').to(self.device)
        loss     = torch.bitwise_xor(output,target)
        #return torch.mean((output - target)**2)
        #loss     = ((loss&1).sum()+(loss&2>>1).sum())
        loss = loss.type('torch.FloatTensor')
        loss.requires_grad = True
        loss = torch.mean((loss)**2)
        return loss
    
    def Generate_Network_Model(self):
        NN = None
        #MODEL COFING
        if(self.loss_type == MSE):
            NN  = LinearNet(input_size=self.N, hidden_size=2*self.N, num_classes=self.N)
            
        if(self.loss_type == CROSSENTROPY):
            NN  = QAMDemod(input_size=2*self.N,num_classes=self.data.sym_no)
            
        if(self.loss_type == MSE_COMPLETE):
            NN  = Linear_concat(input_size=self.N,hidden_size=2*self.N)
                
        NN = NN.to(self.device)
        
        self.optimizer = optim.Adam(NN.parameters(),lr=LEARNING_RATE,eps=EPSILON)
        
        if(self.loss_type == MSE):
            self.criterion = nn.MSELoss()
            #self.criterion = nn.L1Loss()
        if(self.loss_type == MSE_COMPLETE):
            #self.criterion = self.BER_LOS
            self.criterion = nn.MSELoss()
            
        if(self.loss_type == CROSSENTROPY):
            #self.criterion = nn.CrossEntropyLoss()
            self.criterion = nn.MSELoss()
        
        return NN        

    def Generate_SNR(self,SNR,real_imag):
        r = None
        self.data.AWGN(SNR)
        #right side of equalizer
        Entry = np.empty((self.data.sym_no,self.data.total,1),dtype=self.data.Qsym.r.dtype)
        for i in range(0,self.data.total):
            Y = self.data.Qsym.r[:,i]
            H = np.matrix(self.data.H[:,:,i])
            Entry[:,i]=H.H@Y
        
        r_real = torch.tensor(Entry.real,device  = torch.device('cuda'),dtype=torch.float64)
        r_imag = torch.tensor(Entry.imag,device  = torch.device('cuda'),dtype=torch.float64)
        if(real_imag == REAL):      
           r = r_real
        if(real_imag == IMAG):
            r = r_imag
        if(real_imag == BOTH):
            r = torch.cat((r_real,r_imag),0)
        if(real_imag == ABS):
            r = torch.tensor(np.abs(Entry),device  = torch.device('cuda'),dtype=torch.float64)
        if(real_imag == ANGLE):
            r = torch.tensor(np.angle(Entry)/pi,device  = torch.device('cuda'),dtype=torch.float64)   
        del Entry
        torch.cuda.empty_cache()
        return r
    
    def Get_ground_truth(self,Truth):#self.data.Qsym.GroundTruth
        #MAG and ANGLE
        if(self.real_imag == ABS):
            gt = torch.tensor(np.abs(Truth),device = self.device,dtype=torch.float64)
        if(self.real_imag == ANGLE):
            gt = torch.tensor(np.angle(Truth)/pi,device = self.device,dtype=torch.float64)
        #Real IMAG
        if(self.real_imag == REAL and self.loss_type == MSE):
            gt = torch.tensor(Truth.real,device = self.device,dtype=torch.float64)
        if(self.real_imag == IMAG and self.loss_type == MSE):
            gt = torch.tensor(Truth.imag,device = self.device,dtype=torch.float64)
        #Equalizer or crossentropy
        if(self.loss_type == CROSSENTROPY or self.loss_type == MSE_COMPLETE):
            gt = torch.tensor(Truth,device = self.device,dtype=torch.float64)
        
        if(self.real_imag==BOTH and self.loss_type == MSE):#pseudoinverse or test 
            gt_real = torch.tensor(Truth.real,device = self.device,dtype=torch.float64)
            gt_imag = torch.tensor(Truth.imag,device = self.device,dtype=torch.float64)
            gt      = torch.cat((self.gt_real,self.gt_imag),0)
            del self.gt_real
            del self.gt_imag
                          
        torch.cuda.empty_cache()
        return gt


    
