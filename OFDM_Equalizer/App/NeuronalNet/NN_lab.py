from operator import index
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


from   tqdm import tqdm
from   Networks import Inverse_Net,Linear_concat,AngleNet,MagNet,SymbolNet
from ComplexNetworks import ComplexNet
import pandas as pd
from datetime import datetime

from math import pi,sqrt
import os 
import sys

import torch.nn.functional as F

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"controllers")

from config import *
from Constants import *
from Recieved import RX


class NetLabs(object):
    def __init__(self, loss_type=MSE,
                       best_snr = 60,
                       worst_snr = 5,
                       toggle=False,
                       step = -1,
                       real_imag = REAL):
        
        #Noise gen and stepping
        self.BEST_SNR  = best_snr
        self.WORST_SNR = worst_snr
        self.step      = step
        #Image type
        self.real_imag = real_imag
        #which loss type
        self.loss_type = loss_type
        #Read RAW Data set
        self.data   = RX(16,"Unit_Pow")
        #Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Data symbols to be sent thought the channel
        self.N = self.data.sym_no
        #Training data advice
        self.training_data = int(self.data.total*.8)
        
        #Toggle between hard aproach and inverse aproach
        self.toggle = toggle
        
    #Custom Loss Functions
    def Complex_distance(self,output,target):
        y    = torch.column_stack((target.real,target.imag))
        loss = torch.sqrt(torch.pow(torch.dist(output[:,0], y[:,0], 1),2)+torch.pow(torch.dist(output[:,1], y[:,1], 1),2))
        return loss
    
    def Complex_MSE(self,output,target):
        return torch.sum((target-output).abs())
    
    def Complex_MSE_polar(self,output,target):
        return torch.sum(torch.log(torch.pow(output.abs()/target.abs(),2))+torch.pow(output.angle()-target.angle(),2))
        
    #Model Loader, Optimizer, and Loss type
    def Generate_Network_Model(self):
        NN = None
        
        #MODEL SELECT
        if(self.loss_type == MSE):
            if(self.real_imag == ANGLE or self.real_imag==REAL or self.real_imag==IMAG):
                NN  = AngleNet(input_size=self.N, hidden_size=int(self.N*1.5)).double()
            elif(self.real_imag == ABS):
                NN  = MagNet(input_size=self.N, hidden_size=int(self.N*1.5)).double()
            elif(self.real_imag == COMPLEX):
                NN  = ComplexNet(input_size=self.N,hidden_size=int(self.N*1.5))
        if(self.loss_type == MSE_COMPLETE):
            NN  = Linear_concat(input_size=self.N, hidden_size=int(self.N*3)).double()
        if(self.loss_type == MSE_INV):
            NN = Inverse_Net(input_size=self.N)
        if(self.loss_type == BCE):
            NN = SymbolNet(1<<self.data.bitsframe)
                
        NN = NN.to(self.device)
        
        # OPTIMIZER
        self.optimizer = optim.Adam(NN.parameters(),lr=LEARNING_RATE,eps=EPSILON)
        
        #LOSS TYPES
        if(self.loss_type == MSE_INV):
            self.criterion = nn.MSELoss().double()
        if(self.loss_type == MSE):
            if(self.real_imag == COMPLEX):
                self.criterion  = self.Complex_MSE
            else:
                self.criterion = nn.MSELoss()
            #self.criterion = nn.L1Loss()
        if(self.loss_type == MSE_COMPLETE):
            self.criterion = self.Complex_distance
            #self.criterion = nn.MSELoss()
        if(self.loss_type == CROSSENTROPY):
            #self.criterion = nn.CrossEntropyLoss()
            self.criterion = nn.MSELoss()
        if(self.loss_type == BCE):
            self.criterion = nn.BCELoss()
        return NN        

    def Generate_SNR(self,SNR,real_imag):
        r = None
        self.data.AWGN(SNR) #self.data.Qsym.r + noise
        #right side of equalizer
        Entry = np.empty((self.data.sym_no,self.data.total,1),dtype=self.data.Qsym.r.dtype)
        for i in range(0,self.data.total):
            Y = self.data.Qsym.r[:,i]
            H = np.matrix(self.data.H[:,:,i])
            Entry[:,i]=H.H@Y
        
        if(real_imag == COMPLEX):
            return torch.tensor(Entry, device  = torch.device(self.device)).type(torch.complex64)
        
        r_real = torch.tensor(Entry.real,device  = torch.device(self.device),dtype=torch.float64)
        r_imag = torch.tensor(Entry.imag,device  = torch.device(self.device),dtype=torch.float64)
        
        if(real_imag == REAL):      
           r = r_real
        if(real_imag == IMAG):
            r = r_imag
        if(real_imag == BOTH):
            r = torch.cat((r_real,r_imag),0)
        if(real_imag == ABS):
            r = torch.tensor(np.abs(Entry),device  = torch.device(self.device),dtype=torch.float64)
        if(real_imag == ANGLE):
            r = torch.tensor(np.angle(Entry)/pi,device  = torch.device(self.device),dtype=torch.float64)
        if(real_imag == FOUR):
            abs   = torch.tensor(np.abs(Entry),device  = torch.device(self.device),dtype=torch.float64)
            angle = torch.tensor(np.angle(Entry)/pi,device  = torch.device(self.device),dtype=torch.float64)
            r     = torch.stack((r_real,r_imag,angle,abs),dim=2)
        
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
        if(self.loss_type == CROSSENTROPY or self.loss_type == MSE_COMPLETE or self.loss_type == BCE):
            gt = torch.tensor(Truth,device = self.device,dtype=torch.float64)
        
        if(self.real_imag==BOTH and self.loss_type == MSE):#pseudoinverse or test 
            gt_real = torch.tensor(Truth.real,device = self.device,dtype=torch.float64)
            gt_imag = torch.tensor(Truth.imag,device = self.device,dtype=torch.float64)
            gt      = torch.cat((gt_real,gt_imag),0)
            del gt_real
            del gt_imag
            
        if(self.real_imag == INV):
        
            channels = np.concatenate((Truth.real,Truth.imag),axis=0)
            channels = np.reshape(channels,(2,self.data.sym_no,self.data.total))
            gt       = torch.from_numpy(channels).to(self.device)
            
        if(self.real_imag == COMPLEX):
            gt = torch.tensor(Truth,device  = torch.device(self.device)).type(torch.complex64)
            
        torch.cuda.empty_cache()
        return gt


    
