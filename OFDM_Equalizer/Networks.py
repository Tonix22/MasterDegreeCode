import torch
from math import sqrt
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

class Linear_concat(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Linear_concat, self).__init__()
        self.fading1 = int(hidden_size*.7)
        self.real_stage = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size,bias=True),
            nn.Linear(hidden_size, input_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(input_size, input_size,bias=True),
        )
        self.imag_stage = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size,bias=True),
            nn.Linear(hidden_size, input_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(input_size, input_size,bias=True),
        )
        self.QPSK = nn.Sequential(
            nn.Linear(input_size*2, int(input_size*1.7)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_size*1.7), int(input_size*1.5)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_size*1.5), int(input_size)),
        )
        
    def forward(self, x1, x2):
        out1 = self.real_stage(x1)
        out2 = self.imag_stage(x2)
        out  = torch.cat((out1,out2))
        out  = self.QPSK(out)
        return out
        

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearNet, self).__init__()
        
        self.denoiser = nn.Sequential(
            nn.Hardtanh(),
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            nn.Hardtanh(),
        )
    
    def forward(self, x,SNR):
        return self.denoiser(x)


class Inverse_Net(nn.Module):
    
    def __init__(self, input_size):
        super(Inverse_Net, self).__init__()
        self.double()
        self.matrix_size = input_size
        self.stage1 = nn.Sequential(
            #(2,48,48)
            #input_size//4 = 12
            nn.Conv2d(2, 4, kernel_size=input_size//8+1,padding='same'),#no compress
            nn.Hardtanh(),
            nn.ConvTranspose2d(4,4,kernel_size=input_size//8),# (4,54,54) , + 6 
            nn.ConvTranspose2d(4,4,kernel_size=input_size//8),# (4,60,60) , + 6
            nn.Hardtanh(),
            nn.Conv2d(4, 4, kernel_size=input_size//8-3),# (2,54,54)
            nn.Hardtanh(),
            nn.Conv2d(4, 2, kernel_size=input_size//4-3),# (4,48,48)
            nn.Hardtanh(),
        ).double()
        
        
    def forward(self,inv_inside):
        out = self.stage1(inv_inside)#pseudoinverse
        return torch.squeeze(out)