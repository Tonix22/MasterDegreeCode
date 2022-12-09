import torch
from math import sqrt
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import normalize

class Linear_concat(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Linear_concat, self).__init__()
        self.denoiser_real = nn.Sequential(
            nn.Hardtanh(),
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            torch.nn.Hardtanh(min_val=- 1.20, max_val=1.2),
        )
        self.denoiser_imag = nn.Sequential(
            nn.Hardtanh(),
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            torch.nn.Hardtanh(min_val=- 1.20, max_val=1.2)
        )

    def forward(self, x1, x2):
        x1 = normalize(x1, p=2.0, dim = 0)
        x2 = normalize(x2, p=2.0, dim = 0)
        real = self.denoiser_real(x1)
        imag = self.denoiser_imag(x2)
        return torch.column_stack((real,imag))
        

class AngleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AngleNet, self).__init__()
        
        self.denoiser = nn.Sequential(
            nn.Hardtanh(),
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size*int(1.5),bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size*int(1.5), input_size,bias=True),
            nn.Hardtanh(),
        )
    
    def forward(self, x):
        #mean, std = torch.mean(x), torch.std(x)
        #t  = (x-mean)/std
        
        return self.denoiser(x)
    
class MagNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MagNet, self).__init__()
        
        self.denoiser = nn.Sequential(
            #nn.Hardtanh(),
            nn.Linear(input_size, int(input_size*1.3),bias=True),
            nn.Hardtanh(),
            nn.Linear(int(input_size*1.3), hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, int(input_size*1.3),bias=True),
            nn.Linear(int(input_size*1.3), input_size,bias=True),
        )
    
    def forward(self, x):
        mean, std = torch.mean(x), torch.std(x)
        x  = (x-mean)/std
        x = self.denoiser(x)
        return x


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
    
#from 16 QAM to beyond
class SymbolNet(nn.Module):
    def __init__(self,QAM):
        super(SymbolNet, self).__init__()
        
        self.denoiser = nn.Sequential(
            nn.Hardtanh(),
            nn.Linear(4, 8,bias=True),
            nn.Hardtanh(),
            nn.Linear(8, 8,bias=True),
            nn.Hardtanh(),
            nn.Linear(8, 16,bias=True),
            nn.Hardsigmoid(),
            nn.Linear(16, QAM,bias=True),
            nn.Hardsigmoid(),
        ).double()
    
    def forward(self, x):
        mean, std = torch.mean(x), torch.std(x)
        x  = (x-mean)/std
        x = self.denoiser(x)
        return x
