from sympy import init_printing
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
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
        self.imag_stage = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
        self.QPSK = nn.Sequential(
            nn.Linear(input_size*2, int(input_size*1.5)),
            nn.ReLU(),
            nn.Linear(int(input_size*1.5), int(input_size)),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        out1 = self.real_stage(x1)
        out2 = self.imag_stage(x2)
        out  = torch.cat((out1,out2))
        out  = self.QPSK(out)
        return out
        

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        fading1 = int(input_size*.7)
        fading2 = int(input_size*1.4)
        self.denoiser = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, num_classes),
            nn.Tanh()
        )
        """
        self.linear1 = 
        self.f1      = nn.Tanh()
        #hidden Layers
        self.linear2 = nn.Linear(fading1, fading2)
        self.f2      = nn.Tanh()
        self.linear3 = nn.Linear(fading2, fading3)
        self.f3      = nn.ReLU()
        #final Layer
        self.linear4 = nn.Linear(fading3, num_classes)
        self.f4      = nn.Tanh()
        #final Layer
        self.linear4 = nn.Linear(fading3, num_classes)
        self.f4      = nn.Tanh()
        """
    
    def forward(self, x):
        #Input
        return self.denoiser(x)


class QAMDemod(nn.Module):
    def __init__(self, input_size, num_classes):
        super(QAMDemod, self).__init__()
        fading1 = int(input_size*.8)
        fading2 = int(input_size*.6)
        self.linear1 = nn.Linear(input_size, fading1)
        self.f1 = nn.ReLU()
        # hidden Layers
        self.linear2 = nn.Linear(fading1, fading2)
        self.f2 = nn.ReLU()
        # final Layer
        self.linear3 = nn.Linear(fading2, num_classes)

    def forward(self, x):
        # Input
        out = self.linear1(x)
        out = self.f1(out)
        # Hidden
        out = self.linear2(out)
        out = self.f2(out)
        # Final
        out = self.linear3(out)
        return out

class Chann_EQ_Net(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super(Chann_EQ_Net, self).__init__()
        self.stage1 = nn.Sequential(
                          nn.Linear(input_size, input_size**2),
                          nn.Tanh(),
                          nn.Unflatten(0,(1,1,input_size,input_size))
        )
        self.linear_size = int(input_size/2)-5-11-5-11-1-5+6
        #input_size-5-1
        self.stage2 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=1),#48-5+1 = 44
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=11, stride=1),#43-11+1  = 34
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, stride=1),#34-5+1 = 30
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=11, stride=1),#30-11+1  = 20
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1),#20-1+1 = 20
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=1),#20-5+1    = 16
        )
        self.stage5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16,10*10),
            nn.Tanh(),
            nn.Linear(10*10,num_classes)
        )
        
        
    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        return torch.squeeze(out)