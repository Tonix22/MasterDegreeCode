import torch
from math import sqrt
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.f1      = nn.Tanh()
        #hidden Layers
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.f2      = nn.Tanh()
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.f3      = nn.Tanh()
        #final Layer
        self.linear4 = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        #Input
        out = self.linear1(x)
        out = self.f1(out)
        #Hidden
        out = self.linear2(out)
        out = self.f2(out)
        
        out = self.linear3(out)
        out = self.f3(out)
        #Final 
        out = self.linear4(out)
        return out

class Chann_EQ_Net(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super(Chann_EQ_Net, self).__init__()
        self.stage1 = nn.Sequential(
                          nn.Linear(input_size, 2*int(input_size/2)**2),
                          nn.Tanh(),
                          nn.Unflatten(0,(1,2,int(input_size/2),int(input_size/2)))
        )
        self.linear_size = int(input_size/2)-5-11-5-11-1-5+6
        #input_size-5-1
        self.stage2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=5, stride=1),#48-5+1 = 44
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=11, stride=1),#43-11+1  = 34
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=5, stride=1),#34-5+1 = 30
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=11, stride=1),#30-11+1  = 20
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1),#20-1+1 = 20
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=1),#20-5+1   = 16
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