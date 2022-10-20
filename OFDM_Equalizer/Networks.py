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
        self.f2      = nn.ReLU()
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
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.stage1 = nn.Sequential(
                          nn.Linear(input_size, input_size*input_size),
                          nn.ReLU(),
                          nn.Unflatten(1, torch.Size([input_size, 2]))
        )
        #input_size-5-1
        self.stage2 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, kernel_size=5, stride=1),#48-6 = 42
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=11, stride=1),#42-11+1 = 32
        )
        self.stage3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 2, kernel_size=5, stride=1),#32-6 = 26
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=11, stride=1),#25-11+1 = 16
        )
        self.stage4 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16*16,10*10),
            torch.nn.RelU(),
            torch.nn.Linear(10*10,num_classes)
        )
        
        
    def forward(self, x):
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)