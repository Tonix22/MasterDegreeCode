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
        self.f1    = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.f2   = nn.Tanh()
        self.linear3 = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.f1(out)
        out = self.linear2(out)
        out = self.f2(out)
        out = self.linear3(out)
        return out
