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
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.denoiser = nn.Sequential(
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Hardtanh(),
            nn.Linear(hidden_size, hidden_size,bias=True),
            nn.Linear(hidden_size, input_size,bias=True),
            nn.Tanh(),
            nn.Linear(input_size, num_classes,bias=True),
            nn.Hardtanh(),
        )
    
    def forward(self, x):
        #Input
        return self.denoiser(x)


class Inverse_Net(nn.Module):
    
    def __init__(self, input_size):
        super(Inverse_Net, self).__init__()
        self.double()
        self.matrix_size = input_size
        self.stage1 = nn.Sequential(
            #(2,48,48)
            #input_size//4 = 12
            nn.Conv2d(2, 4, kernel_size=input_size//8),#(2,42,42)
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=input_size//8-3),# (2,54,54)
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=input_size//4-3),# (4,48,48)
            nn.ReLU(),
        ).double()
        
        
    def forward(self,inv_inside):
        out = self.stage1(inv_inside)#pseudoinverse
        return torch.squeeze(out)
    
    
#resnet 

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride =1,downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding =1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding =1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out =self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0  = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1  = self._make_layer(block, 64, layers[1], stride = 2)
        self.layer2  = self._make_layer(block, 64, layers[2], stride = 2)
        self.layer3  = self._make_layer(block, 64, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc      = nn.Sequential(nn.Linear(256, 512),
                                     nn.Linear(512, 1024),
                                     nn.Linear(1024, 4608),
                                     nn.Unflatten(1, (2, 48, 48))
                                     )
        
        
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride!=1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes,kernel_size=1,stride=stride),
                nn.BatchNorm2d(planes),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x[None, :, :, :]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.squeeze(x)
    

