import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from torch.nn.functional import tanh ,hardtanh

def complex_tanh(input):
    return tanh(input.real).type(torch.complex64)+1j*tanh(input.imag).type(torch.complex64)
def complex_hardtanh(input):
    return hardtanh(input.real).type(torch.complex64)+1j*hardtanh(input.imag).type(torch.complex64)


class ComplexNet(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(ComplexNet, self).__init__()
        self.fc1 = ComplexLinear(input_size, int(input_size*1.3))
        self.fc2 = ComplexLinear(int(input_size*1.3), hidden_size)
        self.fc3 = ComplexLinear(hidden_size, int(input_size*1.3))
        self.fc4 = ComplexLinear(int(input_size*1.3), input_size)
    
    def forward(self,x):
        x = self.fc1(x)
        x = complex_hardtanh(x)
        x = self.fc2(x)
        x = complex_hardtanh(x)
        x = self.fc3(x)
        x = complex_hardtanh(x)
        x = self.fc4(x)
        return x