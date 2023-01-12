import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexConvTranspose2d,ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from torch.nn.functional import tanh ,hardtanh

def complex_tanh(input):
    return torch.tanh(input.real).type(torch.complex64)+1j*torch.tanh(input.imag).type(torch.complex64)
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
        x = complex_tanh(x)
        x = self.fc2(x)
        x = complex_tanh(x)
        x = self.fc3(x)
        x = complex_tanh(x)
        x = self.fc4(x)
        return x
    
    
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.conv1 = ComplexConv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = ComplexConv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = ComplexBatchNorm2d(16)
        self.conv4 = ComplexConv2d(16, 32, 3, stride=2, padding=0)
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.L1 = ComplexLinear(800, 128)
        self.L2 = ComplexLinear(128, encoded_space_dim)
        
    def forward(self, x): # 1,1,48,48
        # cnn layers
        x = self.conv1(x) # 1,8,24,24
        x = complex_hardtanh(x)
        x = self.conv2(x) #1,16,12,12
        x = self.conv3(x)
        x = complex_hardtanh(x)
        x = self.conv4(x) #1,32,5,5
        x = complex_hardtanh(x)

        # flatten
        x = self.flatten(x)

        # latent vector
        x = self.L1(x)
        x = complex_hardtanh(x)
        z = self.L2(x)
        return z
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.L1 = ComplexLinear(encoded_space_dim, 128)
        self.L2 = ComplexLinear(128, 800)

        # unflatten 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 5, 5))

        self.conv1 = ComplexConvTranspose2d(32, 16, 3, stride=2, output_padding=1)
        self.conv2 = ComplexBatchNorm2d(16)
        self.conv3 = ComplexConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = ComplexBatchNorm2d(8)
        self.conv5 = ComplexConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        # from z to linear FC
        x = self.L1(z)
        x = complex_hardtanh(x)
        x = self.L2(x)
        x = complex_hardtanh(x)
        # unflatten
        x = self.unflatten(x)
        #Conv stage decoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = complex_hardtanh(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = complex_hardtanh(x)
        x = self.conv5(x)
        return x
    
    
