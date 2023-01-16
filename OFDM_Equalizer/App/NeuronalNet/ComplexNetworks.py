import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexConvTranspose2d,ComplexBatchNorm2d
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
    
    
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(Encoder,self).__init__()
        
        ### Convolutional section
        #http://layer-calc.com/
        #(1,48,48)->(4,48,48)
        self.conv1 = ComplexConv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = ComplexConv2d(4, 8, 3, stride=2, padding=1)
        #NORM
        self.norm1 = ComplexBatchNorm2d(8)
        
        #(4,48,48)->(8,24, 24)
        self.conv3 = ComplexConv2d(8, 16, 3, stride=2, padding=1)
        #(8,24, 24)->(16,12, 12)
        self.norm2 = ComplexBatchNorm2d(16)
        
        #(16,12, 12)->(32,5, 5)
        self.conv4 = ComplexConv2d(16, 32, 3, stride=2, padding=0)
        #(32,5, 5) -> (48,3, 3)
        self.conv5 = ComplexConv2d(32, 48, 3, stride=1, padding=0)
        #NORM
        self.norm3 = ComplexBatchNorm2d(48)
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.L1 = ComplexLinear(432, 128)
        self.L2 = ComplexLinear(128, encoded_space_dim)
        
    def forward(self, x): 
        # cnn layers
        x = self.conv1(x)
        x = complex_tanh(x)
        x = self.conv2(x)
        x = self.norm1(x) #NORM
        
        #conv 3
        x = self.conv3(x) 
        x = complex_tanh(x)
        x = self.norm2(x) #NORM
        
        #conv 4,5
        x = self.conv4(x)
        x = complex_tanh(x)
        x = self.conv5(x)
        x = complex_tanh(x)
        x = self.norm3(x) #NORM
        # flatten
        x = self.flatten(x)

        # latent vector
        x = self.L1(x)
        x = complex_hardtanh(x)
        z = self.L2(x)
        return z
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(Decoder,self).__init__()
        self.L1 = ComplexLinear(encoded_space_dim, 128)
        self.L2 = ComplexLinear(128, 800)
        # unflatten 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(48, 3, 3))
        #(Hin-1)*stride+(k-1)+op+1
        #(48, 3, 3)->(32, 5, 5)
        self.conv1 = ComplexConvTranspose2d(48, 32, 3, stride=1, output_padding=0)
        self.conv2 = ComplexBatchNorm2d(32)
        #(32, 5, 5)->(16,12,12)
        self.conv3 = ComplexConvTranspose2d(32, 16, 3, stride=2,output_padding=1)
        self.conv4 = ComplexBatchNorm2d(16)
        #(16,12,12)->(8,24, 24)
        self.conv4 = ComplexConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        #(8,24, 24)->(4,48,48)
        self.conv5 = ComplexConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1)
        #(4,48,48)->(1,48,48)
        self.conv6 = ComplexConvTranspose2d(4, 1, 3, stride=1, padding=1, output_padding=0)
        
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
    
    
class Encode_plus_data(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encode_plus_data,self).__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)
        self.fc1     = ComplexLinear(encoded_space_dim*2, int(encoded_space_dim*1.5))
        self.fc2     = ComplexLinear(int(encoded_space_dim*1.5), encoded_space_dim)
        
    def forward(self, chann,y):
        latent    = self.encoder(chann)
        chann_hat = self.decoder(latent)
        concat    = torch.cat((latent, y), dim=0)
        out       = self.fc1(concat)
        out       = complex_tanh(out)
        out       = self.fc2(out)
        return chann_hat,out
        
        
        