import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from torch.nn.functional import tanh ,hardtanh

def complex_tanh(input):
    return torch.tanh(input.real).type(torch.float64)+1j*torch.tanh(input.imag).type(torch.float64)
def complex_hardtanh(input):
    return hardtanh(input.real).type(torch.float64)+1j*hardtanh(input.imag).type(torch.float64)

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(Encoder,self).__init__()
        
        ### Convolutional section
        #http://layer-calc.com/
        #(1,48,48)->(4,48,48)
        self.conv1 = ComplexConv2d(1, 4, 3, stride=1, padding=1).double()
        self.conv2 = ComplexConv2d(4, 8, 3, stride=2, padding=1).double()
        #NORM
        self.norm1 = ComplexBatchNorm2d(8).double()
        
        #(4,48,48)->(8,24, 24)
        self.conv3 = ComplexConv2d(8, 16, 3, stride=2, padding=1).double()
        #(8,24, 24)->(16,12, 12)
        self.norm2 = ComplexBatchNorm2d(16).double()
        
        #(16,12, 12)->(32,5, 5)
        self.conv4 = ComplexConv2d(16, 32, 3, stride=2, padding=0).double()
        #(32,5, 5) -> (48,3, 3)
        self.conv5 = ComplexConv2d(32, 48, 3, stride=1, padding=0).double()
        #NORM
        self.norm3 = ComplexBatchNorm2d(48).double()
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1).double()

        ### Linear section
        self.L1 = ComplexLinear(432, 128).double()
        self.L2 = ComplexLinear(128, encoded_space_dim).double()
        
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
        self.L1 = ComplexLinear(encoded_space_dim, 128).double()
        self.L2 = ComplexLinear(128, 432).double()
        # unflatten 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(48, 3, 3)).double()
        #(Hin-1)*stride+(k-1)+op+1
        #(48, 3, 3)->(32, 5, 5)
        self.conv1 = ComplexConvTranspose2d(48, 32, 3, stride=1, output_padding=0).double()
        self.conv2 = ComplexBatchNorm2d(32)
        #(32, 5, 5)->(16,12,12)
        self.conv3 = ComplexConvTranspose2d(32, 16, 3, stride=2,output_padding=1).double()
        self.conv4 = ComplexBatchNorm2d(16)#! ERROR FOUND TWICE CONV4
        #(16,12,12)->(8,24, 24)
        self.conv4 = ComplexConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1).double()
        #(8,24, 24)->(4,48,48)
        self.conv5 = ComplexConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1).double()
        #(4,48,48)->(1,48,48)
        self.conv6 = ComplexConvTranspose2d(4, 1, 3, stride=1, padding=1, output_padding=0).double()
        
    def forward(self, z):
        # from z to linear FC
        x = self.L1(z)
        x = complex_hardtanh(x)
        x = self.L2(x)
        x = complex_tanh(x)
        # unflatten
        x = self.unflatten(x)
        #Conv stage decoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = complex_tanh(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = complex_tanh(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

def apply_complex(fr, fi, input, dtype = torch.complex128):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConvTranspose2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
    def forward(self,input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


#custom implementation with complex 128
class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex128))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex128)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex128)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0,2,3])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps 
       
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

    
        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag).type(torch.complex128) \
                + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real).type(torch.complex128)

        if self.affine:
            input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
                    self.bias[None,:,0,None,None]).type(torch.complex128) \
                    +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
                    self.bias[None,:,1,None,None]).type(torch.complex128)

        return input
