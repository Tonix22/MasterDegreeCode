import torch
import torch.nn as nn
import numpy as np
import os
import sys
from  tqdm import tqdm
from complexPyTorch.complexLayers import ComplexLinear
from torch.nn.functional import tanh ,hardtanh

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")

from config import *
from Constants import *
from Recieved import RX
from utils import vector_to_pandas ,get_time_string

def Psuedo_inv(H):
    H = np.matrix(H)
    inv = np.linalg.inv(H.H@H)@H.H
    return np.squeeze(np.asarray(inv))
def Complex_MSE(output,target):
    return torch.sum((target-output).abs())
def complex_tanh(input):
    return tanh(input.real).type(torch.complex64)+1j*tanh(input.imag).type(torch.complex64)

input_size  = 48*48
hidden_size = input_size
output_size = input_size

# Define the neural network
class MatrixInverseNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MatrixInverseNet, self).__init__()
    self.fc1 = ComplexLinear(input_size, hidden_size)
    self.fc2 = ComplexLinear(hidden_size, output_size)
  
  def forward(self, x):
    x = self.fc1(x)
    x = complex_tanh(x)
    x = self.fc2(x)
    x = complex_tanh(x)
    return x

# Create the neural network and define the loss function and optimizer
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
model = MatrixInverseNet(input_size, hidden_size, output_size)
model.to(device)
lr= 0.001

loss_fn   = Complex_MSE
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=.0001)
data      = RX(4,"Unit_Pow")


#normalize channel data

# Calculate the magnitudes of the complex numbers
# Normalize the complex numbers by dividing each component by the magnitude
data.H   = data.H / np.abs(data.H)
#save invser in memory
H_inv = np.zeros(data.H.shape,dtype=data.H.dtype)
loop  = tqdm(range(0,data.total),desc="Progress")
for i in loop:
    H_inv[:,:,i] = np.linalg.inv(data.H[:,:,i])

#Normalize inverse
# save this as variable because we will use it from denormalize
normalize_factor = np.abs(H_inv) 
H_inv = H_inv / normalize_factor


# Train the model
num_epochs = 2
for epoch in range(num_epochs):
    loop  = tqdm(range(0,int(data.total*.6)),desc="Progress")
    for n in loop:
        input_matrix  = torch.from_numpy(data.H[:,:,n]).to(device)
        input_vector  = input_matrix.view(-1)
        target_matrix = torch.from_numpy(H_inv[:,:,n]).to(device)
        target_vector = target_matrix.view(-1)
        # Forward pass
        output_vector = model(input_vector)
        loss = loss_fn(target_vector, output_vector)
        
        if(n % 10 == 0):
            loop.set_description(f"EPOCH[{epoch}]")
            loop.set_postfix(loss=loss.cpu().detach().numpy())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


BER    = []
# Test the model
for SNR in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,-1*GOLDEN_STEP):
        loop   = tqdm(range(0,data.total),desc="Progress")
        errors = 0
        data.AWGN(SNR)
        for i in loop:
            #Get realization
            Y = data.Qsym.r[:,i]
            
            input_matrix  = torch.from_numpy(data.H[:,:,i]).to(device)
            input_vector  = input_matrix.view(-1)
            txbits = np.squeeze(data.Qsym.bits[:,i],axis=1)
            
            output_matrix = model(input_vector)
            inverse = output_matrix.view(48, 48).cpu().detach().numpy()
            inverse = inverse*normalize_factor
            X_hat   = inverse@Y
            
            rxbits = data.Qsym.Demod(X_hat)
            errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
            
            #Status bar and monitor  
            if(i % 500 == 0):
                loop.set_description(f"SNR [{SNR}]")
                loop.set_postfix(ber=errors/((data.bitsframe*data.sym_no)*data.total))
                
        BER.append(errors/((data.bitsframe*data.sym_no)*data.total))
        
vector_to_pandas("BER_Inv_layer_SNR{}.csv".format(get_time_string()),BER)
