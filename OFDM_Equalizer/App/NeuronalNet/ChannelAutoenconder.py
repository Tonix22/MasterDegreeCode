from ComplexNetworks import Encoder,Decoder
import torch
import torch.nn as nn
import os 
import sys
from  tqdm import tqdm
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"controllers")
from config import *
from Constants import *
from Recieved import RX


def Complex_MSE(output,target):
    return torch.sum((target-output).abs())

def Complex_MSE_polar(output,target):
    return torch.sum(torch.log(torch.pow(output.abs()/target.abs(),2))+torch.pow(output.angle()-target.angle(),2))

### Define the loss function
loss_fn = Complex_MSE

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 48

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d)
decoder = Decoder(encoded_space_dim=d)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)
#load data with QAM 16
data   = RX(16,"Unit_Pow")

# Set train mode for both the encoder and the decoder
encoder.train()
decoder.train()
train_loss = []

#loop is the progress bar

#TRAIN
loop  = tqdm(range(0,int(data.total*.6)),desc="Progress")
for i in loop:
    chann = torch.from_numpy(data.H[:,:,i]).to(device)
    chann = chann[None,None,:,:]
    #### Encode data
    z = encoder(chann)
    decoded_image = decoder(z)
    # Evaluate loss
    loss = loss_fn(decoded_image, chann)

    # Backward pass
    optim.zero_grad()
    loss.backward()
    optim.step()
    if(i % 50 == 0):
        loop.set_postfix(loss=loss.cpu().detach().numpy())
    
#TEST    
encoder.eval()
decoder.eval() 
loop  = tqdm(range(int(data.total*.6),data.total),desc="Progress")
with torch.no_grad():
    input_image = []
    recon_image = []
    for i in loop:
        chann = torch.from_numpy(data.H[:,:,i]).to(device)
        chann = chann[None,None,:,:]
        z = encoder(chann)
        # Decode data
        decoded_image = decoder(z)
        val_loss = loss_fn(chann, decoded_image)
        #print loss
        if(i % 50 == 0):
            loop.set_postfix(loss=loss.cpu().detach().numpy())
            
        #Append overall
        recon_image.append(decoded_image.cpu()) # AUtoenconder output
        input_image.append(chann.cpu()) # Autoenconder input
        
    input_image = torch.cat(input_image)
    recon_image = torch.cat(recon_image) 
        
    val_loss = loss_fn(input_image, recon_image)
    
