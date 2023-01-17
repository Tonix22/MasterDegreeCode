import os
import os.path
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from math import sqrt
import matplotlib.pyplot as plt
from  tqdm import tqdm
import numpy as np
import pandas as pd
from EncoderDecoder import AutoencoderNN

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
sys.path.insert(0, main_path+"App/NeuronalNet")

from Recieved import RX
from utils import get_time_string

BEST_SNR  = 45
WORST_SNR = 25
step      = -5

GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = -20

BATCHSIZE = 10

def Complex_MSE(output,target):
        return torch.sum((target-output).abs())
    
def Complex_MSE_polar(output,target):
    return torch.sum(torch.log(torch.pow(output.abs()/target.abs(),2))+torch.pow(output.angle()-target.angle(),2))

def y_awgn(H,x,SNR):
    Y = torch.einsum("ijk,ik->ij", [H, x])
    # Signal Power
    Ps = (torch.sum(torch.abs(Y)**2))/torch.numel(Y)
    # Noise power
    Pn = Ps / (10**(SNR/10))
    # Generate noise
    noise = sqrt(Pn/2)* torch.complex(torch.randn(x.shape),torch.randn((x.shape))).to(H.device)
    # multiply tensors
    
    Y = Y + noise
    return Y

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    #Total loss
    total_loss = 0
    loop = tqdm(dataloader)
    #train loop
    for idx, (chann, x) in enumerate(loop):
        
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #auto encoder
        z,chann_hat = model(chann)
        #calculate loss
        loss_chann = loss_fn(chann_hat, chann)
        #optimzer 
        opt.zero_grad()
        loss_chann.backward()
        opt.step()
        #description
        if(idx %50 == 0):
            loop.set_postfix(loss=loss_chann.detach().item())
        
        #calculating total loss
        total_loss += loss_chann.detach().item()
        
    return total_loss / (len(dataloader))

def validation_loop(model, dataloader,data):
    #models    
    model.eval()
    loop = tqdm(dataloader)
    total_loss = 0
    with torch.no_grad():
        for idx, (chann, x) in enumerate(loop):
            #chann preparation
            chann = chann.permute(0,3,1,2)
            #auto encoder
            z,chann_hat = model(chann)
            #calculate loss
            loss_chann = loss_fn(chann_hat, chann)
            #description
            if(idx %50 == 0):
                loop.set_postfix(loss=loss_chann.detach().item())
            
            #calculating total loss
            total_loss += loss_chann.detach().item()
        
    return total_loss / (len(dataloader))


def test(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    iter_times = 0
    for SNR in range(BEST_SNR,WORST_SNR-1,step):
        loop = tqdm(dataloader)
        iter_times+=1
        with torch.no_grad():
            for idx, (chann, x) in enumerate(loop):
                Y = y_awgn(chann,x,SNR)
                chann = chann.unsqueeze(0)
                chann = chann.permute(1,0,2,3)
            
                chann_hat,x_hat = model(chann,Y)
                
                loss_chann = loss_fn(chann_hat, chann)
                loss_x     = loss_fn(x_hat,x)
                total_loss += loss_chann.detach().item()+loss_x.detach().item()
                if(idx %100 == 0):
                    loop.set_description(f"SNR [{SNR}]]")
                    loop.set_postfix(loss=loss_chann.detach().item()+loss_x.detach().item())
            
    return total_loss / len(dataloader*iter_times) # average loss

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs,data):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list.append(train_loss)
        
        if((epoch+1) %5 == 0):
            train_loss_list += [train_loss]
            print("_"*10, f"validation","_"*10)
            validation_loss = validation_loop(model, val_dataloader,data)
            print(f"Validation loss: {validation_loss:.4f}")
            validation_loss_list.append(validation_loss)
        
        print(f"Training loss: {train_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list

def plot_results(train_loss_list,validation_loss_list):
    #PLOT results
    plt.plot(train_loss_list, label = "Train loss")
    plt.plot(validation_loss_list, label = "Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()

def plot_histogram(train_dataloader):
    bins = 50
    # Initialize an array to store the histogram
    histogram = torch.zeros(bins)

    # Iterate over the Dataloader
    for i, (chann,x) in enumerate(train_dataloader):
        #Compute the histogram
        histogram += torch.histc(chann.cpu(), bins=bins, min=-1, max=1)
    
    plt.bar(range(bins), histogram)
    plt.savefig('TrainingData_hist.png')


if __name__ == '__main__':
    ### Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ### Define the loss function
    loss_fn = torch.nn.MSELoss()
    ### Define an optimizer (both for the encoder and the decoder!)
    lr= 0.0005
    ### Set the random seed for reproducible results
    torch.manual_seed(0)
    ### Initialize the model
    d = 48
    model = AutoencoderNN(96)
    model.to(device)
    
    ### Optimizer
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5,eps=.005)

    #fix sed for split dataloader 
    torch.manual_seed(0)

    #load data with QAM 16
    data   = RX(16,"Unit_Pow")
    dataset     = data
    # Define the split ratios (training, validation, testing)
    train_ratio = 0.6
    val_ratio   = 0.2
    test_ratio  = 0.2
    # Calculate the number of samples in each set
    train_size = int(train_ratio * len(dataset))
    val_size   = int(val_ratio * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCHSIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCHSIZE, shuffle=True)
    
    plot_histogram(train_loader)
    
    """
    #define epochs
    epochs = 50
    #model fit
    train_loss_list, validation_loss_list = fit(model, optim, loss_fn, train_loader, val_loader, epochs,data)
    #save model
    formating = "Enconder_{}".format(get_time_string())
    torch.save(model.state_dict(),"OFDM_Eq_AutoEnconder_{}.pth".format(formating))
    #test(model,loss_fn)
    """