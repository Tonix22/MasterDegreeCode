from NeuronalNet.ComplexNetworks import Encode_plus_data
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

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX
from utils import get_time_string

BEST_SNR  = 45
WORST_SNR = 25
step      = -5

GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = -2

BATCHSIZE = 10

def Complex_MSE(output,target):
        return torch.sum((target-output).abs())
    
def Complex_MSE_polar(output,target):
    return torch.sum(torch.log(torch.pow(output.abs()/target.abs(),2))+torch.pow(output.angle()-target.angle(),2))

def y_awgn(H,x,SNR):
    # Assume noise singal is 1
    # Noise power
    Pn = 1 / (10**(SNR/10))
    # Generate noise
    noise = sqrt(Pn/2)* torch.complex(torch.randn(x.shape),torch.randn((x.shape)))
    # multiply tensors
    Y = torch.einsum("ijk,ik->ij", [H, x])
    Y +=noise
    return Y

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    iter_times = 0
    polar_loss_fn = Complex_MSE_polar
    for SNR in range(BEST_SNR,WORST_SNR-1,step):
        loop = tqdm(dataloader)
        iter_times+=1
        for idx, (chann, x) in enumerate(loop):
            
            Y = y_awgn(chann,x,45)
            chann = chann.unsqueeze(0)
            chann = chann.permute(1,0,2,3)
            chann_hat,x_hat = model(chann,Y)
                
            loss_chann = loss_fn(chann_hat, chann)
            loss_x     = loss_fn(x_hat,x) #TODO complex MSE polar

            opt.zero_grad()
            loss_chann.backward(retain_graph=True)
            loss_x.backward()
            opt.step()
            
            if(idx %50 == 0):
                loop.set_description(f"SNR [{SNR}]]")
                loop.set_postfix(loss=loss_chann.detach().item()+loss_x.detach().item())
    
            total_loss += loss_chann.detach().item()+loss_x.detach().item()
    return total_loss / (len(dataloader)*iter_times)

def validation_loop(model, dataloader,data):
    model.eval()
    iter_times = 0
    BER        = []
    for SNR in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP):
        loop = tqdm(dataloader)
        iter_times+=1
        errors = 0
        with torch.no_grad():
            for idx, (chann, x) in enumerate(loop):
                Y = y_awgn(chann,x,SNR)
                chann = chann.unsqueeze(0)
                chann = chann.permute(1,0,2,3)
                _,x_hat = model(chann,Y)
                tx_bits = data.Qsym.Demod(x_hat.cpu().detach().numpy()).astype(np.uint8)
                rx_bits = data.Qsym.Demod(Y.cpu().detach().numpy()).astype(np.uint8)
                errors+=np.unpackbits((tx_bits^rx_bits).view('uint8')).sum()
                
                if(idx %100 == 0):
                    loop.set_description(f"SNR [{SNR}]]")
                    loop.set_postfix(ber=errors/((data.bitsframe*data.sym_no)*len(dataloader)*BATCHSIZE))
        BER.append(errors/((data.bitsframe*data.sym_no)*len(dataloader)*BATCHSIZE))
            
    return BER # average loss

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
        train_loss_list += [train_loss]
        print("_"*10, f"validation","_"*10)
        validation_loss = validation_loop(model, val_dataloader,data)
        
        # Check if the file exists
        if os.path.isfile('Ber_val_data.csv'):
            # Read the existing CSV file into a DataFrame
            df = pd.read_csv('Ber_val_data.csv')
        else:
            # Create a new empty DataFrame
            df = pd.DataFrame()
    
        # Convert the list to a pandas Series
        s = pd.Series(validation_loss)
        # Append the Series to the DataFrame
        df = pd.concat([df, s], axis=1)
        # Save the DataFrame back to the CSV file
        df.to_csv('Ber_val_data.csv', index=False)

        
        print(f"Training loss: {train_loss:.4f}")
        #print(f"Validation loss: {validation_loss:.4f}")
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
 



if __name__ == '__main__':
    ### Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ### Define the loss function
    loss_fn = Complex_MSE
    ### Define an optimizer (both for the encoder and the decoder!)
    lr= 0.0001
    ### Set the random seed for reproducible results
    torch.manual_seed(0)
    ### Initialize the model
    d = 48
    model = Encode_plus_data(d)
    ### Optimizer
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5,eps=.001)

    #fix sed for split dataloader 
    torch.manual_seed(0)

    #load data with QAM 16
    data   = RX(16,"Norm")
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
    
    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True,num_workers=8)
    val_loader   = DataLoader(val_set,   batch_size=BATCHSIZE, shuffle=True,num_workers=8)
    test_loader  = DataLoader(test_set,  batch_size=BATCHSIZE, shuffle=True,num_workers=8)

    #define epochs
    epochs = 50
    #model fit
    train_loss_list, validation_loss_list = fit(model, optim, loss_fn, train_loader, val_loader, epochs,data)
    #save model
    formating = "Enconder_{}".format(get_time_string())
    torch.save(model.state_dict(),"{}/OFDM_Eq_{}.pth".format(formating))
    #test(model,loss_fn)