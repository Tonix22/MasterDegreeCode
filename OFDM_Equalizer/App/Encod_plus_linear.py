from NeuronalNet.ComplexNetworks import Encode_plus_data
import os 
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from math import sqrt

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX

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
    
    for chann,x in dataloader:
        
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
    
        total_loss += loss_chann.detach().item()+loss_x.detach().item()
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for chann,x in dataloader:
            Y = y_awgn(chann,x,45)
            chann = chann.unsqueeze(0)
            chann = chann.permute(1,0,2,3)
        
            chann_hat,x_hat = model(chann,Y)
            
            loss_chann = loss_fn(chann_hat, chann)
            loss_x     = loss_fn(x_hat,x)
            total_loss += loss_chann.detach().item()+loss_x.detach().item()
            
    return total_loss / len(dataloader) # average loss

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list
    

def Complex_MSE(output,target):
        return torch.sum((target-output).abs())
    
def Complex_MSE_polar(output,target):
    return torch.sum(torch.log(torch.pow(output.abs()/target.abs(),2))+torch.pow(output.angle()-target.angle(),2))

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
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)

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
    
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=10, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=10, shuffle=True)

    #define epochs
    epochs = 10
    #model fit
    fit(model, optim, loss_fn, train_loader, val_loader, epochs)
