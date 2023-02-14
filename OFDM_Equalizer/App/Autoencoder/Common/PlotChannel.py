import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_channel(AE,feature):

    idx = 10
    H_idx        = AE.data.H[:,:,idx]
    # Create complex tensor
    chann_real   = torch.tensor(H_idx.real).to(torch.float64).to(AE.device)
    chann_imag   = torch.tensor(H_idx.imag).to(torch.float64).to(AE.device)     
    chann = torch.complex(chann_real,chann_imag)
    # Normalize complex tensor
    max_magnitude = torch.max(torch.abs(chann),dim=1, keepdim=True)[0]
    chann         = chann / max_magnitude
    # Make image as 1 channel
    chann         = chann.unsqueeze(dim=0).unsqueeze(dim=0)
    # Select feature
    
    if(feature == "angle"):
        chann = (torch.angle(chann)+torch.pi)/(2*torch.pi) #input
        z,chann_hat   = AE(chann)
    if(feature == "abs"):
        chann = torch.abs(chann)
        z,chann_hat   = AE(chann)
    
    # Detach gradient and convert to numpy 
    chann     = chann.squeeze().detach().numpy()
    chann_hat = chann_hat.squeeze().detach().numpy()
    
    fig =  plt.figure()
    ax1  = plt.subplot2grid((1, 2), (0, 0))
    ax2  = plt.subplot2grid((1, 2), (0, 1))
    
    im1 = ax1.imshow(chann, cmap='viridis')
    im2 = ax2.imshow(chann_hat, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar2 = fig.colorbar(im2, ax=ax2)
    ax1.set_title("Chann")
    ax2.set_title("Chann Reconstruction")
    
    #plt.show()
    plt.savefig("./{}.png".format(AE.__class__.__name__))