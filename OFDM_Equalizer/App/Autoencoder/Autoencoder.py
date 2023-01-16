import os 
import sys
import torch

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"App/NueronalNet")

from ComplexNetworks import Encoder,Decoder

class AutoEnconder():
    def __init__(self,
                 dim,
                 device):
        ### Define the loss function
        loss_fn = self.Complex_MSE

        ### Set the random seed for reproducible results
        torch.manual_seed(0)

        ### Initialize the two networks
        #model = Autoencoder(encoded_space_dim=encoded_space_dim)
        self.encoder = Encoder(encoded_space_dim=dim)
        self.decoder = Decoder(encoded_space_dim=dim)
        
        # Move both the encoder and the decoder to the selected device
        self.encoder.to(device)
        self.decoder.to(device)
    
    # Set train mode for both the encoder and the decoder
    def set_train(self):
        self.encoder.train()
        self.decoder.train()
        
    def set_eval (self):
        self.encoder.eval()
        self.decoder.eval() 
    
    

