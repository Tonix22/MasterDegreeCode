import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(Encoder,self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.Tanh(),
            nn.Conv2d(32, 48, 3, stride=1, padding=0),
            nn.Tanh(),
            nn.BatchNorm2d(48),
            nn.Flatten(start_dim=1),
            nn.Linear(432, 128),
            nn.Hardtanh(),
            nn.Linear(128, encoded_space_dim)
        ).double()

        
    def forward(self, x): 
        z = self.encode(x)
        return z


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.Hardtanh(),
            nn.Linear(128, 432),
            nn.Tanh(),
            nn.Unflatten(dim=1, unflattened_size=(48, 3, 3)),
            nn.ConvTranspose2d(48, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0, output_padding=0)
        ).double()


        
    def forward(self, z):
        x = self.decoder(z)
        return x