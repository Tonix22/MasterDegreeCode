import torch
import torch.nn as nn
import torch.nn.functional as F
    
class AutoencoderNN(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super(AutoencoderNN,self).__init__()
        self.enconder = nn.Sequential(
            nn.Conv2d(2, 4, 3, stride=1, padding=1), #(4, 48, 48)
            nn.MaxPool2d(4,stride=1),#(4, 45, 45)
            nn.Hardtanh(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),#(8, 23, 23)
            nn.Hardtanh(),
            nn.MaxPool2d(4,stride=1),#(8, 20, 20)
            nn.Conv2d(8, 16, 3, stride=2, padding=1),#(16, 10, 10)
            nn.Hardtanh(),
            nn.MaxPool2d(4,stride=1),
            nn.Hardtanh(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #(32, 3, 3)
            nn.Flatten(start_dim=1),
            nn.Linear(288, 128),
            nn.Hardtanh(),
            nn.Linear(128, encoded_space_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.Hardtanh(),
            nn.Linear(128, 288),
            nn.Hardtanh(),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(16,12, 4, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(8, 6, 3, stride=1, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(6, 4, 3, stride=1, padding=0),
            nn.Hardtanh(),
            nn.ConvTranspose2d(4, 2, 3, stride=1, padding=0)
        )
        
    def forward(self, x): 
        z   = self.enconder(x)
        out = self.decoder(z)
        return z,out
    
