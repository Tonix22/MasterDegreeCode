import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX,Rx_loader


#Hyperparameters
BATCHSIZE  = 10
NUM_EPOCHS = 1000

#mandatory positional enconding
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding   = torch.zeros(max_len, dim_model)
        #position list is (N_rows and 1 column)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term  = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        # array[start:end:step]
        # even position , start at 0 step at 2
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        # odd position 
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model  = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model = 2, 
            dropout_p = dropout_p, 
            max_len   = 48)
        
        #contains num_tokens of dim_model size
        #2 is legacy from IQ data 
        self.embedding         = nn.Embedding(num_tokens, 2)
        self.transformer       = nn.Transformer(
            d_model            = dim_model,
            nhead              = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout            = dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def y_awgn(H,x,SNR):
        #TODO COMPLEX MULTPILE WITH INDEPENDENT STAGES
        Y = torch.einsum("ijk,ik->ij", [H, x])
        # Signal Power
        Ps = (torch.sum(torch.abs(Y)**2))/torch.numel(Y)
        # Noise power
        Pn = Ps / (10**(SNR/10))
        # Generate noise
        noise = torch.sqrt(Pn/2)* torch.complex(torch.randn(x.shape),torch.randn((x.shape))).to(H.device)
        # multiply tensors
        
        Y = Y + noise
        return Y
    
    def forward(self, src, tgt,H,SNR,tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(2)
        #src = H*src+w
        
        tgt = self.embedding(tgt) * math.sqrt(2)
        #positional enconder
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        #TODO CHECK IF DATA CAN FEED TRANSFORMER
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

class ComTransformer(pl.LightningModule,Rx_loader):
    def __init__(self,constelation):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE)#Rx_loader constructor
        self.constelation = constelation  #Values Range
        self.bitsframe    = int(math.log2(constelation))#Bits to Send 
        self.transformer  = Transformer(
            self.constelation, # Number of tokens
            48,                # Dimension model ->data lenght
            self.bitsframe,    # Number of heads
            5,# Enconder Layers 
            5,# Decoder  Layers
            dropout_p=0.1
        )
        
    def forward(self,data,H):
        
        
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def training_step(self, batch, batch_idx):
        data_tensor = torch.randint(0, self.constelation-1, (BATCHSIZE,self.data.sym_no), dtype=torch.long)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #auto encoder
        z,chann_hat = self(data_tensor,chann) #model eval
        loss        = self.loss_f(chann,chann_hat)
        
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann = chann.permute(0,3,1,2)
        #auto encoder
        z,chann_hat = self(chann)
        loss        = self.loss_f(chann,chann_hat)
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loade

tf = ComTransformer(16)

