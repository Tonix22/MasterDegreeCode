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
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX,Rx_loader


#Hyperparameters
BATCHSIZE  = 10
#from 35 SNR to 5 takes 30 EPOCHS, so calculate epochs caount in this
NUM_EPOCHS = 10
REAL_EPOCS = 30*NUM_EPOCHS


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

class Transformer(pl.LightningModule,Rx_loader):
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
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE)#Rx_loader constructor

        self.constelation = num_tokens  #Values Range
        self.bitsframe    = int(math.log2(self.constelation))#Bits to Send 
        self.loss_f = torch.nn.CrossEntropyLoss()


        # INFO
        self.model_type = "Transformer"
        self.dim_model  = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        #contains num_tokens of dim_model size
        #Number of tokens QAM alphabet
        self.embedding   = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model = dim_model,
            nhead   = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    #TODO testme
    def y_awgn(self,H,x,SNR):
        real = 0
        imag = 1
        Yreal = torch.einsum('bjk,bk->bj', H[:, real, :, :], x[:, :, real]) - \
                torch.einsum('bjk,bk->bj', H[:, imag, :, :], x[:, :, imag])
                
        Yimag = torch.einsum('bjk,bk->bj', H[:, real, :, :], x[:, :, imag]) + \
                torch.einsum('bjk,bk->bj', H[:, imag, :, :], x[:, :, real])
        
        distance = torch.pow(Yreal,2) + torch.pow(Yimag,2)
        
        # Signal Power
        Ps = (torch.sum(distance))/torch.numel(Yreal)
        # Noise power
        Pn = Ps / (10**(SNR/10))
        # Generate noise
        noise_real = torch.sqrt(Pn/2)* torch.randn(x[:,:,real].shape,requires_grad=True)
        noise_imag = torch.sqrt(Pn/2)* torch.randn(x[:,:,imag].shape,requires_grad=True)
        # multiply tensors
        return torch.stack([Yreal + noise_real,Yimag + noise_imag],dim=2)
    
    def forward(self, src,tgt,H,SNR,tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.y_awgn(H,src,SNR)
        
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

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
        
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def training_step(self, batch, batch_idx):
        SNR = (30-self.current_epoch*5)%31 +5
        #x = torch.randint(0, self.constelation-1, (BATCHSIZE,self.data.sym_no), dtype=torch.int64)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann    = chann.permute(0,3,1,2)
        chann.requires_grad = True
        # Get mask to mask out the next words
        sequence_length = x.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)
        
        #auto encoder
        x_hat = self(x,x.clone(),chann,SNR,tgt_mask=tgt_mask) #model eval
        x_hat = x_hat.permute(1, 2, 0)
        loss  = self.loss_f(x_hat,x)
        
        self.log("train_loss", loss) #tensorboard logs
        self.log('SNR', SNR, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        SNR = (30-self.current_epoch)%31 +5
        #x = torch.randint(0, self.constelation-1, (BATCHSIZE,self.data.sym_no), dtype=torch.int64)
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        #chann preparation
        chann    = chann.permute(0,3,1,2)
        chann.requires_grad = True
        # Get mask to mask out the next words
        sequence_length = x.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)
        
        #auto encoder
        x_hat = self(x,x.clone(),chann,SNR,tgt_mask=tgt_mask) #model eval
        x_hat = x_hat.permute(1, 2, 0)
        loss  = self.loss_f(x_hat,x)
        
        self.log("train_loss", loss) #tensorboard log
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
        return self.test_loader

if __name__ == '__main__':
    
    trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=True, max_epochs=REAL_EPOCS)
    tf      = Transformer(num_tokens=16, dim_model=2, num_heads=2, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1) # 16QAM
    trainer.fit(tf)

