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
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader
from utils import vector_to_pandas, get_time_string

#Hyperparameters
BATCHSIZE  = 10
QAM        = 16
NUM_EPOCHS = 20
#NN parameters
INPUT_SIZE  = 48
HIDDEN_SIZE = 72 #48*1.5
#Optimizer
LEARNING_RATE = .001
EPSILON = .01

# Model hyperparameters
GRID_STEP = 1/8
embedding_size = 128 
num_heads      = 128 
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.10
max_len = 50
forward_expansion = 4
src_pad_idx       = 1


class GridTransformer(pl.LightningModule,Rx_loader):
    def __init__(self,
        embedding_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len):
        
        
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE,QAM,"Complete")
        self.snr_db_values = [(50,40), (100,30), (125,20), (130,10),(150,5)]
        #Grid config
        step = GRID_STEP
        # Define bins for the real and imaginary parts of the data
        self.binsx = torch.arange(-1, 1 + step, step)
        self.binsy = torch.arange(-1, 1 + step, step)
        # Create a 2D bin index matrix for encoding
        self.binxy = torch.arange(start=4, end=4 + (len(self.binsx) - 1) * (len(self.binsy) - 1)).view(len(self.binsx) - 1, len(self.binsy) - 1)
        # Create empty tensors for storing the indices of the bins for the real and imaginary parts of the data
        self.x_indices = torch.zeros((BATCHSIZE, self.data.sym_no), dtype=torch.long)
        self.y_indices = torch.zeros((BATCHSIZE, self.data.sym_no), dtype=torch.long)
        # sos 2 eos 3
        self.sos = torch.full((BATCHSIZE, 1), fill_value=2)
        self.eos = torch.full((BATCHSIZE, 1), fill_value=3)
        
        self.loss_f     = torch.nn.CrossEntropyLoss()
    
        self.vocab_size = self.binxy.max()
        #Embedding section
        #src embedding
        self.src_word_embedding     = nn.Embedding(self.vocab_size , embedding_size)
        self.src_position_embedding = nn.Embedding(max_len,        embedding_size)
        #target embedding
        self.trg_word_embedding     = nn.Embedding(self.vocab_size , embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len,        embedding_size)
        
        #transfomer network
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out      = nn.Linear(embedding_size, self.vocab_size)
        self.dropout     = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self,src,trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )
            
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )
        
        src_padding_mask = self.make_src_mask(src)
        trg_mask         = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
    
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=.0007,eps=.007)
    
    #This function already does normalization 
    def grid_token(self,data):
        # Get the absolute maximum value of the data along the sequence dimension (dim=1)
        # Keep the first dimension with keepdim=True for broadcasting purposes
        src_abs_factor = torch.max(torch.abs(data),dim=1, keepdim=True)[0]

        # Normalize the data by dividing it with the maximum absolute value
        data = data/src_abs_factor
        
        # Loop over the bins for the real and imaginary parts
        for i in range(len(self.binsx) - 1):
            # Find the indices of the data that lie within the current bin for the real part
            self.x_indices[(self.binsx[i] <= data.real) & (data.real < self.binsx[i + 1])] = i
            # Same for imaginary part
            self.y_indices[(self.binsy[i] <= data.imag) & (data.imag < self.binsy[i + 1])] = i
        

        # Encode the data by selecting the corresponding bin indices from the bin index matrix
        encoded = self.binxy[self.y_indices, self.x_indices]
        encoded = torch.cat((self.sos,encoded,self.eos),dim=1)
        # Clear tensor for next encoding
        self.y_indices.zero_()
        self.x_indices.zero_()
        
        return encoded
      
    def SNR_select(self):
            
        if(self.current_epoch < 150):
            for lower, higher in self.snr_db_values:
                if (self.current_epoch) <= lower:
                    self.SNR_db = higher
                    break
        else:
            self.SNR_db = (self.SNR_db+1)%35+5
      
    def training_step(self, batch, batch_idx):
        #self.SNR_db = ((40-self.current_epoch*10)%41)+5
        self.SNR_select()
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        # Chann Formating
        chann = chann.permute(0,3,1,2) 
        # Prepare GridTransformer
        #TODO test with conj True and False
        Y     = self.Get_Y(chann,x,conj=False) 
        # Transform src values to grid tokens 
        src_tokens = self.grid_token(Y).permute(1,0) # [length,batch]
        
        #Get target tokens
        tgt_tokens = self.grid_token(x).permute(1,0) # [length,batch]
        
        #model eval transformer
        output = self(src_tokens,tgt_tokens[:-1, :])
        output = output.reshape(-1, output.shape[2]) # then we merge first two dim
        
        #loss func
        target = tgt_tokens[1:].reshape(-1)
        loss   = self.loss_f(output,target)
        
        self.log('SNR', self.SNR_db, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_loss", loss) #tensorboard logs
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        #self.SNR_db = ((40-self.current_epoch*10)%41)+5
        self.SNR_select()
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        # Chann Formating
        chann = chann.permute(0,3,1,2) 
        # Prepare GridTransformer
        #TODO test with conj True and False
        Y     = self.Get_Y(chann,x,conj=False) 
        # Transform src values to grid tokens 
        src_tokens = self.grid_token(Y).permute(1,0) # [length,batch]
        
        #Get target tokens
        tgt_tokens = self.grid_token(x).permute(1,0) # [length,batch]
        
        #model eval transformer
        output = self(src_tokens,tgt_tokens[:-1, :])
        output = output.reshape(-1, output.shape[2]) # then we merge first two dim
        
        #loss func
        target = tgt_tokens[1:].reshape(-1)
        loss   = self.loss_f(output,target)
        
        self.log('val_loss', loss) #tensorboard logs
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        if(batch_idx < 100):
            chann, x = batch
            chann    = chann.permute(0,3,1,2)
            Y        = self.Get_Y(chann,x,conj=True)
            
            #normalize factor, normalize by batch
            src_abs_factor = torch.max(torch.abs(Y.abs()),dim=1, keepdim=True)[0]
            src_abs        = Y.abs() / src_abs_factor
            
            output_abs = self(src_abs)
            #de normalize
            output_abs  = output_abs*src_abs_factor
            
            #torch polar polar(abs: Tensor, angle: Tensor)
            x_hat    = torch.polar(output_abs,x.angle()).cpu().to(torch.float32)
            self.SNR_calc(x_hat,x) 
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=False, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/GridTransformer/lightning_logs/version_6/checkpoints/epoch=9-step=12000.ckpt')
    tf = GridTransformer(
    embedding_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len)
    
    trainer.fit(tf)
    
    #name of output log file 
    #formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"GridTransformer",get_time_string())
    #Cn.SNR_BER_TEST(trainer,formating)
    

    