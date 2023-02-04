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
BATCHSIZE  = 20
QAM        = 16
NUM_EPOCHS = 144
#
LAST_LIST   = 250
CONJ_ACTIVE = True
#If use x or x_mse
GROUND_TRUTH_SOFT = False # fts
NOISE = False

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
        
        # ------------ Telecom Stuff ------------
        self.snr_db_values = [(80,40), (130,30), (170,20), (200,10),(LAST_LIST,5)]
        
        # ------------ Grid config ------------
        self.step = GRID_STEP
        
        #       ------------ Encode ------------
        
        # Define bins for the real and imaginary parts of the data
        self.binsx = torch.arange(-1, 1 + self.step, self.step)
        self.binsy = torch.arange(-1, 1 + self.step, self.step)
        # Create a 2D bin index matrix for encoding
        self.binxy = torch.arange(start=4, end=4 + (len(self.binsx) - 1) * (len(self.binsy) - 1)).view(len(self.binsx) - 1, len(self.binsy) - 1)
        # Create empty tensors for storing the indices of the bins for the real and imaginary parts of the data
        self.x_indices = torch.zeros((BATCHSIZE, self.data.sym_no), dtype=torch.long)
        self.y_indices = torch.zeros((BATCHSIZE, self.data.sym_no), dtype=torch.long)
        
        #       ------------ Decode ------------
        # Create empty tensors for storing the decoded values for the real and imaginary parts of the data
        self.real_decoded = torch.zeros(self.x_indices.shape,dtype=torch.float64)
        self.imag_decoded = torch.zeros(self.y_indices.shape,dtype=torch.float64)
        
        # ------------ Alphabet ------------
        
        # sos 2 eos 3
        self.sos = torch.full((BATCHSIZE, 1), fill_value=2)
        self.eos = torch.full((BATCHSIZE, 1), fill_value=3)
        
        # ----------- Network Sections -----
        
        #Loss
        self.loss_f     = torch.nn.CrossEntropyLoss()
        #soft_ground truth
        self.ground_truth = GROUND_TRUTH_SOFT

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
        return torch.optim.Adam(self.parameters(),lr=.0007,eps=.007,weight_decay=1e-5)
    
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
        
        return encoded.to(self.device)
    
    def grid_decode(self,encoded):
        # Loop over the bins for the real and imaginary parts
        for i in range(len(self.binsx) - 1):
            for j in range(len(self.binsy) - 1):
                # Find the indices of the encoded values that correspond to the current bin
                indices = (encoded == self.binxy[j, i])
                # Fill the decoded values for the real part
                self.real_decoded[indices] = self.binsx[i] + self.step/2
                # Fill the decoded values for the imaginary part
                self.imag_decoded[indices] = self.binsy[j] + self.step/2
        
        # Create a tensor for the decoded data
        data_decoded = torch.complex(self.real_decoded, self.imag_decoded)

        # Clear tensor for next decoding
        self.real_decoded.zero_()
        self.imag_decoded.zero_()
        
        return data_decoded
        
      
    def SNR_select(self):
            
        if(self.current_epoch < LAST_LIST):
            for lower, higher in self.snr_db_values:
                if (self.current_epoch) <= lower:
                    self.SNR_db = higher
                    break
        else:
            self.SNR_db = (self.SNR_db+1)%35+5
      
    def training_step(self, batch, batch_idx):
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        #self.SNR_select()
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        # Chann Formating
        chann = chann.permute(0,3,1,2) 
        # Prepare GridTransformer
        #TODO test with conj True and False
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE) 
        # Transform src values to grid tokens 
        src_tokens = self.grid_token(Y).permute(1,0) # [length,batch]
        
        #Get target tokens
        if(self.ground_truth == True):
            x = self.MSE_X(chann,Y)
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
        self.SNR_db = ((40-self.current_epoch*10)%41)+5
        #self.SNR_select()
        # training_step defines the train loop. It is independent of forward
        chann, x = batch
        # Chann Formating
        chann = chann.permute(0,3,1,2) 
        # Prepare GridTransformer
        #TODO test with conj True and False
        Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE) 
        # Transform src values to grid tokens 
        src_tokens = self.grid_token(Y).permute(1,0) # [length,batch]
        
        #Get target tokens
        if(self.ground_truth == True):
            x = self.MSE_X(chann,Y)
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
        
        if(batch_idx < 10):
            #self.SNR_db = ((40-self.current_epoch*10)%41)+5
            #self.SNR_select()
            # training_step defines the train loop. It is independent of forward
            chann, x = batch
            # Chann Formating
            chann = chann.permute(0,3,1,2) 
            # Prepare GridTransformer
            #TODO test with conj True and False
            Y     = self.Get_Y(chann,x,conj=CONJ_ACTIVE,noise_activ=NOISE) 
            # Transform src values to grid tokens 
            src_tokens = self.grid_token(Y).permute(1,0) # [length,batch]
                        
            x_hat = torch.zeros(x.shape)
            for batch_i in range(0,BATCHSIZE):
                outputs = [2]#"<sos>"
                sentence_tensor = src_tokens[:,batch_i].unsqueeze(1)
                for symbol in range(0,47):
                    trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
                    output     = self(sentence_tensor, trg_tensor)
                    best_guess = output.argmax(2)[-1, :].item()
                    outputs.append(best_guess)
                    # Concatenate previous input with predicted best word
                x_hat[batch_i] = torch.Tensor(outputs)
        
            #Get target tokens
            #degrid rx_bits
            x_hat = self.grid_decode(x_hat)
            
            src_abs_factor = torch.max(torch.abs(x),dim=1, keepdim=True)[0]
            # Normalize the data by dividing it with the maximum absolute value
            x = x/src_abs_factor
            
            self.SNR_calc(x_hat,x) 
            
        return 0 
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run=False,accelerator='gpu',callbacks=[TQDMProgressBar(refresh_rate=2)],auto_lr_find=True, max_epochs=NUM_EPOCHS)
                #resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/GridNet/lightning_logs/version_12/checkpoints/epoch=144-step=87000.ckpt')
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
    """
    #name of output log file 
    formating = "Test_(Golden_{}QAM_{})_{}".format(QAM,"GridTransformer",get_time_string())
    tf.SNR_BER_TEST(trainer,formating)
    """

    