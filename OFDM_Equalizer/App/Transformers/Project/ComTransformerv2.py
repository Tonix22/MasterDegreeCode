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
from Recieved import RX,Rx_loader


#Hyperparameters
BATCHSIZE  = 10
#from 35 SNR to 5 takes 30 EPOCHS, so calculate epochs caount in this
NUM_EPOCHS = 200

GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2

# Model hyperparameters
src_vocab_size = 16
trg_vocab_size = 16
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 48
forward_expansion = 4
src_pad_idx       = 1



class Transformer(pl.LightningModule,Rx_loader):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len
    ):
        pl.LightningModule.__init__(self)
        Rx_loader.__init__(self,BATCHSIZE)#Dataloader parent class
        
        #Select device
        self.constelation = src_vocab_size  #Values Range
        self.bitsframe    = int(math.log2(self.constelation))#Bits to Send 
        #loss function
        self.loss_f       = torch.nn.CrossEntropyLoss()
        
        #Embedding section
        self.src_word_embedding     = nn.Embedding(src_vocab_size, 2)
        self.layer_norm_src         = nn.LayerNorm([48,2])
        self.fc_expand_IQ           = nn.Linear(2, embedding_size)
        
        self.src_position_embedding = nn.Embedding(max_len,        embedding_size)
        self.trg_word_embedding     = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len,        embedding_size)

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out      = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout     = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        
        #Communications stuff
        self.SNR_db   = 35
        self.errors   = 0
        self.bits_num = 0
        

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg,H,snr):
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
        IQ_encond = self.src_word_embedding(src)
        IQ_encond = IQ_encond.permute(1,0,2)
        #pass throught the noise
        Y         = self.y_awgn(H,IQ_encond,snr)
        Y         = self.layer_norm_src(Y).permute(1,0,2)
        IQ_expand = self.fc_expand_IQ(Y)

        embed_src = self.dropout(
            (IQ_expand + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

    def y_awgn(self,H,x,snr):
        real = 0
        imag = 1
        Yreal = torch.einsum('bjk,bk->bj', H[:, real, :, :], x[:, :, real]) - \
                torch.einsum('bjk,bk->bj', H[:, imag, :, :], x[:, :, imag])
                
        Yimag = torch.einsum('bjk,bk->bj', H[:, real, :, :], x[:, :, imag]) + \
                torch.einsum('bjk,bk->bj', H[:, imag, :, :], x[:, :, real])
        
        distance = torch.pow(Yreal,2) + torch.pow(Yimag,2)
        
        # Signal Power
        Ps = torch.mean(distance)
        # Noise power
        Pn = Ps / (10**(snr/10))
        # Generate noise
        noise_real = torch.sqrt(Pn/2)* torch.randn(x[:,:,real].shape,requires_grad=True).to(self.device)
        noise_imag = torch.sqrt(Pn/2)* torch.randn(x[:,:,imag].shape,requires_grad=True).to(self.device)
        # multiply tensors
        #return torch.stack([Yreal + noise_real,Yimag + noise_imag],dim=2)
        return torch.stack([Yreal ,Yimag],dim=2)
          
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(),lr=0.0005,weight_decay=1e-5,eps=.005)
    
    def training_step(self, batch, batch_idx):
        #(30-5*(n+1))%30+20
        snr = (30-self.current_epoch*5)%31 +5
        chann, x = batch
        #chann preparation
        chann  = chann.permute(0,3,1,2) # Batch, channel, height, weight 
        chann.requires_grad = True
        #input data preparation
        inp_data = x.permute(1,0) # [length,batch]

        #copy the input data as we want to recover the signal
        target = inp_data.clone()
                
        # Forward prop
        # target[:-1, :] remove the last row of the target tensor 
        # and all columns of the tensor
        # Last row is a padding (1) or is a eos (3)
        output = self(inp_data,target[:-1, :],chann,snr)
        output = output.reshape(-1, output.shape[2]) # then we merge first two dim
        
        #target[1:] will select all rows of the target tensor except for the first one
        #so we remove the start of sentence
        target = target[1:].reshape(-1) # merge (target_seq_lenght-eos-1,batch_size) into single dim
        
        loss  = self.loss_f(output,target)
        
        self.log("train_loss", loss) #tensorboard logs
        self.log('SNR', snr, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        snr = (30-self.current_epoch*5)%31 +5
        chann, x = batch
        #chann preparation
        chann  = chann.permute(0,3,1,2) # Batch, channel, height, weight 
        chann.requires_grad = True
        #input data preparation
        inp_data = x.permute(1,0) # [length,batch]

        #copy the input data as we want to recover the signal
        target = inp_data.clone()
                
        # Forward prop
        # target[:-1, :] remove the last row of the target tensor 
        # and all columns of the tensor
        # Last row is a padding (1) or is a eos (3)
        output = self(inp_data,target[:-1, :],chann,snr)
        output = output.reshape(-1, output.shape[2]) # then we merge first two dim
        
        #target[1:] will select all rows of the target tensor except for the first one
        #so we remove the start of sentence
        target = target[1:].reshape(-1) # merge (target_seq_lenght-eos-1,batch_size) into single dim
        
        loss  = self.loss_f(output,target)
                
        self.log("train_loss", loss) #tensorboard log
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss) #tensorboard logs
        return {'val_loss':avg_loss}
    
    def predict_step(self, batch, batch_idx):
        chann, x = batch
        chann    = chann.permute(0,3,1,2)
        SNR      = self.SNR_db
        inp_data = x.permute(1,0) # [length,batch]
        
        rx_bits = torch.zeros(x.shape)
        for batch_i in range(0,BATCHSIZE):
            outputs = [2]#"<sos>"
            sentence_tensor = inp_data[:,batch_i].unsqueeze(1)
            H = chann[batch_i].unsqueeze(0)
            for symbol in range(0,47):
                trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
                output = self(sentence_tensor, trg_tensor,H,self.SNR_db)
                best_guess = output.argmax(2)[-1, :].item()
                outputs.append(best_guess)
                # Concatenate previous input with predicted best word
            rx_bits[batch_i] = torch.Tensor(outputs)
        
        tx_bits   = np.uint8(x.cpu().detach().numpy())
        rx_bits   = np.uint8(rx_bits.cpu().detach().numpy())
        xor       = np.unpackbits((tx_bits^rx_bits))
        self.bits_num += len(xor)
        self.errors  += xor.sum()
        BER           = self.errors/self.bits_num
        #self.log('SNR', SNR, on_step=True, prog_bar=True, logger=True)
        #self.log('BER', BER, on_step=True, prog_bar=True, logger=True)
        
        return BER
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader

if __name__ == '__main__':
    
    trainer = Trainer(accelerator='cuda',callbacks=[TQDMProgressBar(refresh_rate=10)],auto_lr_find=True, max_epochs=NUM_EPOCHS,
                      resume_from_checkpoint='/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/Project/lightning_logs/version_8/checkpoints/epoch=199-step=240000.ckpt')
    tf = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len)
    
    trainer.fit(tf)
    trainer.predict(tf)
    """
    for n in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP*-1):
        tf.error  = 0
        tf.SNR_db = n
        trainer.predict(tf)
    """