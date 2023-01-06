import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(self.head_dim *heads == embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.keys   = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim,embed_size)
        
    def forward(self,values,keys,query,mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shapes[1], keys.shape[1],query.shape[1]
        
        #Split embbeding into self.heads pieces
        values = values.reshape(N,value_len,self.heads,self.head_dim)
        keys = keys.reshape(N,key_len,self.heads,self.head_dim)
        queries = query.reshape(N,query_len,self.heads,self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd -> nhqk",[queries,keys])
        # queries shape: (N,query_len, heads, heads_dim)
        # keys shape : N,key_len,heads,head_dim
        # energy shape: (N,heads,query_len,key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # small value to maks
            
        #softmax
        #dim = 3 normalizing acrros key lenght
        attention = torch.softmax(energy/ (self.embed_size ** (1/2)),dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
           N,query_len,self.heads*self.head_dim 
        )
        # attention shape: (N,heads,query_len, key_len)
        # values shape: (N,value_len,heads, heads_dim)
        # (N,query_len,heads,head_dim)
        out = self.fc_out(out)
        return out
    
class TransfromerBlock(nn.Module):
    def __init__(self, embed_size,heads,dropout,forward_expansion):
        super(TransfromerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1     = nn.LayerNorm(embed_size)
        self.norm2     = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)
        x         = self.dropout (self.norm1(attention+query))
        forward   = self.feed_forward(x)
        out       = self.dropout(self.norm2(forward+x))
        
        return out
    
    
        
        
        
        
        
        