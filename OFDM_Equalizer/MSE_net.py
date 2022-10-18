from tkinter.tix import Tree
import torch
from math import sqrt
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
#from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from Recieved import RX
import GPUtil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

PLOT_REALIZATIONS = True
PLOT_LOSS         = False
TOGGLE            = False

# Multiclass problem
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.f1    = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.f2   = nn.Tanh()
        self.linear3 = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.f1(out)
        out = self.linear2(out)
        out = self.f2(out)
        out = self.linear3(out)
        return out


def Generate_SNR(data,SNR):
    data.AWGN(SNR)
    r_real = torch.tensor(data.Qsym.r.real,device  = torch.device('cuda'),dtype=torch.float64)
    r_imag = torch.tensor(data.Qsym.r.imag,device  = torch.device('cuda'),dtype=torch.float64)
    r      = torch.cat((r_real,r_imag),0)
    del r_real
    del r_imag
    torch.cuda.empty_cache()
    return r

#Data set read
data = RX()
#Data numbers
N = data.sym_no

#ground truth
gt_real = torch.tensor(data.Qsym.GroundTruth.real,device  = torch.device('cuda'),dtype=torch.float64)
gt_imag = torch.tensor(data.Qsym.GroundTruth.imag,device  = torch.device('cuda'),dtype=torch.float64)
gt = torch.cat((gt_real,gt_imag),0)
del gt_real
del gt_imag
torch.cuda.empty_cache()

#MODEL COFING

model  = NeuralNet(input_size=2*N, hidden_size=4*N, num_classes=2*N)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)

#criteria based on MSE
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=.001,eps=.001)
training_data = int(data.total*.8)

# Training
losses = []
BEST_SNR  = 60
WORST_SNR = 5

for SNR in range(BEST_SNR,WORST_SNR,-1):
    r = Generate_SNR(data,SNR)
    loop  = tqdm(range(0,training_data),desc="Progress")
    for i in loop:     
        X  = torch.squeeze(r[:,i],1)  # input
        Y  = torch.squeeze(gt[:,i],1) # groung thruth
                
        # Compute prediction and loss
        pred = model(X.float())
        loss = criterion(pred,Y.float())
        
        losses.append(torch.mean(loss).cpu().detach().numpy())
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Loss/train", loss, SNR)
        
        if(i % 100 == 0):
            loop.set_description(f"SNR [{SNR}]")
            loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
            print(GPUtil.showUtilization())

writer.flush()

plt.plot(losses)
plt.ylabel('loss') #set the label for y axis
plt.xlabel('epochs') #set the label for x-axis
#plt.show()
plt.savefig('loss.png')

torch.save(model.state_dict(),"OFDM_Equalizer.pth")
#torch.onnx.export(model,r[:,0].float(),"MSE_net.onnx", export_params=True,opset_version=10)
