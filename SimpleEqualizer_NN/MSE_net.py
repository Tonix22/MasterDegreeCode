from tkinter.tix import Tree
import torch
from math import sqrt
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd 
from torch.autograd import Variable
#from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from config import *


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
N = 16

#READ PANDNDAS DATA SET
df = pd.read_csv('Realizations.csv', index_col=0)
##GET X_IN AND R_OUT

columns =[]
for i in range (0,N):
    columns.append('X'+str(i))

X_in = df.loc[:,columns].to_numpy()
columns.clear()

for i in range (0,2*N-1):
    columns.append('R'+str(i))

R_out= df.loc[:,columns].to_numpy()

##CREATE MODEL

model  = NeuralNet(input_size=N, hidden_size=N, num_classes=2*N-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)

X_in  = torch.tensor(X_in,device  =torch.device('cuda'),dtype=torch.float64)
R_out = torch.tensor(R_out,device =torch.device('cuda'),dtype=torch.float64)
Y_perfect = torch.zeros(R_out.size(),device =torch.device('cuda'),dtype=torch.float64)

#criteria based on MSE
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=.001,eps=.00001)

training_data = (int)(len(df)*.01)

# Training

losses = []

for epoch in range(0,20):
    print("epoch: %d" % epoch)
    for i in range(0,training_data):     
        X  = X_in[i]# entrada
        Y  = R_out[i]# que es lo que espero
        
        #plt.subplot(2, 1, 1)
        #plt.stem(X.cpu().detach().numpy(), markerfmt=' ')
        #plt.title("Channel imp resp with gaussian noise")
        #
        #plt.subplot(2, 1, 2)
        #plt.stem(Y.cpu().detach().numpy(), markerfmt=' ')
        #plt.title("Equalized resp with pseudoinv")
        #
        #
        #plt.show()
        
        # Compute prediction and loss
        pred = model(X.float()) # paso la entrada a la red nuronal
        
        if(epoch > 10 and TOGGLE):
            Y = torch.flip(Y,[0])
        if(epoch%2 == 0 and TOGGLE):
            Y_perfect[torch.argmax(Y)] = 1
            loss = criterion(pred, Y_perfect.float()) #error
            Y_perfect[torch.argmax(Y)] = 0
        else:
            loss = criterion(pred, Y.float()) #error
        
        losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        
        optimizer.step()

if PLOT_LOSS == True:
    plt.plot(losses)
    plt.ylabel('loss') #set the label for y axis
    plt.xlabel('epochs') #set the label for x-axis
    plt.show()

if(EXPORT_ONNX):
    torch.onnx.export(model,X_in[0].float(),"MSE_net.onnx", export_params=True,opset_version=10)


if PLOT_REALIZATIONS == True:
    vanished = 0
    testing_err = []
    for n in range(training_data,len(df)):

        realization_predicted = (model(X_in[n].float())).cpu().detach().numpy()
        if(np.all(realization_predicted==0)):
            vanished = vanished+1
        else:
            Diff_vect = abs(R_out[n].cpu().detach().numpy() - realization_predicted)
            testing_err.append(Diff_vect)
            
            R = R_out[n].cpu().detach().numpy()
            plt_y = np.linspace(1, len(R), num=len(R))
            max_point_x = np.argmax(R)
            max_point_y = R[max_point_x]
            plt.subplot(1, 2, 1)
            plt.stem(plt_y,R, markerfmt=' ')
            plt.ylabel('Amplitud') #set the label for y axis
            plt.xlabel('Sample') #set the label for x-axis
            plt.text(max_point_x,max_point_y,"({:.4f})".format(max_point_y))
            plt.title("Original")
            
            max_point_x = np.argmax(realization_predicted)
            max_point_y = realization_predicted[max_point_x]
            plt_y = np.linspace(1, len(realization_predicted), num=len(realization_predicted))
            plt.subplot(1, 2, 2)
            plt.stem(plt_y,realization_predicted, markerfmt=' ')
            plt.ylabel('Amplitud') #set the label for y axis
            plt.xlabel('Sample') #set the label for x-axis
            plt.text(max_point_x,max_point_y,"({:.4f})".format(max_point_y))
            plt.title("Predicted")            
            plt.show()
        
"""
###
testing_err = np.asarray(testing_err)
print("average error: %d" % np.mean(testing_err))
print("average mean square error: %f " % (sqrt(np.mean(np.square(testing_err)))))

testing_samples = len(df)-training_data
accurate_percentage = (testing_samples-vanished)/testing_samples

print("Testing samples: %d" %(testing_samples))
print("Predicted samples: %d" %(testing_samples-vanished))
print("Predicted in percent : %f" % (accurate_percentage))



y_pred = model(Variable(torch.from_numpy(np.array(df.iloc[EPOCHS:len(df)-1,5:9])).type(torch.FloatTensor)))
y_pred = y_pred.data.numpy()
Y_test = np.array(df.iloc[EPOCHS:len(df)-1,5:9])

Diff_vect = abs(Y_test - y_pred)
RMS       = sqrt(np.mean(np.square(Diff_vect)))
print("Accuracy")
print(1-RMS)
"""
