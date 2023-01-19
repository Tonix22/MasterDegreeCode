import os
import os.path
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import norm,rayleigh
import numpy as np

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX

BATCHSIZE = 10

def plot_histogram(train_dataloader):
    bins = 1000
    # Initialize an array to store the histogram
    histogram = torch.zeros(bins)
    data_tensor = torch.empty(0, 48,48,2)
    # Iterate over the Dataloader
    for i, (chann,x) in enumerate(train_dataloader):
        chann = chann.cpu()
        #Compute the histogram
        histogram += torch.histc(chann, bins=bins)
        data_tensor = torch.cat((data_tensor, chann), dim=0)
    
    #Data stats
    max = int(data_tensor.max())
    min = int(data_tensor.min())
    range = max-min
    print("Max: ", torch.max(data_tensor))
    print("Min: ", torch.min(data_tensor))
    print("Mean: ", torch.mean(data_tensor))
    print("Median: ", torch.median(data_tensor))
    print("Standard deviation: ", torch.std(data_tensor))
    
    #fiting a model
    #mu, std = norm.fit(data_tensor)
    # Generate x values for the PDF
    #pdf_x = np.linspace(norm.ppf(0.01, loc=mu, scale=std), norm.ppf(0.99, loc=mu, scale=std), 100)
    # Calculate the PDF
    #pdf_y = norm.pdf(pdf_x, loc=mu, scale=std)
    
    # Create a plot
    #plt.plot(pdf_x, pdf_y, '-', lw=2, label='pdf')
    
    x_range = np.arange(min,max,range/bins)
    #plt.bar(np.arange(300,700), histogram[300:700])
    plt.bar(x_range[350:650], histogram[350:650])
    
    #plt.yscale("log")
    plt.savefig('ValData_hist.png')


#Generate Dataset

data    = RX(4,"Unit_Pow")
dataset = data
# Define the split ratios (training, validation, testing)
train_ratio = 0.6
val_ratio   = 0.2
test_ratio  = 0.2
# Calculate the number of samples in each set
train_size = int(train_ratio * len(dataset))
val_size   = int(val_ratio * len(dataset))
test_size  = len(dataset) - train_size - val_size
# Split the dataset
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCHSIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCHSIZE, shuffle=True)


plot_histogram(val_loader)
"""
# Create a tensor to store the data
data_tensor = torch.empty(0, 48,48,2)
for i, (chann,x) in enumerate(val_loader):
    data_tensor = torch.cat((data_tensor, chann.cpu()), dim=0)
"""