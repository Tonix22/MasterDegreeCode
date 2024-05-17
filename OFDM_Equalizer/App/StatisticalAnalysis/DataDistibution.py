import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

BATCHSIZE  = 100
QAM = 16
main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"tools")
from Recieved import RX,Rx_loader


loader =  Rx_loader(BATCHSIZE,QAM,"Complete")
loader.data.AWGN(35)

# Assuming 'loader' is your data object and you've loaded your data correctly
Qsym_r = loader.data.Qsym.r.reshape(-1)  # Flatten the array to 1D
GroundTruth = loader.data.Qsym.GroundTruth.reshape(-1)  # Flatten the array to 1D

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(Qsym_r, bins=100, alpha=0.75, label='Qsym.r')
plt.title('Histogram of Qsym.r')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(GroundTruth, bins=50, alpha=0.75, color='r', label='GroundTruth')
plt.title('Histogram of GroundTruth')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()