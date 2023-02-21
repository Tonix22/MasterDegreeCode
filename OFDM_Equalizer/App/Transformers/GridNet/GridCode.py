import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import sys

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX

class GridCode():
    def __init__(self,step):
        # Define the step size for binning
        self.step          = step
        self.indices_shape = None
        self.decoded_shape = None

        # Define bins for the real and imaginary parts of the data
        self.binsx = torch.arange(-.85, .85 + self.step, self.step,dtype=torch.float64)
        self.binsy = torch.arange(-.85, .85 + self.step, self.step,dtype=torch.float64)
        # Create a 2D bin index matrix for encoding
        self.binxy = torch.arange(start=4, end=4 + (len(self.binsx) - 1) * (len(self.binsy) - 1)).view(len(self.binsx) - 1, len(self.binsy) - 1)

    def Encode(self,data):
        
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
        # Create empty tensors for storing the indices of the bins for the real and imaginary parts of the data
        if data.shape != self.indices_shape:
            self.x_indices  = torch.zeros(data.shape, dtype=torch.long)
            self.y_indices  = torch.zeros(data.shape, dtype=torch.long)
            self.indices_shape = data.shape
        else:
            self.x_indices.zero_()
            self.y_indices.zero_()
            
        # Loop over the bins for the real and imaginary parts
        for i in range(len(self.binsx) - 1):
            # Find the indices of the data that lie within the current bin for the real part
            self.x_indices[(self.binsx[i] <= data.real) & (data.real < self.binsx[i + 1])] = i
            # Same for imaginary part
            self.y_indices[(self.binsy[i] <= data.imag) & (data.imag < self.binsy[i + 1])] = i

        # Encode the data by selecting the corresponding bin indices from the bin index matrix
        encoded = self.binxy[self.y_indices, self.x_indices]
        return encoded

    def Decode(self,encoded):
        # Create empty tensors for storing the decoded values for the real and imaginary parts of the data
        
        if encoded.shape!= self.decoded_shape:
            self.real_decoded  = torch.zeros((encoded.shape),dtype=torch.float64)
            self.imag_decoded  = torch.zeros((encoded.shape),dtype=torch.float64)
            self.decoded_shape = encoded.shape
        else:
            self.real_decoded.zero_()
            self.imag_decoded.zero_()
        
        if type(encoded) == np.ndarray:
            encoded = torch.from_numpy(encoded)
        # Loop over the bins for the real and imaginary parts
        for i in range(len(self.binsx) - 1):
            for j in range(len(self.binsy) - 1):
                # Find the indices of the encoded values that corresponself.d to the current bin
                indices = (encoded == self.binxy[j, i])
                # Fill the decoded values for the real part
                self.real_decoded[indices] = self.binsx[i] + self.step/2
                # Fill the decoded values for the imaginary part
                self.imag_decoded[indices] = self.binsy[j] + self.step/2

        # Create a tensor for the decoded data
        data_decoded = torch.complex(self.real_decoded, self.imag_decoded)        
        return data_decoded
    
    def plot_scatter_values(self,ax,vect,title):
        # ----------- Color Map to scatter ---------
        cmap       = cmx.jet
        c_norm     = colors.Normalize(vmin=0, vmax=vect.size)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
        colorsMap  = [scalar_map.to_rgba(i) for i in range(vect.size)]
        real = np.real(vect)
        imag = np.imag(vect)
        
        #fig, ax2 = plt.subplots()
        ax.scatter(real, imag,s=10, c=colorsMap,marker='*')
        for i, txt in enumerate(range(vect.size)):
            ax.text(real[i], imag[i],i,fontsize=10)

        ax.grid(True)
        ax.set_xticks(self.binsx)
        ax.set_yticks(self.binsy)
        #plot title
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        #x2-y axis line
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_title(title)

"""
coding = GridCode(1/7)

fig =  plt.subplots()
ax1  = plt.subplot2grid((1, 2), (0, 0))
ax2  = plt.subplot2grid((1, 2), (0, 1))
# Ground truth
rx        = RX(16,"Unit_Pow","Complete")
rx.Qsym.GroundTruth = rx.Qsym.GroundTruth/np.max(np.abs(rx.Qsym.GroundTruth))
rx.Qsym.GroundTruth = coding.Decode(coding.Encode(rx.Qsym.GroundTruth)).numpy()

values    = rx.H[:,:,6063] @ rx.Qsym.GroundTruth[:,10]
values    = np.array(np.matrix(rx.H[:,:,10]).H@values)
values    = values/np.max(np.abs(values))

coding.plot_scatter_values(ax1,rx.Qsym.GroundTruth[:10,10],"Truth")
coding.plot_scatter_values(ax2,values[:10],"Centered")
"""

#rx.Qsym.QAM_norm_arr = coding.Decode(coding.Encode(rx.Qsym.QAM_norm_arr )).numpy()
#coding.plot_scatter_values(ax1,rx.Qsym.QAM_norm_arr,"QAM")

# Generate complex valued random data and convert it to the torch complex128 data type
#data           = torch.complex(torch.rand((4, 48))*2-1 ,torch.rand((4, 48))*2-1).to(torch.complex128)
#src_abs_factor = torch.max(torch.abs(data),dim=1, keepdim=True)[0]
# Normalize the data by dividing it with the maximum absolute value
#data = data/src_abs_factor
#data_decoded = coding.Decode(coding.Encode(data))
#plot_scatter_values(ax1,data[0][:10].numpy(),"Original",coding.binsx,coding.binsy)
#plot_scatter_values(ax2,data_decoded[0][:10].numpy(),"Decoded",coding.binsx,coding.binsy)

#plt.show()