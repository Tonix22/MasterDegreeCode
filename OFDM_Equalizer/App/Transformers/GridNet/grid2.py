import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# Define the batch size and sequence length
batch, seqlen = 4, 48
# Define the step size for binning
step = 1/4

# Define bins for the real and imaginary parts of the data
binsx = torch.arange(-1, 1 + step, step)
binsy = torch.arange(-1, 1 + step, step)


# ----------- Color Map to scatter ---------
cmap       = cmx.jet
c_norm     = colors.Normalize(vmin=0, vmax=48)
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)


def plot_scatter_values(ax,vect,title):
    
    colorsMap     = [scalar_map.to_rgba(i) for i in range(vect.size)]
    real = np.real(vect)
    imag = np.imag(vect)
    
    #fig, ax2 = plt.subplots()
    ax.scatter(real, imag,s=10, c=colorsMap,marker='*')
    for i, txt in enumerate(range(vect.size)):
        ax.text(real[i], imag[i],i,fontsize=10)

    ax.grid(True)
    ax.set_xticks(binsx)
    ax.set_yticks(binsy)
    #plot title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    #x2-y axis line
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_title(title)


# Generate complex valued random data and convert it to the torch complex128 data type
data           = torch.complex(torch.rand((batch, seqlen))*2-1 ,torch.rand((batch, seqlen))*2-1).to(torch.complex128)

# Get the absolute maximum value of the data along the sequence dimension (dim=1)
# Keep the first dimension with keepdim=True for broadcasting purposes
src_abs_factor = torch.max(torch.abs(data),dim=1, keepdim=True)[0]

# Normalize the data by dividing it with the maximum absolute value
data = data/src_abs_factor

# Create a 2D bin index matrix for encoding
binxy = torch.arange(start=4, end=4 + (len(binsx) - 1) * (len(binsy) - 1)).view(len(binsx) - 1, len(binsy) - 1)

# Create empty tensors for storing the indices of the bins for the real and imaginary parts of the data
x_indices = torch.zeros((batch, seqlen), dtype=torch.long)
y_indices = torch.zeros((batch, seqlen), dtype=torch.long)

# Loop over the bins for the real and imaginary parts
for i in range(len(binsx) - 1):
    # Find the indices of the data that lie within the current bin for the real part
    x_indices[(binsx[i] <= data.real) & (data.real < binsx[i + 1])] = i
    # Same for imaginary part
    y_indices[(binsy[i] <= data.imag) & (data.imag < binsy[i + 1])] = i

# Encode the data by selecting the corresponding bin indices from the bin index matrix
encoded = binxy[y_indices, x_indices]


# Create empty tensors for storing the decoded values for the real and imaginary parts of the data
real_decoded = torch.zeros_like(data.real)
imag_decoded = torch.zeros_like(data.imag)

# Loop over the bins for the real and imaginary parts
for i in range(len(binsx) - 1):
    for j in range(len(binsy) - 1):
        # Find the indices of the encoded values that correspond to the current bin
        indices = (encoded == binxy[j, i])
        # Fill the decoded values for the real part
        real_decoded[indices] = binsx[i] + step/2
        # Fill the decoded values for the imaginary part
        imag_decoded[indices] = binsy[j] + step/2

# Create a tensor for the decoded data
data_decoded = torch.complex(real_decoded, imag_decoded)

# Denormalize the decoded data by multiplying it with the original maximum absolute value
#data_decoded = data_decoded * src_abs_factor


fig =  plt.subplots()
ax1  = plt.subplot2grid((1, 2), (0, 0))
ax2  = plt.subplot2grid((1, 2), (0, 1))

plot_scatter_values(ax1,data[0][:10].numpy(),"Original")
plot_scatter_values(ax2,data_decoded[0][:10].numpy(),"Decoded")

plt.show()