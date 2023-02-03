import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the batch size and sequence length
batch, seqlen = 4, 48
# Define the step size for binning
step = 1/16

# Generate complex valued random data and convert it to the torch complex128 data type
data           = torch.complex(torch.rand((batch, seqlen))*2-1 ,torch.rand((batch, seqlen))*2-1).to(torch.complex128)

# Get the absolute maximum value of the data along the sequence dimension (dim=1)
# Keep the first dimension with keepdim=True for broadcasting purposes
src_abs_factor = torch.max(torch.abs(data),dim=1, keepdim=True)[0]

# Normalize the data by dividing it with the maximum absolute value
data = data/src_abs_factor

# Define bins for the real and imaginary parts of the data
binsx = torch.arange(-1, 1 + step, step)
binsy = torch.arange(-1, 1 + step, step)
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

"""
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
data_decoded = data_decoded * src_abs_factor
"""


# Extract the real and imaginary parts of the complex numbers
real = np.real(data.numpy())
imag = np.imag(data.numpy())

# Plot the real and imaginary parts using a scatter plot
#plt.scatter(real, imag)

# Add labels to the x and y axes
fig, ax = plt.subplots()
# Get the bins to plot in the graph
ax.grid(True)
ax.set_xticks(binsx.numpy())
ax.set_yticks(binsy.numpy())
#data plot
ax.scatter(real, imag,s=10)
#plot title
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
#x-y axis line
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
# Show the plot
plt.show()