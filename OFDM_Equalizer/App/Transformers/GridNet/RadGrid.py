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


class PolarGridCode():
    def __init__(self, step_radius,step_angle):
        # Define the step size for binning
        self.step_radius = step_radius
        self.step_angle  = step_angle # np.pi/8

        # Define bins for the radius and angle coordinates of the data
        self.binsr = torch.arange(0, 1 + self.step_radius, self.step_radius, dtype=torch.float64)
        self.binsa = torch.arange(-np.pi, np.pi + self.step_angle, self.step_angle, dtype=torch.float64)

        # Create a 2D bin index matrix for encoding
        self.binra = torch.arange(start=0, end=(len(self.binsr) - 1) * len(self.binsa)).view(len(self.binsr) - 1, len(self.binsa))

    def Encode(self, data):
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # Convert from rectangular to polar coordinates
        r, a = data.abs(),data.angle()

        # Create empty tensors for storing the indices of the bins for the radius and angle coordinates of the data
        r_indices = torch.zeros(r.shape, dtype=torch.long)
        a_indices = torch.zeros(a.shape, dtype=torch.long)

        # Loop over the bins for the radius and angle coordinates
        for i in range(len(self.binsr) - 1):
            for j in range(len(self.binsa) - 1):
                # Find the indices of the data that lie within the current bin for the radius and angle coordinates
                r_indices[(self.binsr[i] <= r) & (r < self.binsr[i + 1])] = i
                a_indices[(self.binsa[j] <= a) & (a < self.binsa[j + 1])] = j

        # Encode the data by selecting the corresponding bin indices from the bin index matrix
        encoded = self.binra[r_indices, a_indices]
        return encoded

    def Decode(self, encoded):
        # Create empty tensors for storing the decoded values for the radius and angle coordinates of the data
        r_decoded = torch.zeros(encoded.shape, dtype=torch.float64)
        a_decoded = torch.zeros(encoded.shape, dtype=torch.float64)

        if type(encoded) == np.ndarray:
            encoded = torch.from_numpy(encoded)

        # Loop over the bins for the radius and angle coordinates
        for i in range(len(self.binsr) - 1):
            for j in range(len(self.binsa)-1):
                # Find the indices of the encoded values that correspond to the current bin for the radius coordinate
                indices = (encoded == self.binra[i, j])

                # Fill the decoded values for the radius coordinate
                r_decoded[indices] = self.binsr[i] + self.step_radius / 2

                # Fill the decoded values for the angle coordinate
                a_decoded[indices] = self.binsa[j] + self.step_angle /2

        # Convert back to rectangular coordinates
        data_decoded = torch.polar(r_decoded, a_decoded)
        return data_decoded
    
    def plot_scatter_values(self, ax, vect, title):
        # ----------- Color Map to scatter ---------
        cmap = cmx.jet
        c_norm = colors.Normalize(vmin=0, vmax=len(vect))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
        colorsMap = [scalar_map.to_rgba(i) for i in range(len(vect))]
        r, a = torch.abs(vect), torch.angle(vect)
        
        # Plot the values in polar coordinates
        ax.scatter(a, r, s=10, c=colorsMap, marker='*')
        for i, txt in enumerate(range(len(vect))):
            ax.text(a[i], r[i],i,fontsize=10)
        
        # Set the tick values for the radius and angle coordinates
    
        r_ticks = self.binsr
        a_ticks = (self.binsa+torch.pi)
        ax.set_rticks(r_ticks)
        ax.set_thetagrids(a_ticks * 180 / np.pi)
        
        # Set the plot limits
        ax.set_rlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot title and labels
        ax.set_title(title)
        ax.set_xlabel('Angle')
        ax.set_ylabel('Radius')




# Define the step sizes for binning
step_radius = 0.2
step_angle = np.pi/16

# Create an instance of the PolarGridCode class
coding = PolarGridCode(step_radius, step_angle)

# Generate some random complex data
data = np.random.rand(10) * np.exp(1j * np.random.rand(10) * 2 * np.pi)

# Encode the data using the PolarGridCode class
encoded = coding.Encode(data)

# Decode the encoded data using the PolarGridCode class
decoded = coding.Decode(encoded)

data = torch.from_numpy(data)
# Plot the original data in the polar coordinate system
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
coding.plot_scatter_values(ax, data, "Original Data")

# Plot the decoded data in the polar coordinate system
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
coding.plot_scatter_values(ax, decoded, "Decoded Data")

plt.show()
#plt.savefig('RadGridTest.png')