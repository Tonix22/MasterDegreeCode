
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
#Header import
main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX


QAM = 16
realization =  6063 # the argmax given out study in the dataset
#max_index = np.unravel_index(np.argmax(rx.H), rx.H.shape)
#Get all samples
rx = RX(QAM,"Unit_Pow","Complete")
bits      = rx.Qsym.bits[:,realization].squeeze()
#Data multiplied by channel
data      = rx.Qsym.r[:,realization]
#data multiplied by hermitian
hermitian = np.array(np.matrix(rx.H[:,:,realization]).H@rx.Qsym.r[:,realization])
hermitian = hermitian/np.max(np.abs(hermitian))
#ground truth
truth     = rx.Qsym.GroundTruth[:,realization]
truth     = truth/np.max(np.abs(truth))

cmap = cmx.jet
c_norm = colors.Normalize(vmin=0, vmax=48)
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
colors = [scalar_map.to_rgba(i) for i in range(48)]


def plot_scatter_values(ax,rx,vect):
    real = np.real(vect)
    imag = np.imag(vect)
    
    #fig, ax2 = plt.subplots()
    ax.scatter(real, imag,s=10, c=colors,marker='*')
    for i, txt in enumerate(range(48)):
        ax.text(real[i], imag[i],format(bits[i], '0{}b'.format(rx.bitsframe)),fontsize=10)

    ax.grid(True)
    #plot title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    #x2-y axis line
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_title("H^H*Y")


# Add labels to the x and y axes
fig =  plt.subplots()
ax1  = plt.subplot2grid((1, 3), (0, 0))
ax2  = plt.subplot2grid((1, 3), (0, 1))
ax3  = plt.subplot2grid((1, 3), (0, 2))

plot_scatter_values(ax1,rx,data)
ax1.set_title("Y*H")

plot_scatter_values(ax2,rx,hermitian)

ax2.set_title("H^H*Y")

plot_scatter_values(ax3,rx,truth)
ax3.set_title("Truth")

plt.show()