
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



# ----------- Data Setup  ---------
QAM         = 16
realization =  6063 # the argmax given out study in the dataset
#BINS analysis
step = 1/4
#max_index = np.unravel_index(np.argmax(rx.H), rx.H.shape)

# ----------- Data Generation ---------
#Get all samples
rx        = RX(QAM,"Unit_Pow","Complete")
bits      = rx.Qsym.bits[:,realization].squeeze()

# Data multiplied by channel
data      = rx.Qsym.r[:,realization]
data      = rx.Qsym.r[:,realization]/np.max(np.abs(data))

# Data multiplied by hermitian
hermitian = np.array(np.matrix(rx.H[:,:,realization]).H@rx.Qsym.r[:,realization])
hermitian = hermitian/np.max(np.abs(hermitian))
# Ground truth
truth     = rx.Qsym.GroundTruth[:,realization]
truth     = truth/np.max(np.abs(truth))

# ----------- Color Map to scatter ---------
cmap       = cmx.jet
c_norm     = colors.Normalize(vmin=0, vmax=48)
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
colors     = [scalar_map.to_rgba(i) for i in range(48)]

# ----------- Grid Creation ----------------
# Define bins for the real and imaginary parts of the data
binsx = np.arange(-1, 1 + step, step)
binsy = np.arange(-1, 1 + step, step)

# ----------- Generic Plot function ----------
def plot_scatter_values(ax,rx,vect,title):
    real = np.real(vect)
    imag = np.imag(vect)
    
    #fig, ax2 = plt.subplots()
    ax.scatter(real, imag,s=10, c=colors,marker='*')
    for i, txt in enumerate(range(48)):
        ax.text(real[i], imag[i],format(bits[i], '0{}b'.format(rx.bitsframe)),fontsize=10)

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


# Add labels to the x and y axes
fig =  plt.subplots()
ax1  = plt.subplot2grid((1, 3), (0, 0))
ax2  = plt.subplot2grid((1, 3), (0, 1))
ax3  = plt.subplot2grid((1, 3), (0, 2))

plot_scatter_values(ax1,rx,data,"Y*H")
plot_scatter_values(ax2,rx,hermitian,"H^H*Y")
plot_scatter_values(ax3,rx,truth,"Truth")

#plt.savefig('./result.png')
plt.show()