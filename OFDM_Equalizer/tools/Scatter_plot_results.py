
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

class ComparePlot():
    def __init__(self,x,x_hat,batch_idx = 0):
        #take only first batch
        x     = x[batch_idx].cpu().numpy()
        x_hat = x_hat[batch_idx].cpu().numpy()
        self.max_idx = 10
        x = x[:self.max_idx]
        x_hat = x_hat[:self.max_idx]
        #BINS analysis
        self.step = 1/4
        # ----------- Grid Creation ----------------
        # Define bins for the real and imaginary parts of the data
        self.binsx = np.arange(-1, 1 + self.step, self.step)
        self.binsy = np.arange(-1, 1 + self.step, self.step)
        
        # ----------- Color Map to scatter ---------
        cmap        = cmx.jet
        c_norm      = colors.Normalize(vmin=0, vmax=self.max_idx)
        scalar_map  = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
        self.colors = [scalar_map.to_rgba(i) for i in range(self.max_idx)]
        
        # Add labels to the x and y axes
        fig =  plt.subplots()
        ax1  = plt.subplot2grid((1, 2), (0, 0))
        ax2  = plt.subplot2grid((1, 2), (0, 1))
        self.plot_scatter_values(ax1,x,"x")
        self.plot_scatter_values(ax2,x_hat,"x_hat")
        plt.savefig('./result.png')
        #plt.show()

    # ----------- Generic Plot function ----------
    def plot_scatter_values(self,ax,vect,title):
        real = np.real(vect)
        imag = np.imag(vect)
        
        #fig, ax2 = plt.subplots()
        ax.scatter(real, imag,s=10, c=self.colors,marker='*')
        for i, txt in enumerate(range(self.max_idx)):
            ax.text(real[i], imag[i],str(i),fontsize=10)

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



