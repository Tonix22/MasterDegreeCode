
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
#Header import
main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"controllers")
from Recieved import RX

#Get all samples
rx = RX()
#Get one NLOS sample
NLOS = abs(rx.H[:,:,0])
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(20*np.log(NLOS))
plt.title( "LOS Channel" )
plt.show()
