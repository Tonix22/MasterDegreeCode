import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html

M = 512
NW = 2.5
win, eigvals = windows.dpss(M, NW, 4, return_ratios=True)
fig, ax = plt.subplots(1)
ax.plot(win.T, linewidth=1.)
ax.set(xlim=[0, M-1], ylim=[-0.1, 0.1], xlabel='Samples',
       title='DPSS, M=%d, NW=%0.1f' % (M, NW))
ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
           for ii, ratio in enumerate(eigvals)])
fig.tight_layout()
plt.savefig("Prolate")