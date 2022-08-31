#Imports and general parameters
import numpy as np
from math import floor
from math import sqrt
from math import atan
from math import pi
import matplotlib.pyplot as plt

class rayleightFading():
    def __init__(self,fm = 70,fs = 7.68e6):
        self.N=2**20
        self.fm = fm  # doppler frequency
        self.fs   = 7.68e6
        self.Ts   = 1/fs
        self.mean = 0
        self.variance = 1
        self.F  = np.zeros(self.N)
        dopplerRatio = fm/fs
        km = floor(dopplerRatio*self.N)
        # Generate the F piecewise function
        for k in range(0,self.N):
            if k==0:
                self.F[k]=0
            elif k>=1 and k<=(km-1):
                self.F[k]=sqrt(1/(2*sqrt(1-((k-1)/(self.N*dopplerRatio))**2)))
            elif k==km:
                self.F[k]=sqrt(km/2*(pi/2-atan((km-1)/sqrt(2*km-1))))
            elif k>=(km+1) and k<=(self.N-km-1):
                self.F[k] = 0
            elif k==(self.N-km):
                self.F[k]=sqrt(km/2*(pi/2-atan((km-1)/sqrt(2*km-1))))
            else:
                self.F[k]=sqrt(1/(2*sqrt(1-((self.N-(k-1))/(self.N*dopplerRatio))**2)))
    
    def Generate(self,size=2**20):
        g1 = np.random.normal(self.mean, self.variance, self.N)
        g2 = np.random.normal(self.mean, self.variance, self.N)
        g  = g1-1j*g2
        X = g*self.F
        x = np.fft.ifft(X)
        #Generate Rayleight gain model
        self.r = abs(x)
        self.r = self.r/np.mean(self.r)
        idx = min(self.r.size,size)
        return self.r[0:idx]
    
    def Plot(self,val):
        # Plot the Rayleigh envelope
        T = val.size*self.Ts
        t = np.arange(0,T,self.Ts)

        plt.plot(t, 10*np.log10(val))

        plt.xlabel('Time(sec)')
        plt.ylabel('Signal Amplitude (dB)')
        plt.show()

def Test():
    Fade = rayleightFading()
    gen = Fade.Generate(size = 67591)
    Fade.Plot(gen)