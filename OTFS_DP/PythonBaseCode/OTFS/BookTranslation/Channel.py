from Constants import *
import numpy as np
from GPP3 import *
from math import cos,pi,sqrt
from random import randint

#delay_resolution
#k_max

class Channel():
    def __init__(self,Transmiter, GPP_standard = "EVA" , Spectrum=SprectrumType.JAKES):
        
        self.TX   = Transmiter
        self.taps = None
        self.gs   = None # Tapped Channel
        self.G    = None # Time channel matrix
        self.H_tilda   = None    # Delay-time channel matrix
        self.H    = None # Delay-Doppler
        self.grid_size = self.Tx.N*self.Tx.M
        
        # assuming Jakes spectrum
        if(Spectrum == SprectrumType.STANDARD):
            Model = GPP_3_models[GPP_standard]
            Model.To_DB_and_Normalize()
            # generate channel coefficients (Rayleigh fading)
            self.g_i = np.sqrt(Model.pdp_linear)@(np.sqrt(2)*(np.random.randn((1,Model.taps))+1j*np.random.randn((1,Model.taps))))
            # generate delay taps (assuming integer delay taps)
            self.l_i=np.around(Model.delay/self.TX.delay_resolution)
            # Generate Doppler taps (assuming Jakes spectrum)
            self.k_i = (self.TX.k_max*cos(2*pi*np.random.randn((1,Model.taps))))
            
            self.taps = Model.taps

        # assuming uniform spectrum
        if(Spectrum == SprectrumType.SYNTHETIC):
            # number of propagation paths
            taps = 6
            #maximum normalized delay and Doppler spred
            l_max = 4
            k_max = 4
            #generate normalized delay and Doppler spread
            self.g_i = sqrt(1/taps)@(np.sqrt(2)*(np.random.randn((1,Model.taps))+1j*np.random.randn((1,Model.taps))))
            #generate delay taps uniformely from [0,l_max]
            self.l_i = np.random.randint(0,high = l_max, size = (1,taps))
            self.l_i = self.l_i - np.min(self.l_i)
            #generate Doppler taps (assuming uniform spectrum) [-k_max,k_max])
            self.k_i  = k_max-2*k_max*np.random.randn((1,taps))
            self.taps = taps
            
        self.delay_spread = max(self.l_i)
        
        
    def Gen_Tapped_delay_line(self):
        # Generate discrete delay-time channel coefficients and matrix.
        #S[q] tap shifting
        z = np.exp(1j*2*pi/(self.grid_size))
        
        #Generate discrete-time baseband channel in TDL form (Eq. (2.22))
        #DISCRETE TIME
        self.gs = np.zeros((self.delay_spread+1,self.grid_size))
        for q in range (0,self.grid_size):
            for i in range(0,self.taps):
                shift = z**(self.k_i[i]*(q-self.l_i[i]))
                self.gs[self.l_i,q]= self.gs[self.l_i,q]+self.g_i[i]*(shift)
        
    def Gen_discrete_time(self):
        self.G = np.zeros(self.grid_size,self.grid_size)
        for q in range(0,self.grid_size):
            for ell in range(0,self.delay_spread):
                if q >ell:
                    self.G[q,q-ell] = self.gs[ell,q]
        
        

    def Gen_delay_time(self):
        self.H_tilda=P*G*P.T
        
        # Generate r by passing the Tx signal through the channel.


        
    