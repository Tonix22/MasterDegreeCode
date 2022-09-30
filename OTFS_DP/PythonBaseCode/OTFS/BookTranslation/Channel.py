from Constants import *
import numpy as np
from GPP3 import *
from math import cos,pi,sqrt
from random import randint

class Channel():
    def __init__(self,delay_resolution,k_max, GPP_standard = "EVA" , Spectrum=SprectrumType.JAKES):
        
        # assuming Jakes spectrum
        if(Spectrum == SprectrumType.STANDARD):
            Model = GPP_3_models[GPP_standard]
            Model.To_DB_and_Normalize()
            # generate channel coefficients (Rayleigh fading)
            self.g_i = np.sqrt(Model.pdp_linear)@(np.sqrt(2)*(np.random.randn((1,Model.taps))+1j*np.random.randn((1,Model.taps))))
            # generate delay taps (assuming integer delay taps)
            self.l_i=np.around(Model.delay/delay_resolution)
            # Generate Doppler taps (assuming Jakes spectrum)
            self.k_i = (k_max*cos(2*pi*np.random.randn((1,Model.taps))))

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
            self.k_i = k_max-2*k_max*np.random.randn((1,taps))
            
        # Generate discrete delay-time channel coefficients and matrix.

        # Generate delay-time and delay-Doppler channel matrix.
        
        # Generate r by passing the Tx signal through the channel.


        
    