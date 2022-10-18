import numpy as np
import matplotlib.pyplot as plt


class QPSK():
    def __init__(self,num_symbols, noise = False, noise_power = 0.05):
        np.random.seed(42)#fixed bits
        x_int       = np.random.randint(0,4,num_symbols)
        phase_noise = 0
        gray_pos    = [45,315,135,225]
        x_degrees = []
        for n in x_int:
            x_degrees.append(gray_pos[n])
        x_degrees = np.asarray(x_degrees)   
        x_radians   = x_degrees*np.pi/180
        # this produces our QPSK complex symbols
        self.GroundTruth = np.cos(x_radians) + 1j*np.sin(x_radians)

        if(noise == True):
            noise_power = 0.05
            #AWGN with unity power
            n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2)
            #adjust multiplier for "strength" of phase noise
            phase_noise = np.random.randn(len(self.GroundTruth)) * noise_power
            #additive gausian noise
            self.r = self.GroundTruth * np.exp(1j*phase_noise) + n * np.sqrt(noise_power)
        else:
            self.r = self.GroundTruth
            
        #Adjust shape
        self.GroundTruth = np.expand_dims(self.GroundTruth,axis=1)
        self.r = np.expand_dims(self.r,axis=1)
          
    def QPSK_Plot(self,vect):
        
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.plot(np.real(vect), np.imag(vect), '.')
        ax.grid(True)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        #plt.show()
        plt.savefig('QPSK.png')
    