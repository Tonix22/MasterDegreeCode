import numpy as np
import matplotlib.pyplot as plt


class QPSK():
    def __init__(self,num_symbols, noise = False, noise_power = 0.05, plot=False):
        
        x_int       = np.random.randint(0,4,num_symbols)
        gray_pos    = [45,315,135,225]
        x_degrees = []
        for n in x_int:
            x_degrees.append(gray_pos[n])
        x_degrees = np.asarray(x_degrees)   
        x_radians   = x_degrees*np.pi/180
        # this produces our QPSK complex symbols
        self.symbols = np.cos(x_radians) + 1j*np.sin(x_radians)

        if(noise == True):
            noise_power = 0.05
            #AWGN with unity power
            n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2)
            #adjust multiplier for "strength" of phase noise
            phase_noise = np.random.randn(len(self.symbols)) * noise_power
            
        if(plot == True):
            #additive gausian noise
            r = self.symbols* np.exp(1j*phase_noise) + n * np.sqrt(noise_power)
            plt.plot(np.real(r), np.imag(r), '.')
            plt.grid(True)
            plt.show()