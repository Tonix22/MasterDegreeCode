import numpy as np
import matplotlib.pyplot as plt


class QPSK():
    def __init__(self,num_symbols, noise = False, noise_power = 0.05):
        np.random.seed(42)#fixed bits
        self.bits       = np.random.randint(0,4,num_symbols)
        phase_noise = 0
        gray_pos    = [225,135,315,45]
        x_degrees   = []
        for n in self.bits:
            x_degrees.append(gray_pos[n])
        x_degrees = np.asarray(x_degrees)   
        x_radians = x_degrees*np.pi/180
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
            self.r = np.copy(self.GroundTruth)
            
        #Adjust shape
        self.GroundTruth = np.expand_dims(self.GroundTruth,axis=1)
        self.r = np.expand_dims(self.r,axis=1)
    
    
    def Demod(self,vect):
        angle = None
        bits  = []
        for n in vect:
            angle = np.angle(n,deg=True)
            #First quadrant
            if(angle >=0 and angle <= 90):
                bits.append(3)#11
            #Second quadrant
            elif(angle > 90 and angle <=180):
                bits.append(1)#01
            #Third Quadrant
            elif(angle >= -180 and angle < -90):
                bits.append(0)#00
            #Fourth quadrant
            elif(angle > -90 and angle < 0):
                bits.append(2)#10
            else:
                print("error")
                print(angle)
                
        return np.asarray(bits)      
        
    
        
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
    