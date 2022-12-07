import numpy as np
import matplotlib.pyplot as plt
from QAM_arrays import QAM_dict
from math import sqrt

class QAM():
    #const_type : Data, Unit_Pow, Norm
    def __init__(self,num_symbols,constelation=4,cont_type= "Data", noise = False, noise_power = 0.05):
        
        #check valid input values
        if(constelation!=4 and constelation!=16 and constelation!=32 and constelation!=64):
            raise ValueError("Enter valid constelation size: 4,6,16,32")
        if(cont_type!="Data" and cont_type!="Unit_Pow" and cont_type!="Norm"):
            raise ValueError("Enter valid constelation type: Data,Unit_Pow,Norm")
        
        #constelation size
        self.constelation = constelation
        #get QAM array
        self.QAM_N_arr = QAM_dict[cont_type][constelation]
        #fixed random seed
        np.random.seed(42)
        #Generate N bits
        self.bits   = np.random.randint(0,self.constelation,num_symbols)
        #This array will collect true conteslation bits complex plane points
        self.GroundTruth = np.zeros(self.bits.size,dtype=complex)
        
        for n in range(0,self.bits.size):
            self.GroundTruth[n] = self.QAM_N_arr[self.bits[n]]
        
        #noise config or not
        if(noise == True):
            phase_noise = 0
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
    
    def DemodUpgrade(self,vect):
        bits  = []
        IQ_defect  = np.asarray(vect)
        for IQ in np.nditer(IQ_defect):
            distances = np.zeros(self.QAM_N_arr.size)
            for n in range(0,self.QAM_N_arr.size):
                Block_angle = self.QAM_N_arr[n]
                distances[n]= sqrt((IQ.real-Block_angle.real)**2+(IQ.imag-Block_angle.imag)**2)
            bits.append(np.argmin(distances))

        return np.asarray(bits)   
            
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
    