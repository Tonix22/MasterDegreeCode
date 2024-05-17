## Models
#Delays are in nanometers
import numpy as np

class GPP3():
    def __init__(self):
        self.delay = None
        self.pdp   = None
        self.pdp_linear = None
        self.taps  = None
        
    def To_DB_and_Normalize(self):
        #transform to db
        self.pdp_linear = 10^(self.pdp/10)
        #normalization
        self.pdp_linear = self.pdp_linear/np.sum(self.pdp_linear)
        self.taps       = self.pdp_linear.size

class EPA(GPP3):
    def __init__(self) -> None:
        self.delay = np.array([0,     30,   70,   90,  110,   190,   410])*10e-9
        self.pdp   = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
        
class EVA(GPP3):
    def __init__(self):
        self.delay = np.array([0,     30,  150,  310,  370,  710, 1090,  1730,  2510])*10e-9
        self.pdp   = np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])

class ETU(GPP3):
    def __init__(self):
        self.delay = np.array([0,      50,  120, 200, 230, 500, 1600, 2300, 5000])*10e-9 
        self.pdp   = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0])

GPP_3_models = {
    "EPA" : EPA(),
    "EVA" : EVA(),
    "ETU" : ETU()
}
