import sys
import numpy as np
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"tools")

from Recieved import RX
from utils import vector_to_pandas ,get_time_string
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP

data = RX(16,"Unit_Pow")

# Calculate the magnitudes of the complex numbers
#magnitudes = np.sqrt(data.H.real**2 + data.H.imag**2)
# Normalize the complex numbers by dividing each component by the magnitude
#data.H = data.H / magnitudes


max_val = 3.5889728+1j*0.11739896
min_val = -3.4096806-1j*0.37191758
data.H = (data.H - min_val) / (max_val - min_val)

max_val_inv = 3094.211-1j*575.58716
min_val_inv = -3075.4216+1j*294.03375

if __name__ == '__main__':
    max = []
    min = []
    loop   = tqdm(range(0,data.total),desc="Progress")
    for i in loop:
        #Get realization
        #Y = data.Qsym.r[:,i]
        H = np.matrix(data.H[:,:,i])
        H_inv = np.linalg.inv(H)
        H_inv = (H_inv - min_val_inv) / (max_val_inv - min_val_inv)
        max_local = np.max(H_inv)
        min_local = np.min(H_inv)
        max.append(max_local)
        min.append(min_local)
        
        #Status bar and monitor  
        #if(i % 500 == 0):
            #loop.set_description(f"max [{max_local} min [{min_local}]")
                
    max = np.array(max) 
    min = np.array(min)
    max_global = np.max(max)
    min_global = np.min(min)
    print(max_global)
    print(min_global)



    
