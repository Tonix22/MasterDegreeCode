import sys
import numpy as np
from tqdm import tqdm
import os

main_path = os.path.dirname(os.path.abspath(__file__))+"/../../"
sys.path.insert(0, main_path+"controllers")
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"tools")

from Recieved import RX
from utils import vector_to_pandas ,get_time_string
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP

data = RX(32,"Norm")
BER    = []

def LMSE(H,Y,SNR):
    return np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))@H.H@Y
   
def MSE(H,Y,SNR):
    return np.linalg.inv(H.H@H)@H.H@Y


EqType = {
  "MSE": MSE,
  "LMSE": LMSE,
}

if __name__ == '__main__':
    
    Select = sys.argv[1]
    if Select not in EqType:
        print("INVALID ARG")
        exit()

    for SNR in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,-1*GOLDEN_STEP):
        loop   = tqdm(range(0,data.total),desc="Progress")
        errors = 0
        data.AWGN(SNR)
        for i in loop:
            #Get realization
            Y = data.Qsym.r[:,i]
            H = np.matrix(data.H[:,:,i])
            txbits = np.squeeze(data.Qsym.bits[:,i],axis=1)
            
            X_hat  = EqType[Select](H,Y,SNR)
            
            rxbits = data.Qsym.Demod(X_hat)
            errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
            
            #Status bar and monitor  
            if(i % 500 == 0):
                loop.set_description(f"SNR [{SNR}] T=[{Select}]")
                loop.set_postfix(ber=errors/((data.bitsframe*data.sym_no)*data.total))
                
        BER.append(errors/((data.bitsframe*data.sym_no)*data.total))
        
    vector_to_pandas("Golden_{}_BER_SNR{}.csv".format(Select,get_time_string()),BER)



    
