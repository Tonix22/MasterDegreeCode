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

data = RX(4,"Unit_Pow")
BER    = []

def LMSE(H,Y,SNR):
    Ps = (np.sum(np.abs(Y)**2))/np.size(Y)
    # Noise power
    Pn = Ps / (10**(SNR/10))
    # inverse
    return np.linalg.inv(H.H@H+np.eye(48)*Pn)@H.H@Y
   
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
        bits   = 0
        data.AWGN(SNR)
        for i in loop:
            #Get realization
            Y = data.Qsym.r[:,i]
            H = np.matrix(data.H[:,:,i])
            txbits = np.squeeze(data.Qsym.bits[:,i],axis=1)
            
            X_hat  = EqType[Select](H,Y,SNR)
            
            rxbits = data.Qsym.Demod(X_hat)
            xor    = np.unpackbits((txbits^rxbits).view('uint8'))
            errors+= xor.sum()
            bits  += 16*48 # 16 QAM times 48 symbols
            #Status bar and monitor  
            if(i % 500 == 0):
                loop.set_description(f"SNR [{SNR}] T=[{Select}]")
                loop.set_postfix(ber=errors/(bits))
                
        BER.append(errors/(bits))
        
    vector_to_pandas("Golden_{}_BER_SNR{}.csv".format(Select,get_time_string()),BER)



    
