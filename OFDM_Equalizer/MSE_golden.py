from Recieved import RX
import numpy as np
from   tqdm import tqdm
import matplotlib.pyplot as plot
from datetime import datetime

BEST_SNR = 50
WORST_SNR = 5

def get_time_string():
    current_time = datetime.now()
    day  = current_time.day
    mon  = current_time.month
    year = current_time.year
    hr   = current_time.time().hour
    mn   = current_time.time().minute
    return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)

def Equalizer(H,Y,SNR):
    return np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))@H.H@Y

data = RX()
BER    = []

for SNR in range(BEST_SNR,WORST_SNR,-5):
    loop   = tqdm(range(0,data.total),desc="Progress")
    errors = 0
    LOS_cnt  = 0
    NLOS_cnt = 0
    data.AWGN(SNR)
    for i in loop:
        #Get realization
        Y = data.Qsym.r[:,i]
        if(i&1):
            H = np.asmatrix(data.LOS[LOS_cnt])
            LOS_cnt+=1
        else:
            H = np.asmatrix(data.NLOS[NLOS_cnt])
            NLOS_cnt+=1
        
        txbits = np.squeeze(data.Qsym.bits[:,i],axis=1)
        X_hat  = Equalizer(H,Y,SNR)
        rxbits = data.Qsym.Demod(X_hat)
        errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
        
        #Status bar and monitor  
        if(i % 500 == 0):
            loop.set_description(f"SNR [{SNR}]")
            loop.set_postfix(ber=errors/((data.bitsframe*data.sym_no)*data.total))
            
            
    BER.append(errors/((data.bitsframe*data.sym_no)*data.total))
    
    
indexValues = np.arange(WORST_SNR,BEST_SNR,5)
BER = np.asarray(BER)
BER = np.flip(BER)
plot.grid(True, which ="both")
plot.semilogy(indexValues,BER)
plot.title('SNR and BER')
# Give x axis label for the semilogy plot
plot.xlabel('SNR')
# Give y axis label for the semilogy plot
plot.ylabel('BER')
plot.savefig('plots/Test_Golden_MMSE_BER_SNR{}.png'.format(get_time_string()))
           
    
