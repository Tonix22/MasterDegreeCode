from numbers import Complex
import numpy as np
import pandas as pd
from math import sqrt
N  = 512    # Delay-Frecuency elements(taps) 1024
M  = 128    # Doppler-Time elements    48

df = pd.read_csv('input_data.csv',header=None)
input = df.to_numpy(copy=False)
data = []
for n in input:
    for m in n:
        data.append(complex(m))
data = np.array(data,dtype=Complex)
data = np.reshape(data,(N,M))
#X = np.fft.ifft(input,axis=1)
X = sqrt(M)/sqrt(N)*np.fft.fft(np.fft.ifft(input,axis=1),axis=0)
np.set_printoptions(suppress=True)
for n in X[0]:
    print (n,file=open('pythonSFFT.txt', 'a'))