import numpy as np
import sys
from math import sqrt
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
 
# adding Folder_2 to the system path
sys.path.insert(0, '/home/tonix/HardDisk/Documents/Maestria/Tesis/Codigo_OTFS/PythonLab/QAM')
sys.path.insert(0,'/home/tonix/HardDisk/Documents/Maestria/Tesis/Codigo_OTFS/PythonLab/Channels')
from QPSK import QPSK
from RayleightFadeYungs import rayleightFading


Fs = 7.68e6 # Frecuencia de muestreo-Sample Rate
Ts = 1/Fs   # Periodo de muestreo
N  = 512    # Delay-Frecuency elements(taps) 1024
M  = 128    # Doppler-Time elements    48
CP = 16     # Cyclic prefix

# Maximum channel size
ChanSizeDelay       = 20 #Estimated maximum channel delay size (actual value is 2x+1)(conventional OTFS method)
ChanSizeDoppler     = 5  #Estimated maximum channel doppler size (actual value is 2x+1)(conventional OTFS method)

# Spacing
DelayDataSpacing    = 3  #spacing tabs 3
DopplerDataSpacing  = 3

#pilot margin
#size of pilot
pilotmargingdelay   = ChanSizeDelay     #Margen eje delay ventana-piloto
pilotmargingdoppler = ChanSizeDoppler*2 #Margen eje doppler ventana-piloto

ChanModDelayEVA=7

#signal integrity
SNR     = 60
ppower  = 10000 #pilot power
SNRVECT = np.arange(0,30,5) # SNR iterator

frames   = 1000   #Frames 40
Fdoppler = 444.44 # 70-300 Hz

#Tamano Trama OTFS
x = np.zeros((N,M),dtype=complex) 

# SYMBOLS timing
# Delay-Frecuency + Cyclic Prefix
Tsymbol=(N+CP)*Ts  #Tiempo de Simbolo 
Fsymbol=1/Tsymbol  #Simbolos por segundo

# Generacion Indices y Ventana Piloto
# Posicion fila/columna de los datos--espaciado
NVector= np.arange(3,N-3,DelayDataSpacing)
MVector= np.arange(1,M,  DopplerDataSpacing)

# Creacion de indices para posicion de datos
# Arreglo para par de coordenadas 
dpos = np.zeros((NVector.size*MVector.size,2))
k    = 0  #Numero de posiciones de datos-1

for i in NVector:
    for j in MVector:
        dpos[k] = [i,j]
        k=k+1

dataPositions    = dpos #Coordenadas-indices para pos de datos
vecdataPositions = dataPositions[:,0]+(dataPositions[:,1]-1)*N
vecdataPositions = vecdataPositions.astype(int)
print("vecdataPositions: "+str(vecdataPositions.size))

#generate random bits for QPSK
txbits = QPSK(vecdataPositions.size)
i = 0
#Hacemos flattening para la insersion de datos. 
x = x.flatten()

for n in vecdataPositions:
    x[n] = txbits.symbols[i] # Acomodo en grid x
    i=i+1
    
#***********************************************
#** SEÑAL OTFS **
x = x.reshape((N,M))
ax = sns.heatmap(data = abs(x))
plt.show()


# Senal en frequencia/tiempo
# Convertir de OTFS a OFDM
# Hacer ifft por fila y luego la fft por columna
X = sqrt(M)/sqrt(N)*np.fft.fft(np.fft.ifft(x,axis=1),axis=0)
#pasar OFDM a tiempo
W = np.ones((N,M),dtype=complex)
#pasar OFDM a tiempo
s = sqrt(N)*W*np.fft.ifft(X,axis=0)
#Energia total trama
stxpw_nsit=dataPositions.size+ppower

s=s/sqrt(stxpw_nsit/(N*M))
#Agregar Ciclic prefix
stx = s[-(CP+1):-1]
#Agregar Ciclic prefix
stx = np.concatenate((stx,s))
stx = stx.flatten()
stx = np.concatenate((stx,np.zeros(ChanModDelayEVA)))

Fade = rayleightFading(fs=Fs, fm=70)
gen = Fade.Generate(size = stx.size)
#Fade.Plot(gen)

#Apply rayleight fading
ChanRx = gen*stx
#Generate guassian noise
g1 = np.random.normal(0, 1, stx.size)
g2 = np.random.normal(0, 1, stx.size)
g  = g1-1j*g2
g  = sqrt(10**(-SNR/20))*g
#Add complex guassian noise
ChanRx = ChanRx+ g

#Compensacion Delay Modulo de Canal- Despues de Canal
ChanRx = ChanRx[ChanModDelayEVA:] # remove from 0 to ChanModDelayEVA

#Recepccion y eliminacion de CP
Rx = ChanRx.reshape((N+CP,M))  #N+CP, M
Rx = Rx[CP:,:]  # Eliminar CP
RxOFDM=1/sqrt(N)*(1/W)*np.fft.fft(Rx,axis=0) # Pasar tiempo a trama OFDM
RxOTFS=sqrt(N)/sqrt(M)*np.fft.ifft(np.fft.fft(RxOFDM,axis=1),axis=0) # Senal en frequencia/tiempo a OTFS

ax = sns.heatmap(data = abs(RxOTFS))
plt.show()
