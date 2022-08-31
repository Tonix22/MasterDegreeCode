from numpy import sqrt
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

N  = 20
fm = 70.0
df = (2*fm)/(N-1) # df the frequency spacing
fs = 1000
M  = round(fs/df) # M total number of samples in frequency
T  = 1/df
Ts = 1/fs

# Generating first Gaussian RV set
g   = randn(int(N/2))+1j*randn(int(N/2))
gc  = np.conj(g)
gcr = gc[::-1]
g1  = np.concatenate((gcr,g),axis=0)

# Generating second Guassian RV set
g=randn(int(N/2))+1j*randn(int(N/2))
gc=np.conj(g)
gcr=gc[::-1]
g2=np.concatenate((gcr,g),axis=0)

# Generate the doppler spectrum 
f = np.arange(-fm,fm+df,df)
S = 1.5/(np.pi*fm*sqrt(1-(f/fm)**2))
S[0]  = 2*S[1]-S[2]
S[-1] = 2*S[-2]-S[-3]

# Shaping the RV sequence g1 and taking IFFT
X = g1*sqrt(S)
X = np.concatenate((np.zeros(int((M-N)/2)),X),axis=0)
X = np.concatenate((X,np.zeros(int((M-N)/2))),axis=0)
x = np.abs(np.fft.ifft(X))

# Shaping the RV sequence g2 and taking IFFT 
Y = g2*sqrt(S)
Y = np.concatenate((np.zeros(int(((M-N)/2))), Y), axis=0)
Y = np.concatenate((Y, np.zeros(int(((M-N)/2)))), axis=0)
y = np.abs(np.fft.ifft(Y))

# Generating complex envelope
z = x+1j*y
r = np.abs(z)

# Plotting the envelope in the time domain 
t = np.arange(0,T,Ts)
#plt.plot(t,10*np.log10(r/np.max(r)),'b')
plt.plot(t,r,'b')
plt.show()
plt.xlabel('Time(msecs)')
plt.ylabel('Envelope(dB)')
plt.grid(True)
plt.title('Rayleigh Fading')