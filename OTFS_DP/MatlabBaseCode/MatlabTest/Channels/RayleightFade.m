%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RAYLEIGH FADING SIMULATOR BASED UPON YOUNG'S METHOD
% N is the number of points in the frequency domain
% fm is the Doppler frequency in Hz
% fs is the sampling frequency in Hz
% Copyright RAYmaps (www.raymaps.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;                          
 
N=2^20;
fm=300; % doppler frequency
fs=7.68e6;
Ts=1/fs;                                
 
% Generate 2xN IID zero mean Gaussian variates
g1=randn(1,N);  
g2=randn(1,N);
g=g1-j*g2;                              
 
% Generate filter F
F  = zeros(1,N);
dopplerRatio = fm/fs; 
km = floor(dopplerRatio*N);

for k=1:N
    if k==1,
        F(k)=0;
    elseif k>=2 && k<=km,
        F(k)=sqrt(1/(2*sqrt(1-((k-1)/(N*dopplerRatio))^2)));
    elseif k==km+1,
        F(k)=sqrt(km/2*(pi/2-atan((km-1)/sqrt(2*km-1))));
    elseif k>=km+2 && k<=N-km,
        F(k) = 0;
    elseif k==N-km+1,
        F(k)=sqrt(km/2*(pi/2-atan((km-1)/sqrt(2*km-1))));
    else
        F(k)=sqrt(1/(2*sqrt(1-((N-(k-1))/(N*dopplerRatio))^2)));
    end    
end

% Multiply F with Gaussian random sequence
X=g.*F;

% Take IFFT
x=ifft(X,N);
r=abs(x);
r=r/mean(r); % Canal finalizado


% Plot the Rayleigh envelope
T=length(r)*Ts;
t=0:Ts:T-Ts;
plot(t,10*log10(r))
xlabel('Time(sec)')
ylabel('Signal Amplitude (dB)')
axis([0 0.05 -15 5])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%