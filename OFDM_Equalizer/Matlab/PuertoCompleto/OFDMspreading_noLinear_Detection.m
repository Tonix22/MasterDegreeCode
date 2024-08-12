clc;
clear all;
close all;


%% parameter 

M = 4; % QAM modulation order
CP = 0;
FFTSize = 48;

numSymPerFrame = 1;
k = log2(M);

constellation = qammod(0:M-1,M,"gray");    

%% channel V2V

%load('v2v80211p_LOS.mat')
load('v2v80211p_NLOS.mat')

 H = vectReal32b;
 channelCont=1;
%% system model 

SNR = 0:5:25; % Range of SNR values, in dB.
EffEsp=(log2(M)*(48)/(64));

for n= 1:length(SNR)
    
    
    % Reset the error and bit counters
    numErrs = 0;
    numBits = 0;


    while numErrs < 1e3 && numBits < 1e6
        % Generate binary data and convert to symbols
        tx = randi([0 1],FFTSize*log2(M),1);
       
        % QAM modulate using 'Gray' symbol mapping
         qpskSig = qammod(tx,M,"gray","InputType","bit","UnitAveragePower",true);
         
         % pass throuht v2v channel
      %   [ H, G ] = MultipathFadingChannel(FFTSize,sampleRate, maxDopplerShift, delayVector, gainVector, KFactor, specDopplerShift );
         G = H(:,:,channelCont);
         channelCont = channelCont +1;
         if channelCont == 10000
             channelCont =1
         end
      
         TxSig = qpskSig;
         TxSig = fft(qpskSig,FFTSize); 
         
         
      
        RxSignal = G*TxSig;
        
        H1= ifft(G,FFTSize);
        H1= fft(H1.');
        H1=H1.';
         %figure
         %imagesc(mag2db(abs(H1)));
         %colorbar

        RxSignal = awgn(RxSignal,SNR(n),"measured");

        
        rxSig=ifft(RxSignal,FFTSize);


 
        NoiseVar=10^(-SNR(n)/10);

        NoiseAmpl=sqrt(NoiseVar);
       
       nstd = sqrt( (1 /( 10^(SNR(n)/10 ) ) )/2 );
       
       
       %% nolinear detection
       
       %[Q R] = qr(G);
       
       %yp = Q'*rxSig;
       
       
       [yp,R,orden] = MMSESortedQRC(H1,nstd,rxSig,48,0);
       
       %[rxSig, nodos] = OSIC_Det(yp,R,constellation,orden);

       rxSig  = QRM_Det4b(yp,R,constellation,orden,M);
       
       
   
 
       %% LS
       
       % rxSig=G\rxSig;

       
       %% LMMSE
       
       %rxSig=inv((G'*G + NoiseVar*eye(FFTSize))) * G'*rxSig;

        
      %% Demulate
   

        rx = qamdemod(rxSig,M, 'OutputType', 'bit');  % Demodulate
        
        % Calculate the number of bit errors
        nErrors = biterr(tx,rx);
        
        % Increment the error and bit counters
        numErrs = numErrs + nErrors;
        numBits = numBits + FFTSize*log2(M);
        
    end
    
    
    %scatterplot(RxSig)

    % Estimate the BER
    berEst(n) = numErrs/numBits
    
end



%% plot



berTheorical = berawgn(SNR,'qam',M,'nondiff');

figure
%semilogy(SNR,berTheorical,'k');
%hold on
semilogy(SNR,berEst,'b*-');

legend('DFT-OFDM-NML');
xlabel('SNR (dB)','Interpreter','latex'); 
ylabel('BER','Interpreter','latex');
title('Binary QAM over V2V Channel');
grid on

set(gca, 'fontsize', 14)  %tama??o de letra
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');




