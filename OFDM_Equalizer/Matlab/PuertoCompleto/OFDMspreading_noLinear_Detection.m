clc;
clear all;
close all;
addpath('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/Matlab/PuertoLib');

%% parameter 

% QAM modulation order
M = 4; 
fftSize = 48;

k = log2(M);
constellation = qammod(0:M-1,M,"gray");    

%% channel V2V
channel = load('/home/tonix/Documents/MasterDegreeCode/Data/v2v80211p_LOS.mat');

H = channel.data.vectReal32b;
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
        tx = randi([0 1],fftSize*log2(M),1);
       
        % QAM modulate using 'Gray' symbol mapping
        qpskSig = qammod(tx,M,"gray","InputType","bit","UnitAveragePower",true);
         
        % pass throuht v2v channel
        %  [ H, G ] = MultipathFadingChannel(fftSize,sampleRate, maxDopplerShift, delayVector, gainVector, KFactor, specDopplerShift );
        G = H(:,:,channelCont);
        channelCont = channelCont +1;
        if channelCont == 10000
            channelCont = 1;
        end
      
        TxSig = qpskSig;
        TxSig = fft(qpskSig,fftSize); 
         
        RxSignal = G*TxSig;
        
        H1 = ifft(G,fftSize);
        H1 = fft(H1.');
        H1 = H1.';
        %figure
        %imagesc(mag2db(abs(H1)));
        %colorbar

        RxSignal = awgn(RxSignal,SNR(n),"measured");
        rxSig=ifft(RxSignal,fftSize);
        NoiseVar=10^(-SNR(n)/10);
        NoiseAmpl=sqrt(NoiseVar);
        nstd = sqrt( (1 /( 10^(SNR(n)/10 ) ) )/2 );

       %% nolinear detection
       %[Q R] = qr(G);
       %yp = Q'*rxSig;
       [yp,R,orden] = MMSESortedQRC(H1,nstd,rxSig,48,0);
       %[rxSig, nodos] = OSIC_Det(yp,R,constellation,orden);
       rxSig  = QRM_Det4b(yp,R,constellation,orden);
       
       %% LS
       % rxSig=G\rxSig;
       %% LMMSE
       % rxSig=inv((G'*G + NoiseVar*eye(fftSize))) * G'*rxSig;

        %% Demulate
        rx = qamdemod(rxSig,M, 'OutputType', 'bit');  % Demodulate
        
        % Calculate the number of bit errors
        nErrors = biterr(tx,rx);
        
        % Increment the error and bit counters
        numErrs = numErrs + nErrors;
        numBits = numBits + fftSize*log2(M);
        
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




