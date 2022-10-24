clear all;

folder = 'DataBasesOFDM_Channels';  % You specify this!


fullMatFileName = fullfile(folder,  'v2v80211p_LOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  LOS = load(fullMatFileName);
  LOS = LOS.vectReal32b;
end


fullMatFileName = fullfile(folder,  'v2v80211p_NLOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  NLOS = load(fullMatFileName);
  NLOS = NLOS.vectReal32b;
end

% QAM DATA
modorder            = 4;  %constelation size

SNRVECT = 5:5:50;
BER = zeros(numel(SNRVECT),1);
frames   = 20000;
for SNR=SNRVECT
    errors = 0;
    LOS_cnt  = 1;
    NLOS_cnt = 1;
    for i = 1:1:frames
        if(bitand(i,1))
             H = LOS(:,:,LOS_cnt);
             LOS_cnt = LOS_cnt+1;
        end
        if(bitand(i,0))
             H = NLOS(:,:,NLOS_cnt);
             NLOS_cnt = NLOS_cnt+1;
        end
        %generar datos
        txbits    = randi([0 1],size(LOS,1)*log2(modorder),1);
        X = qammod(txbits, modorder, 'gray', 'InputType', 'bit','UnitAveragePower', true);
        Y = H*X;
        Y = Y+sqrt(10^(-SNR/20))*(randn(numel(Y),1)+1j*randn(numel(Y),1));
        %H = H+sqrt(10^(-SNR/20))*(randn(size(H))+1j*randn(size(H)));
        I = inv(H'*H)*H'*Y; %MSE
        X_hat = H*I;
        rxbits = qamdemod(X_hat, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        % Calculo de Errores
        errors = sum(abs(txbits-rxbits))+errors;
        if(mod(i , 500) == 0)
            disp(["SNR: ", SNR])
            disp(["BER: ", errors/(numel(txbits)*frames)])
        end     
    end
    BER(SNR==SNRVECT)=errors/(numel(txbits)*frames);
end

figure
semilogy(SNRVECT,BER)
title('BER vs SNR');









