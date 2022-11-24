clear all;

loadmatlab;

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
        %H = fft(H,[],1)/sqrt(size(LOS,1));
        %H = eye(size(H,1)).* diag(H);
        Y = H*X;
        Y = Y+sqrt((10^(-SNR/10))/2)*(randn(numel(Y),1)+1j*randn(numel(Y),1));
        %H = H+sqrt(10^(-SNR/20))*(randn(size(H))+1j*randn(size(H)));
        X_hat = inv(H'*H+eye(48)*(10^(-SNR/10)))*H'*Y; %MSE
        %X_hat = diag(Y./H);
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

figure1 = figure;
semilogy(SNRVECT,BER)
title('BER vs SNR');
t = datetime('now');
t.Format = 'MMM_dd_yyyy-HH_mm_ss';
name =  strcat('plots/GolderMatlab_Ber_',string(t));
jpg  = strcat(name,'.jpg')

saveas(figure1,jpg)









