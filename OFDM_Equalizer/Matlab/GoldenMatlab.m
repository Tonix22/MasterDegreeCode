clear all;

loadmatlab;

% QAM DATA
modorder            = 4;  %constelation size

SNRVECT = 5:2:45;
BER_MSE  = zeros(numel(SNRVECT),1);
BER_LMSE = zeros(numel(SNRVECT),1);
frames   = 4000;
for SNR=SNRVECT
    errors_MSE = 0;
    errors_LMSE = 0;
    LOS_cnt  = 1;
    NLOS_cnt = 1;
    for i = 16001:1:20000
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

        %Channel matrix multiply
        Y = H*X;

        % Signal Power
        %Ps = sum(abs(Y).^2) / length(Y);
        % Noise power
        Pn = 1 / (10^(SNR/10));
        % Generate noise
        n   = sqrt(Pn/2)* complex(randn(size(Y)), randn(size(Y)));
        %y_n = awgn(Y,SNR);
        y_n = Y+n;

        %Y = Y+sqrt((10^(-SNR/10))/2)*(randn(numel(Y),1)+1j*randn(numel(Y),1));

        %MMSE
        X_hat_MSE  = inv(H'*H)*H'*y_n; %MSE
        %LMMSE
        X_hat_LMSE = inv(H'*H+eye(48)*(10^(-SNR/10)))*H'*y_n; %LMSE
        
        rxbits_MSE = qamdemod(X_hat_MSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_LMSE = qamdemod(X_hat_LMSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        
        % Calculo de Errores
        errors_MSE  = sum(abs(txbits-rxbits_MSE))+errors_MSE;
        errors_LMSE = sum(abs(txbits-rxbits_LMSE))+errors_LMSE;

        %if(mod(i , 500) == 0)
        %    disp(["SNR: ", SNR])
        %    disp(["BER_MSE: ", errors_MSE/(numel(txbits)*frames)])
        %    disp(["BER_LMSE: ", errors_MSE/(numel(txbits)*frames)])
        %end     
    end

    BER_MSE(SNR==SNRVECT)=errors_MSE/(numel(txbits)*frames);
    BER_LMSE(SNR==SNRVECT)=errors_LMSE/(numel(txbits)*frames);
end

figure1 = figure;
semilogy(SNRVECT,BER_MSE)
hold on;
semilogy(SNRVECT,BER_LMSE)

title('BER vs SNR');
t = datetime('now');
t.Format = 'MMM_dd_yyyy-HH_mm_ss';
name =  strcat('plots/GolderMatlab_Ber_',string(t));
jpg  = strcat(name,'.jpg')

saveas(figure1,jpg)









