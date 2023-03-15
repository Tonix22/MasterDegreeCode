clear all;

loadmatlab;

% QAM DATA
modorder   = 16;  %constelation size

SNRVECT    = 5:2:45;
BER_MSE    = zeros(numel(SNRVECT),1);
BER_LMSE   = zeros(numel(SNRVECT),1);
BER_NearML = zeros(numel(SNRVECT),1); % Add BER for Near ML
frames     = 4000;

%& extract normalizing factor
H_gen  = cat(3, LOS, NLOS);
Ph     = sum(abs(H_gen(:)).^2)/numel(H_gen);
H_norm = H_gen/Ph;

conste = qammod(0:modorder-1, modorder, 'gray', 'UnitAveragePower', true); % Constellation points for Near ML
index = 1:size(LOS, 1); % Index vector for Near ML

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

        H = H/Ph;
        %Channel matrix multiply
        Y = H*X;

        % Signal Power
        Ps = sum(abs(Y).^2) / length(Y);
        % Noise power
        Pn = Ps / (10^(SNR/10));
        % Generate noise
        n   = sqrt(Pn/2)* complex(randn(size(Y)), randn(size(Y)));
        y_n = Y+n;

        %MMSE
        X_hat_MSE  = inv(H'*H)*H'*y_n; %MSE
        %LMMSE
        X_hat_LMSE = inv(H'*H+eye(48)*Pn)*H'*y_n; %LMSE

        % Near ML
        R = H' * H;
        [X_hat_NearML, nodos] = Near_ML(y_n, R, conste, index); % Call Near ML function
        
        rxbits_MSE    = qamdemod(X_hat_MSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_LMSE   = qamdemod(X_hat_LMSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_NearML = qamdemod(X_hat_NearML, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true); % Demodulate Near ML

        % Calculo de Errores
        errors_MSE    = sum(abs(txbits-rxbits_MSE))+errors_MSE;
        errors_LMSE   = sum(abs(txbits-rxbits_LMSE))+errors_LMSE;
        errors_NearML = sum(abs(txbits-rxbits_NearML))+errors_NearML;

        %if(mod(i , 500) == 0)
        %    disp(["SNR: ", SNR])
        %    disp(["BER_MSE: ", errors_MSE/(numel(txbits)*frames)])
        %    disp(["BER_LMSE: ", errors_MSE/(numel(txbits)*frames)])
        %end     
    end

    BER_MSE(SNR==SNRVECT)    = errors_MSE/(numel(txbits)*frames);
    BER_LMSE(SNR==SNRVECT)   = errors_LMSE/(numel(txbits)*frames);
    BER_NearML(SNR==SNRVECT) = errors_NearML/(numel(txbits)*frames);
end

figure1 = figure;
semilogy(SNRVECT,BER_MSE)
hold on;
semilogy(SNRVECT,BER_LMSE)
semilogy(SNRVECT,BER_NearML)

title('BER vs SNR');
t = datetime('now');
t.Format = 'MMM_dd_yyyy-HH_mm_ss';
name =  strcat('plots/GolderMatlab_Ber_',string(t));
jpg  = strcat(name,'.jpg')
legend('LS','LMMSE')
saveas(figure1,jpg)









