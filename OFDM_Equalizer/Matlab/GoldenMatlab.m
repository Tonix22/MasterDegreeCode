clear all;

%load .mat files where Channels are saved
loadmatlab;
addpath('.') % add current path to directory

FastRun = false;

% QAM DATA
modorder   = 16;  %constelation size

SNRVECT    = 45:-2:5; % SNR range with step of

% Vectors where BER is saved
BER_MSE    = zeros(numel(SNRVECT),1);
BER_LMSE   = zeros(numel(SNRVECT),1);
BER_NearML = zeros(numel(SNRVECT),1);
% Testing data set range
frames     = 4000;

% Constellation points for Near ML
conste = qammod(0:modorder-1, modorder,'gray','UnitAveragePower', true);
% Index vector for Near ML
index  = 1:size(LOS, 1); 

for SNR=SNRVECT
    %Total errors for each model
    errors_MSE    = 0;
    errors_LMSE   = 0;
    errors_NearML = 0;
    %Channel indices
    LOS_cnt       = 1;
    NLOS_cnt      = 1;
    %Iterations for progres bar
    nIterations   = 4000;
    %ProgressBar
    hWaitBar = waitbar(0, ['SNR: ', num2str(SNR)]);

    for i = 16001:1:20000 % 4000 iterations
        %Progress
        progress = (i-16001) / nIterations;
        ML_curr_err    = errors_NearML/(48*4*(i-16001));
        LMMSE_curr_err = errors_LMSE/(48*4*(i-16001));
        waitbar(progress, hWaitBar, sprintf('SNR: %d, NML BER: %.5f Total: %.1f%%', SNR, ML_curr_err , progress * 100));

        %Alternated betwen LOS and NLOS channel
        if(bitand(i,1))
             H = LOS(:,:,LOS_cnt);
             LOS_cnt = LOS_cnt+1;
        end
        if(bitand(i,0))
             H = NLOS(:,:,NLOS_cnt);
             NLOS_cnt = NLOS_cnt+1;
        end
        %Tx data generation
        txbits = randi([0 1],size(LOS,1)*log2(modorder),1);
        X      = qammod(txbits, modorder, 'gray', 'InputType', 'bit','UnitAveragePower', true);

        %Channel matrix multiply
        Y   = H*X;

        % Signal Power
        Ps  = sum(abs(Y).^2) / length(Y);
        % Noise power
        Pn  = Ps / (10^(SNR/10));
        % Generate noise
        n   = sqrt(Pn/2)* complex(randn(size(Y)), randn(size(Y)));
        y_n = Y+n;

        %MMSE
        X_hat_MSE  = inv(H'*H)*H'*y_n; %MSE
        %LMMSE
        X_hat_LMSE = inv(H'*H+eye(48)*Pn)*H'*y_n; %LMSE

        % Near ML
        R = H;
        [X_hat_NearML, nodos] = Near_ML(y_n, R, conste, index); % Call Near ML function
        
        %qam demod
        rxbits_MSE    = qamdemod(X_hat_MSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_LMSE   = qamdemod(X_hat_LMSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_NearML = qamdemod(X_hat_NearML, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true); % Demodulate Near ML

        %Error Calculation
        errors_MSE    = sum(abs(txbits-rxbits_MSE))+errors_MSE;
        errors_LMSE   = sum(abs(txbits-rxbits_LMSE))+errors_LMSE;
        errors_NearML = sum(abs(txbits-rxbits_NearML))+errors_NearML;

    end
    % Close the waitbar when the computation is finished
    close(hWaitBar);

    BER_MSE(SNR==SNRVECT)    = errors_MSE/(numel(txbits)*frames);
    BER_LMSE(SNR==SNRVECT)   = errors_LMSE/(numel(txbits)*frames);
    BER_NearML(SNR==SNRVECT) = errors_NearML/(numel(txbits)*frames);
    if FastRun == true
       break;
    end
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
legend('LS','LMMSE','NearML')
saveas(figure1,jpg)









