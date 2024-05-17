%Escenario Base usando Estimacion por umbral Canal EVA con F_doppler = 5 70 y 300 Hz
% NOTA: Usando ventana de estimacion de 0:ChanSizeDelay y -ChanSizeDoppler:ChanSizeDoppler


close all
clear

%% Ajustes Generales
Fs = 7.68e6; % Frecuencia de muestreo-Sample Rate
Ts = 1/Fs;   % Periodo de muestreo
N  = 512;    % Delay-Frecuency elements(taps) 1024
M  = 128;    % Doppler-Time elements    48
CP = 16;     % Cyclic prefix

%% QAM DATA
modorder            = 4;  %constelation size

%% Maximum channel size
ChanSizeDelay       = 20; %Estimated maximum channel delay size (actual value is 2x+1)(conventional OTFS method)
ChanSizeDoppler     = 5;  %Estimated maximum channel doppler size (actual value is 2x+1)(conventional OTFS method)

%% Spacing
DelayDataSpacing    = 3;  %spacing tabs 3
DopplerDataSpacing  = 3;

%pilot margin
% size of pilot
pilotmargingdelay   = ChanSizeDelay;     %Margen eje delay ventana-piloto
pilotmargingdoppler = ChanSizeDoppler*2; %Margen eje doppler ventana-piloto

%signal integrity
SNR     = 60;
ppower  = 10000; %pilot power
SNRVECT = 0:5:30; % SNR iterator


% it=2;
% ?   por que 1000? y 444.44 ? 
frames   = 1000;     % Frames 40
Fdoppler = 444.44; % 70-300 Hz

%OCID
% K=64; %frecuency periodicity ?-numero de pilotos por simbolo OFDM
% L=24; %Time periodicity ?   %numero de pilotos por trama en eje Temporal


%% * CREACIÓN DEL CANAL DE RAYLEIGHT * 

%% Channel
% % Propiedades de Canal Rayleigh                                                                                                                                                                                                                               
% chan = comm.RayleighChannel(...
%         "SampleRate",          1/Ts,...
%         "NormalizePathGains",  true,...       
%         "PathDelays",          (0:10)*0.9*Ts,...
%         "AveragePathGains",    10*log10(exp(-2*(0:10))),...
%         "MaximumDopplerShift", Fdoppler,...
%         "RandomStream",        "mt19937ar with seed",...
%         "Seed",                randseed,...
%         "PathGainsOutputPort", true,...
%         "FadingTechnique",     "Filtered Gaussian noise",...
%         "InitialTimeSource",   "Property",...
%         "Visualization",       "Off");
% chantest=chan.clone;    %Canal de Prueba

%% Channel 2 LTE EVA
chcfg.NRxAnts         = 1;    
chcfg.MIMOCorrelation = 'Low';
chcfg.NormalizeTxAnts = 'On';
chcfg.DelayProfile    = 'EVA';
chcfg.DopplerFreq     = Fdoppler; %EVA 5 -70 Hz
chcfg.SamplingRate    = 1/Ts; 
chcfg.InitTime        = 0;
chcfg.NTerms          = 16;       % Opcional
chcfg.ModelType       = 'GMEDS';  % Opcional--o 'Dent'
chcfg.NormalizePathGains = 'On';  % Opcional
chcfg.InitPhase       = 'Random'; % Opcional
chcfg.Seed            = 1; %Poner a cero para seed aleatoria


x = zeros(N,M); %Tamano Trama OTFS
% testsig=x;

%% SYMBOLS timing
%Delay-Frecuency + Cyclic Prefix
Tsymbol=(N+CP)*Ts;  %Tiempo de Simbolo 
Fsymbol=1/Tsymbol;  %Simbolos por segundo

%% Generacion Indices y Ventana Piloto
%  Posicion fila/columna de los datos--espaciado
NVector=3:DelayDataSpacing:N-3;
MVector=1:DopplerDataSpacing:M;

% Creacion de indices para posicion de datos
dpos = zeros(numel(NVector)*numel(MVector),2);  %Arreglo para par de coordenadas  
k=1;    %Numero de posiciones de datos-1
for i = NVector
    for j = MVector
        dpos(k,:)=[i,j];
        k=k+1;
    end
end
dataPositions=dpos; %Coordenadas-indices para pos de datos

%pilot position
pilotPos=[N/2, M/2];
x(pilotPos(1),pilotPos(2))=sqrt(ppower);  %pilot power



%%
%pilot margins 
pilotMargins=zeros(((1+pilotmargingdelay*2)*(2*pilotmargingdoppler+1)),2);% Para coordenadas de ventana alrededor de piloto
k=1;
for i=pilotPos(1)-pilotmargingdelay:pilotPos(1)+pilotmargingdelay
    for j=pilotPos(2)-pilotmargingdoppler:pilotPos(2)+pilotmargingdoppler
        pilotMargins(k,:)=[i,j];
        k=k+1;
    end
end
% pilotMargins(k/2,:)=[]; %Quita posicion donde va piloto
% No es lo mismo M*N/2 que N/2 + ((M/2)-1)*N, para encontrar la posicion de
% piloto

%Vectorizacion de posicion de datos
vecdataPositions=dataPositions(:,1)+(dataPositions(:,2)-1)*N;
%Vectorizacion de posicion de guardas (ventana alrededor piloto)
vecpilotMargins=pilotMargins(:,1)+(pilotMargins(:,2)-1)*N;
%Quitar indices de datos que quedan dentro de ventana de piloto
%#NOTA:piloto no esta protegido
for i=1:numel(vecpilotMargins)
    index=find(vecdataPositions==vecpilotMargins(i));
    vecdataPositions(index)=[];
    dataPositions(index,:)=[];
end
%Quitar posicion de piloto de la ventana creada
pilotMargins(k/2,:)=[]; 
vecpilotMargins(k/2)=[];

%% SYMPLETIC FOURIER TRANSFORM
% SFFT = FFT en dimension columnas(Delay-frec),IFFT dim filas (doppler-tiempo)
% Compensacion de sqrt (M) y (N) para conservar misma potencia en FFT 
Xdummy=sqrt(M)/sqrt(N)*fft(ifft(x,[],2),[],1); %Senal en frequencia/tiempo de X-solopiloto

%% Señal Dummy de OFDM a tiempo
W=ones(size(Xdummy)); %Ventana rectangular unitaria
% Tx Dummy
sdummy=sqrt(N)*W.*ifft(Xdummy,[],1); %pasar OFDM a tiempo, multiplicando ventana
% sdummy=sdummy/sqrt(ppower/(N*M));  %senal normalizada a potencia unitaria antes de insertar CP
stxdummy=[sdummy(end-CP+1:end,:); sdummy];  %Agregar CP
% stxdummy(:)'*stxdummy(:)/numel(stxdummy(:));    %Energia de la señal trama
stxdummy=stxdummy(:)/sqrt(ppower/((N+CP)*M)); %senal transmitida en tiempo sin datos--Normalizada a POT UNIT
%
% Compensacion Delay Modulo de Canal- Antes de Canal
% ChanModDelay=chan.info.ChannelFilterDelay;
ChanModDelayEVA=7;
stxdummy=[stxdummy ;zeros(ChanModDelayEVA,1)];


%% Generacion/ITERACION de Frames

BER=zeros(numel(SNRVECT),1);
for SNR=SNRVECT %Iteracion por SNR
   errors=0;
    for k=1:frames
        %generar datos
        txbits=randi([0 1],size(vecdataPositions(1:end),1)*log2(modorder),1);   %Generar # bits para una trama OTFS
        %txsymbols=modulator.step(txbits);
        % Modulacion de bits
        txsymbols = qammod(txbits, modorder, 'gray', 'InputType', 'bit','UnitAveragePower', true);
        x(vecdataPositions(1:end))=txsymbols;   % Acomodo en grid x
        %convertir de OTFS a OFDM
        %ifft(x,[],2) -> returns the inverse Fourier transform of each row.
        X=sqrt(M)/sqrt(N)*fft(ifft(x,[],2),[],1); %Senal en frequencia/tiempo

        %W=ones(size(X)); %Ventana
        s=sqrt(N)*W.*ifft(X,[],1); %pasar OFDM a tiempo
        stxpw_nsit=(size(dataPositions,1)+ppower);  % Energia total trama
        s=s/sqrt(stxpw_nsit/(N*M));                 % Normalizar a potencia unitaria
        stx=[s(end-CP+1:end,:); s]; %Agregar CP
        stx=stx(:); %senal transmitida en tiempo
        stx=stx/sqrt(stx'*stx/numel(stx));  %Forzar potencia unitaria despues de agregar CP
        %% Compensacion Delay Modulo de Canal- Antes de Canal
        % ChanModDelay=chan.info.ChannelFilterDelay;
        stx=[stx ;zeros(ChanModDelayEVA,1)];

        %% Exposicion a traves de Canal Rayleigh
        [ChanRx,~] = lteFadingChannel(chcfg,stx); %Senal Sx   
        % [ChanRxdummy,~] = chantest(stxdummy);   %Señal Sx Dummy
        [ChanRxdummyEVA,infoEVA]=lteFadingChannel(chcfg,stxdummy);
        %ChanRx=stx;
        % SNR = 20*log10(Amplitud Señal/ Amplitud Ruido)=10*log10(Amp_señal_rms/Amp_Ruido_rms)^2
        ChanRx=ChanRx+sqrt(10^(-SNR/20))*(randn(numel(ChanRx),1)+1j*randn(numel(ChanRx),1));    %Rx+Ruido Complejo

        %% Compensacion Delay Modulo de Canal- Despues de Canal
        ChanRx=ChanRx(ChanModDelayEVA+1:end);

        ChanRxdummy=ChanRxdummyEVA(ChanModDelayEVA+1:end);

        %% Recepcion-Reshape hasta señal en array MxN en DD
        Rx=reshape(ChanRx,N+CP,M);  %N+CP, M
        Rx=Rx(CP+1:end,:);  % Eliminar CP
        RxOFDM=1/sqrt(N)*1./W.*fft(Rx,[],1); %pasar tiempo a trama OFDM
        RxOTFS=sqrt(N)/sqrt(M)*ifft(fft(RxOFDM,[],2),[],1); %Senal en frequencia/tiempo a OTFS
        %%%%%%%%%%%%%%%%hacer zero pading en N para compensar no-circularidad
        %Recepcion RxDummy-Solo piloto
        Rxdummy=reshape(ChanRxdummy,N+CP,M);
        Rxdummy=Rxdummy(CP+1:end,:);
        RxOFDMdummy=1/sqrt(N)*W.*fft(Rxdummy,[],1); %pasar tiempo a trama OFDM
        RxOTFSdummy=sqrt(N)/sqrt(M)*ifft(fft(RxOFDMdummy,[],2),[],1); %Senal en frequencia/tiempo a OTFS
        %%%%%%%%%%%%%%%%hacer zero pading en N para compensar no-circularidad


        %% Estimacion de Canal desde piloto en OTFS--PSF-Point Spread Function
        PSF = RxOTFS((-ChanSizeDelay:ChanSizeDelay)+N/2+ChanSizeDelay,(-ChanSizeDoppler:ChanSizeDoppler)+M/2);%Recorte a tamano de Ventana 2l+1, 2k+1
        PSFideal = RxOTFSdummy((-ChanSizeDelay:ChanSizeDelay)+N/2+ChanSizeDelay,(-ChanSizeDoppler:ChanSizeDoppler)+M/2);
        % PSF=PSF.*(abs(PSF)>.5);    %Recorte de umbral
        PSF=PSFideal;   %Usar Caso ideal

        PSFOFDM = [[PSF ; zeros(N-size(PSF,1),size(PSF,2))] zeros(N,M-size(PSF,2))]; %hacer zero pading en N para compensar no-circularidad
        %padarray(PSF,[N-size(PSF,1), M-size(PSF,2)],0,'post');
        PSFOFDM=sqrt(M)/sqrt(N)*fft(ifft(PSFOFDM,[],2),[],1); %Senal de DD a frequencia/tiempo

        % %% Estimacion de Canal por BEM usando SIT(Realiza antes de pasar de TF a DD)--Uso de BEM compensa offset de trama OTFS(No necesario circshift)
        % rxOFDMPilots=RxOFDM(vecOFDMPilotPos)./OFDMPilotval; % Solo posicion pilotos / pot.de piloto (normalizacion de amp)
        % PSFOfdmBem=estmat*rxOFDMPilots; %Multiplicacion pilotos por matriz de estimacion
        % PSFOfdmBem=reshape(PSFOfdmBem,N,M); %Matriz de canal estimado BEM en dominio TF


        %% Ecualizacion en dominio de la frecuencia
        iPSFOFDM=conj(PSFOFDM)./(abs(PSFOFDM).^1+.5);%Wiener en Frecuencia usando Piloto
        % iPSFOFDM=conj(PSFOfdmBem)./(abs(PSFOfdmBem).^1+.5);%Wiener en Frecuencia por BEM ?? (Normalizar y Offset .5)

        RxEst=RxOFDM.*iPSFOFDM; %Rx Ecualizada en TF-
        RxOTFSEst=sqrt(N)/sqrt(M)*ifft(fft(RxEst,[],2),[],1); %Se�al RxEst en frequencia/tiempo a dominio OTFS

        % Centrar trama OTFS en funcion de posicion piloto
        PSFEq = RxOTFSEst((-ChanSizeDelay:ChanSizeDelay)+N/2,(-ChanSizeDoppler:ChanSizeDoppler)+M/2-ChanSizeDoppler); %Ventana de piloto estimada en DD
        % estima offset para hacer corrimiento circular y adecuar arreglo otfs
        [z,indx]=max(abs(PSFEq(:)));    %Valor pico de CIR e indice
        xi=mod(indx,ChanSizeDelay*2+1); %Posicion en dominio delay
        if(xi==0)
            xi=ChanSizeDelay*2+1;   %Esta en el borde
        end
        %yi=floor((indx-(xi))/(ChanSizeDoppler*2+1))-ChanSizeDoppler;
        yi=floor((indx-1)/(ChanSizeDelay*2+1))+1;   %logica: resultado de division es numero de trama_entera+1=yi
        % xoffset=xi-ChanSizeDelay-floor(ChanSizeDelay/2)+2;  %??
        xoffset=round((1+2*ChanSizeDelay)/2)-xi; 
        % yoffset=yi-1-8;         %??
        yoffset=round((1+2*ChanSizeDoppler)/2)-yi+ChanSizeDoppler;
        RxOTFSEst=RxOTFSEst/z*sqrt(ppower); %Compensacion potencia de piloto
        % Acomodo de trama otfs aprovechando propiedad de periodicidad de la SFFT
        RxOTFSEst=circshift(circshift(RxOTFSEst,xoffset,1),yoffset,2);  % Acomodo de trama OTFS con offsets
        %PSFEq Estimada con circshift
        PSFEq = RxOTFSEst((-ChanSizeDelay:ChanSizeDelay)+N/2,(-ChanSizeDoppler:ChanSizeDoppler)+M/2);

        %% Demodulacion de Datos de bloque OTFS en posicion de datos
        %rxbits=demodulator.step(RxOTFSEst(vecdataPositions));
        rxbits = qamdemod(RxOTFSEst(vecdataPositions), modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        %% Calculo de Errores
        errors=sum(abs(txbits-rxbits))+errors;
    end
    BER(SNR==SNRVECT)=errors/(numel(txbits)*frames);
    SNR
end

% %Trama Rx en dominio OTFS
% surf(real(RxOTFS))
% shading interp
% Rx Ecualizada en dominio OTFS (DD) 
surf(abs(RxOTFSEst)); title('Rx Frame - equalized');
shading interp
% Trama Rx en dominio DD
figure
surf(abs(RxOTFS)); title('Rx Frame');
shading interp
% Ventana de Piloto para estimacion en DD
figure
surf((abs(PSF))); title('Real PSF');
%shading interp
%Ventana de piloto estimada-Ecualizada en DD
figure
surf(abs(PSFEq)); title('Equalized PSF');
%shading interp

figure  % Grafica de puntos de constelacion
f1=scatter(real(RxOTFSEst(vecdataPositions)),imag(RxOTFSEst(vecdataPositions)),'.');
title('Constelacion Rx');
%hold on;
%scatter(real(modulator.constellation),imag(modulator.constellation),'r+');
a1=f1.get('Parent');
a1.XAxis.Limits=[-2 2];
a1.YAxis.Limits=[-2 2];

speceff=log2(modorder)*numel(vecdataPositions)/((N+CP)*M)   % Eficiencia espectral
% BER VS SNR
figure
semilogy(SNRVECT,BER)
title('BER vs SNR');
% % Matriz de Canal Estimado
% figure
% imagesc(real(PSFOfdmBem));
% title('PSFOFDM-BEM');
% Matriz de Canal Real
figure
imagesc(real(PSFOFDM));
title('PSFOFDM Rx');










