%number of Doppler bins (time slots)
N= 16;

%number of delay bins(subcarriers)
M=64;

% normalized DFT matrix
Fn = dftmtx(N);
Fn = Fn/norm(Fn);

%subcarrier spacing
delta_f = 15e3;

%block duration
T = 1/delta_f;

%carrier frequency 
fc=4e9;

% speed of light
c = 299792458;

%OTFS grid delay and Doppler resolution

delay_resolution = 1/(M*delta_f);
Doppler_resolution = 1/(N*T);

%% modulation
%modulation size
mod_size = 4;
%Number of information symboles in one frame
N_syms_per_frame = N*M;
% number of informaton bits in one frame
N_bits_per_frame = N*M*log2(mod_size);
%generate random bits
tx_info_bits = randi([0,1],N_bits_per_frame,1);

%QAM modulation 
tx_info_bits = qammod(tx_info_bits,mod_size,'gray','InputType','bit');

%Generate the MxN OTFS delay_Doppler frame
X = reshape(tx_info_bits,M,N);
x = reshape(X.',N*M,1);%X.' the dot is to dont modify complex part

Im = eye(M);

%row-column permutation matrix
P=zeros(N*M,N*M);
for j=1:N
    for i=1:M
        E=zeros(M,N);
        E(i,j)=1;
        P((j-1)*M+1:j*M,(i-1)*N+1:i*N)=E;
    end
end



% Method 1 (Eqs. (4.19) and (4.20))
X_tilda=X*Fn';
s=reshape(X_tilda,1,N*M);
% Method 2 (Eq. (4.35))
s1=P*kron(Im,Fn')*x;
% Method 3 (Eq. (4.35))s
s2=kron(Fn',Im)*P*x;

