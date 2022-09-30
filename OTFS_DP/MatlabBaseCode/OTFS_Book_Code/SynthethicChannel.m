% number of propagation paths
taps = 6;
%maximum normalized delay and Doppler spred
l_max = 4;
k_max = 4;
%generate normalized delay and Doppler spread
g_i = sqrt(1/taps).*sqrt(1/2)*(randn(1,taps)+1*i*randn(1,taps));

%generate delay taps uniformely from [0,l_max]
l_i = [randi([0,l_max],1,taps)];
l_i = l_i - min(l_i)

%generate Doppler taps (assuming uniform spectrum) [-k_max,k_max])
k_i = k_max-2*k_max*rand(1,taps);