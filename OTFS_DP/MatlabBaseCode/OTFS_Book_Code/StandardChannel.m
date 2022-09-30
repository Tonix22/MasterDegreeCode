%choose channel model, for example: EVA
delays = delay_EPA;
pdp    = pdp_EVA;
pdp_linear = 10.^(pdp/10);
%normalization
pdp_linear = pdp_linear/sum(pdp_linear);

%number of propagtion paths (taps)
taps = length(pdp);

%generate channel coefficientes (Rayleight fading)
g_i = sqrt(pdp_linear).*(sqrt(1/2)*(randn(1,taps)+1i*randn(1,taps)));

%generate delay taps (assuming integer delay taps)
%delay_resolution = 1/(M*delta_f);
l_i = round(delays./delay_resolution);
% Generate Doppler taps (assuming Jakes spectrum)
k_i = k_max*cos(2*pi*rand(1,taps));