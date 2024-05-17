max_UE_speed_kmh = 100;
max_UE_speed     = (max_UE_speed_kmh*1000)/3600;

%maximum Doppler spread(one-sided)
nu_max = (max_UE_speed*fc)/c;
%maximum normalized Doppler spread(one-sided)
k_max  = nu_max/Doppler_resolution;

%% Models
%Delays are in nanometers
% EPA
delay_EPA = [0,     30,   70,   90,  110,   190,   410]*10e-9;
pdp_EPA   = [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];

% EVA model
delays_EVA = [0,     30,  150,  310,  370,  710, 1090,  1730,  2510]*10e-9;
pdp_EVA    = [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9];

% ETU model
delays_ETU = [0,      50,  120, 200, 230, 500, 1600, 2300, 5000]*10e-9 ;
pdp_ETU    = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0];










