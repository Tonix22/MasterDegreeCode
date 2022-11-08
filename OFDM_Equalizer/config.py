

base         = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/"
pth_complete = base + "OFDM_Eq_SNR_(40_5)_(BOTH)_-2_11_2022-15_13.pth"

pth1         = base + "OFDM_Eq_SNR_(40_5)_(REAL)_-3_11_2022-9_6.pth"
pth2         = base + "OFDM_Eq_SNR_(40_5)_(IMAG)_-3_11_2022-9_6.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(40_5)_(ANGLE)_-1_11_2022-15_35.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_5)_(ABS)_-1_11_2022-15_46.pth"


BEST_SNR   = 40
WORST_SNR  = 5
EPOCHS     = 5
LEARNING_RATE = .0001
EPSILON = .0001