

base         = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/models/"
pth_complete = base + "OFDM_Eq_SNR_(40_5)_(BOTH)_-2_11_2022-15_13.pth"

pth1         = base + "OFDM_Eq_SNR_(40_5)_(REAL)_-3_11_2022-9_6.pth"
pth2         = base + "OFDM_Eq_SNR_(40_5)_(IMAG)_-3_11_2022-9_6.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(35_20)_(ANGLE)_-10_11_2022-10_52.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_15)_(ABS)_-9_11_2022-15_37.pth"


BEST_SNR   = 45
WORST_SNR  = 5
STEP_SNR   = -1
EPOCHS     = 1
LEARNING_RATE = .001
EPSILON = .01
#for SNR 45, LR=.005, EPISLON = .02