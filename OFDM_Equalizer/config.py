#src        = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/"
src        = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/"

base       = src+"models/"
Test_PAHT  = src+"Test/"
PLOTS_PATH = src+"plots/"
MATH_PATH  = src+"Data/"

pth_complete = base + "OFDM_Eq_SNR_(40_5)_(BOTH)_-2_11_2022-15_13.pth"

pth1         = base + "OFDM_Eq_SNR_(35_25)_(REAL)_-10_11_2022-11_32.pth"
pth2         = base + "OFDM_Eq_SNR_(35_25)_(IMAG)_-10_11_2022-11_32.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(45_35)_(ANGLE)_-15_11_2022-23_31.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_15)_(ABS)_-9_11_2022-15_37.pth"


BEST_SNR   = 45
WORST_SNR  = 35
STEP_SNR   = -1
EPOCHS     = 1
LEARNING_RATE = .001
EPSILON = .01
#for SNR 45, LR=.001, EPISLON = .01

##TESTING and GOLDEN
GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2