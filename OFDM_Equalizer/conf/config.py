import os
#ls -Art | tail -n 1
src         = os.path.dirname(os.path.abspath(__file__))+"/../"

base       = src+"models/"
Test_PAHT  = src+"Test/"
PLOTS_PATH = src+"plots/"
MATH_PATH  = src+"Data/"

pth_complete = base + "OFDM_Eq_SNR_(45_24)_(BOTH)_-28_11_2022-0_13.pth"

pth1         = base + "OFDM_Eq_SNR_(45_25)_(REAL)_-25_11_2022-19_48.pth"
pth2         = base + "OFDM_Eq_SNR_(45_25)_(IMAG)_-25_11_2022-19_48.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(45_24)_(ANGLE)_-7_12_2022-10_4.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_15)_(ABS)_-9_11_2022-15_37.pth"


BEST_SNR   = 45
WORST_SNR  = 24
STEP_SNR   = -5
EPOCHS     = 1
LEARNING_RATE = .0007
EPSILON = .007
#for SNR 45, LR=.001, EPISLON = .01

##TESTING and GOLDEN
GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2