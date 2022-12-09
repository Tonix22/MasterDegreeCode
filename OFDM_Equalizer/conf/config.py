import os
#ls -Art | tail -n 1
src         = os.path.dirname(os.path.abspath(__file__))+"/../"

base       = src+"models/16QAM/"
Test_PAHT  = src+"Test/"
PLOTS_PATH = src+"plots/"
MATH_PATH  = src+"Data/"

pth_complete = base + "OFDM_Eq_SNR_(20_5)_(BOTH)_-8_12_2022-16_42.pth"

pth1         = base + "OFDM_Eq_SNR_(45_25)_(REAL)_-25_11_2022-19_48.pth"
pth2         = base + "OFDM_Eq_SNR_(45_25)_(IMAG)_-25_11_2022-19_48.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(45_25)_(ANGLE)_-8_12_2022-13_45.pth"
pth_abs    = base + "OFDM_Eq_SNR_(45_25)_(ABS)_-8_12_2022-14_34.pth"


BEST_SNR   = 40
WORST_SNR  = 20
STEP_SNR   = -5
EPOCHS     = 1
LEARNING_RATE = .0005
EPSILON = .005
#for Phase, LR=.0007, EPISLON = .007

##TESTING and GOLDEN
GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2