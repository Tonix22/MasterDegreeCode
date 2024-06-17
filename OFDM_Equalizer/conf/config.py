import os
#ls -Art | tail -n 1
src         = os.path.dirname(os.path.abspath(__file__))+"/../"

base       = src+"models/16QAM/"
Test_PAHT  = src+"Test/"
PLOTS_PATH = src+"plots/"
MATH_PATH  = src+"../Data/kaggle_dataset"

pth_complete = base + "OFDM_Eq_SNR_(20_5)_(BOTH)_-8_12_2022-16_42.pth"

pth1         = base + "OFDM_Eq_SNR_(40_20)_(REAL)_-9_12_2022-12_5.pth"
pth2         = base + "OFDM_Eq_SNR_(40_20)_(IMAG)_-9_12_2022-12_7.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(40_20)_(ANGLE)_-9_12_2022-13_46.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_20)_(ABS)_-9_12_2022-14_5.pth"

pth_bce    = base + "OFDM_Eq_SNR_(40_20)_(FOUR)_-9_12_2022-11_35.pth"

pth_complex = base +"OFDM_Eq_SNR_(45_25)_(COMPLEX)_-16_12_2022-0_57.pth"

BEST_SNR   = 45
WORST_SNR  = 25
STEP_SNR   = -5
EPOCHS     = 1
LEARNING_RATE = .0005
EPSILON = .005
#for Phase, LR=.0007, EPISLON = .007

##TESTING and GOLDEN
GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2
GOLDEN_ACTIVE   = True
GOLDEN_FAST_RUN = False
GOLDEN_SAVE_RESULTS = True
GOLDEN_DATA_RATIO = 0.30