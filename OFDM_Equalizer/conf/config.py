import os
#ls -Art | tail -n 1
src         = os.path.dirname(os.path.abspath(__file__))+"/../"

base       = src+"models/"
Test_PAHT  = src+"Test/"
PLOTS_PATH = src+"plots/"
MATH_PATH  = src+"Data/"


## TESTING PATHS

#Concatenated strategy
pth_complete = base + "OFDM_Eq_SNR_(20_5)_(BOTH)_-8_12_2022-16_42.pth"
#real Imaginary paths
pth1         = base + "QPSK/Real/OFDM_Eq_SNR_(45_25)_(REAL)_-12_1_2023-11_28.pth"
pth2         = base + "QPSK/Imag/OFDM_Eq_SNR_(45_25)_(IMAG)_-12_1_2023-11_28.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"
#polar strategy
pth_angle  = base + "QPSK/Phase/OFDM_Eq_SNR_(45_25)_(ANGLE)_-11_1_2023-17_17.pth"
pth_abs    = base + "OFDM_Eq_SNR_(40_20)_(ABS)_-9_12_2022-14_5.pth"
#binary Cross entropy
pth_bce    = base + "OFDM_Eq_SNR_(40_20)_(FOUR)_-9_12_2022-11_35.pth"
#complex gradient
pth_complex = base +"QPSK/Complex/OFDM_Eq_SNR_(45_15)_(COMPLEX)_-12_1_2023-13_3.pth"

#TRAINING PATHS
folder_training_model = "QPSK"

#TRAIN PARAMETERS
BEST_SNR   = 45
WORST_SNR  = 15
STEP_SNR   = -5
EPOCHS     = 4
LEARNING_RATE = .001
EPSILON = .01
#for Phase, LR=.0007, EPISLON = .007

##TESTING and GOLDEN
GOLDEN_BEST_SNR  = 45
GOLDEN_WORST_SNR = 5
GOLDEN_STEP      = 2