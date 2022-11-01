

base         = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/"
pth_complete = base + "OFDM_Equalizer_both_-28_10_2022-11_20.pth"
pth1         = base + "OFDM_Eq_SNR_(30_25)_(real)_-29_10_2022-14_20.pth"
pth2         = base + "OFDM_Eq_SNR_(30_25)_(imag)_-29_10_2022-14_21.pth"
pth_QAM      = base + "Constelation-26_10_2022-16_49.pth"

pth_angle  = base + "OFDM_Eq_SNR_(25_15)_(ANGLE)_-31_10_2022-23_54.pth"
pth_abs    = base + "OFDM_Eq_SNR_(25_15)_(ABS)_-31_10_2022-23_59.pth"


BEST_SNR   = 25
WORST_SNR  = 15
EPOCHS     = 1
LEARNING_RATE = .0005
EPSILON = .05