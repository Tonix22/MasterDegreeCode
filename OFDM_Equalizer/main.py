from NN_lab import TrainNet,TestNet
import sys
from multiprocessing import Process, Event
import os
import time

TRAIN_MSE_REAL = 0
TRAIN_MSE_IMAG = 1
TEST_MSE       = 2
TRAIN_EQ       = 3
TEST_EQ        = 4
TRAIN_COMPLETE = 5
TEST_COMPLETE  = 6

BEST_SNR   = 30
WORST_SNR  = 15
EPOCHS     = 2

pth_complete = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer_both_-28_10_2022-11_20.pth"
pth1 = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Eq_SNR_(30_25)_(real)_-29_10_2022-14_20.pth"
pth2 = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Eq_SNR_(30_25)_(imag)_-29_10_2022-14_21.pth"
pth_QAM = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/Constelation-26_10_2022-16_49.pth"


def Motor(event):
    
    if(event == TEST_MSE):
        print("testing")
        TN = TestNet(pth_real = pth1,pth_imag = pth2,best_snr=BEST_SNR,worst_snr=WORST_SNR)
        TN.Test()

    if(event == TRAIN_MSE_REAL):
        real_net = TrainNet(real_imag="real",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        real_net.TrainMSE(epochs=EPOCHS)
        
    if(event == TRAIN_MSE_IMAG):   
        imag_net = TrainNet(real_imag="imag",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        imag_net.TrainMSE(epochs=EPOCHS)

        
    if(event == TRAIN_EQ):
        Demod = TrainNet(loss_type="Entropy",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        Demod.TrainQAM(epochs=EPOCHS)
        
    if(event == TEST_EQ):
        TN = TestNet(pth_real = pth1, pth_imag = pth2)
        TN.TestQAM(pth_QAM)
        
    if(event == TRAIN_COMPLETE):
        Complete = TrainNet(real_imag="both",loss_type="MSE_complete",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        Complete.TrainMSE(epochs=EPOCHS)
    if(event == TEST_COMPLETE):
        TN = TestNet(path = pth_complete,loss_type="MSE_complete",best_snr=40,worst_snr=5)
        TN.TestQAM()

if __name__ == '__main__':
    if(sys.argv[1] == "trainMSE_Real"):
        Motor(TRAIN_MSE_REAL)
    if(sys.argv[1] == "trainMSE_Imag"):
        Motor(TRAIN_MSE_IMAG)
    if(sys.argv[1] == "testMSE"):
        Motor(TEST_MSE)
    if(sys.argv[1] == "trainEq"):
        Motor(TRAIN_EQ)
    if(sys.argv[1]=="testEq"):
        Motor(TEST_EQ)
    if(sys.argv[1]=="train_complete"):   
        Motor(TRAIN_COMPLETE)
    if(sys.argv[1]=="test_complete"):   
        Motor(TEST_COMPLETE)