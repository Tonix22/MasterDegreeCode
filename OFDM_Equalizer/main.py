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

BEST_SNR   = 35
WORST_SNR  = 5
EPOCHS     = 5

pth1 = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer_imag_-27_10_2022-12_54.pth"
pth2 = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer_real_-27_10_2022-12_54.pth"
pth_QAM = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/Constelation-26_10_2022-16_49.pth"

def Motor(event):
    
    if(event == TEST_MSE):
        print("testing")
        TN = TestNet(pth1,pth2)
        TN.Test()

    if(event == TRAIN_MSE_REAL):
        real_net = TrainNet(real_imag="real",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        real_net.TrainMSE(epochs=EPOCHS)
        
    if(event == TRAIN_MSE_IMAG):   
        imag_net = TrainNet(real_imag="imag",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        imag_net.TrainMSE(epochs=EPOCHS)

        
    if(event == TRAIN_EQ):
        Demod = TrainNet(loss_type="Entropy",best_snr=BEST_SNR,worst_snr=WORST_SNR)
        Demod.TrainQAM(pth1,pth2,epochs=EPOCHS)
        
    if(event == TEST_EQ):
        TN = TestNet(pth1,pth2)
        TN.TestQAM(pth_QAM)

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