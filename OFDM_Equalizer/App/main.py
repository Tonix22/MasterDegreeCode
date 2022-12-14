import os 
import sys

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/NeuronalNet")
sys.path.insert(0, main_path+"/../conf")

from Test_net import TestNet, TestNet_Angle_Phase,TestNet_BCE,TestNet_COMPLEX
from Train_Net import TrainNet
from Constants import *
from config import *

def Motor(event):
    #Train
    #Real
    if(event == TRAIN_MSE_REAL):
        real_net = TrainNet(real_imag=REAL,best_snr=BEST_SNR,worst_snr=WORST_SNR,toggle=True,step=STEP_SNR)
        real_net.TrainMSE(epochs=EPOCHS)
    #Imag
    if(event == TRAIN_MSE_IMAG):   
        imag_net = TrainNet(real_imag=IMAG,best_snr=BEST_SNR,worst_snr=WORST_SNR,toggle=True,step=STEP_SNR)
        imag_net.TrainMSE(epochs=EPOCHS)
    #Complete
    if(event == TRAIN_COMPLETE):
        Complete = TrainNet(real_imag=BOTH,loss_type=MSE_COMPLETE,best_snr=BEST_SNR,worst_snr=WORST_SNR,step=STEP_SNR)
        Complete.TrainMSE(epochs=EPOCHS)
    #Magnitud
    if(event == TRAIN_MSE_MAG):
        magnitud = TrainNet(real_imag=ABS,loss_type=MSE,best_snr=BEST_SNR,worst_snr=WORST_SNR,step=STEP_SNR)
        magnitud.TrainMSE(epochs=EPOCHS)
    #Phase
    if(event == TRAIN_MSE_PHASE):
        phase = TrainNet(real_imag=ANGLE,loss_type=MSE,best_snr=BEST_SNR,worst_snr=WORST_SNR,toggle=False,step=STEP_SNR)
        phase.TrainMSE(epochs=EPOCHS)
    #Inv
    if(event == TRAIN_MSE_INV):
        inv = TrainNet(real_imag=INV,loss_type=MSE_INV,best_snr=BEST_SNR,worst_snr=WORST_SNR,step=STEP_SNR)
        inv.TraiINV(epochs=EPOCHS)
    
    if(event == TRAIN_BCE):
        bce = TrainNet(real_imag=FOUR,loss_type=BCE,best_snr=BEST_SNR,worst_snr=WORST_SNR,step=STEP_SNR)
        bce.TrainBCE(epochs=EPOCHS)
    
    if(event == TRAIN_COMPLEX):
        complex_net = TrainNet(real_imag=COMPLEX,loss_type=MSE,best_snr=BEST_SNR,worst_snr=WORST_SNR,step=STEP_SNR,toggle=False)
        complex_net.TrainMSE(epochs=EPOCHS)
    
    #TEST   ***********************************  
    if(event == TEST_MSE):
        print("testing")
        TN = TestNet(pth_real = pth1,pth_imag = pth2)
        TN.Test()

    if(event == TEST_COMPLETE):
        TN = TestNet(path = pth_complete,loss_type=MSE_COMPLETE)
        TN.Test()
        
    #Magnitud Phase
    if(event == TEST_POLAR):
        TN = TestNet_Angle_Phase(pth_angle,pth_abs)
        TN.Test()
        
    if(event == TEST_BCE):
        TN = TestNet_BCE(pth_bce)
        TN.Test()
    if(event == TEST_COMPLEX):
        TN = TestNet_COMPLEX(pth_complex)
        TN.Test()
        

if __name__ == '__main__':
    #***** TRAIN ******
    #COMPLETE
    if(sys.argv[1] == "train_real"):
        Motor(TRAIN_MSE_REAL)
    if(sys.argv[1] == "train_imag"):
        Motor(TRAIN_MSE_IMAG)
        
    #MAG PHASE
    if(sys.argv[1] == "train_mag"):
        Motor(TRAIN_MSE_MAG)
    if(sys.argv[1] == "train_phase"):
        Motor(TRAIN_MSE_PHASE)
    
    #Real and Imag concatenated and QAMdemod
    if(sys.argv[1]=="train_complete"):   
        Motor(TRAIN_COMPLETE)
    #PSEUDOINVERSE
    if(sys.argv[1] == "train_inv"):
        Motor(TRAIN_MSE_INV)
    if(sys.argv[1] == "train_bce"):
        Motor(TRAIN_BCE)
    if(sys.argv[1] == "train_complex"):
        Motor(TRAIN_COMPLEX)
    
    #***** TEST ******
    
    #Test Mag and Phase together
    if(sys.argv[1] == "test_polar"):
        Motor(TEST_POLAR)
        
    #Test Real and Imaginary together
    if(sys.argv[1] == "test_real_imag"):
        Motor(TEST_MSE)
    
    #Test equalizer with Real and Imag concatenated
    if(sys.argv[1]=="test_complete"):   
        Motor(TEST_COMPLETE)
    
    #Test Pseudoinverse
    if(sys.argv[1]=="test_inv"):   
        Motor(TEST_MSE_INV)
        
    if(sys.argv[1] == "test_bce"):
        Motor(TEST_BCE)
        
    if(sys.argv[1] == "test_complex"):
        Motor(TEST_COMPLEX)