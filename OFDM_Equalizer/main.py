from NN_lab import NetLabs
import sys

Lab = NetLabs(loss_type="MSE",best_snr = 35,worst_snr=5)

if(sys.argv[1]=="test"):
    print("testing")
    Lab.Testing("/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer-25_10_2022-17_12.pth")
if(sys.argv[1]=="train"):
    Lab.Trainning()
    print("Training")