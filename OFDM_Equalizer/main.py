from NN_lab import TrainNet,TestNet
import sys

BEST_SNR   = 35
WORST_SNR = 5



if(sys.argv[1]=="test"):
    print("testing")
    TN = TestNet("/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer-26_10_2022-0_35.pth",
                "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer-26_10_2022-0_41.pth")
    TN.Test()


if(sys.argv[1]=="train"):
    print("Training")
    real_net = TrainNet(real_imag="real",best_snr=BEST_SNR,worst_snr=WORST_SNR)
    real_net.Train(epochs=3)
    imag_net = TrainNet(real_imag="imag",best_snr=BEST_SNR,worst_snr=WORST_SNR)
    imag_net.Train(epochs=3)
    
