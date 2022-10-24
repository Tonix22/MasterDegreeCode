from NN_lab import NetLabs

Lab = NetLabs(loss_type="MSE",best_snr = 50,worst_snr=5)
#Lab.Trainning()
Lab.Testing("/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/models/OFDM_Equalizer-21_10_2022-17_45.pth")
