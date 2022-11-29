import sys
import os
#path = os.getcwd()
# adding Folder_2 to the system path
#sys.path.insert(0, "{}/{}".format(path,"../"))

#path = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/plots"
path = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/plots"
sys.path.insert(0, "{}/{}".format(path,"../"))
from utils import read_plot_pandas

Test_path = "{}/{}".format(path,"../Test")

#base_Test = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/Test/"
base_Test = "/home/tonix/HardDisk/Documents/Maestria/Tesis/MasterDegreeCode/OFDM_Equalizer/Test/"

paths = [base_Test+"Golden_LMSE_BER_SNR-14_11_2022-22_19.csv",
         base_Test+"BER_SNR_(45_5)_(BOTH)_-25_11_2022-22_33.csv",
         base_Test+"BER_SNR_(45_5)_(2)_-25_11_2022-22_15.csv",
         base_Test+"BER_Tree_SNR_(45_5)_(REGTREE)_-26_11_2022-17_4.csv"]

labels = ["LMMSE","NN_Phase","NN_Two_R_I","Reg Tree Help"]

read_plot_pandas(paths,labels)
