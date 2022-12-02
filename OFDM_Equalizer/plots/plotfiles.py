import sys
import os
#TODO
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
        base_Test+"Testing_Loss_SNR_(45_40)_(BOTH)_-2_12_2022-0_21"]

labels = ["LMMSE","NN_Phase","NN_Two_R_I","Reg Tree Help"]

read_plot_pandas(paths,labels)
