import sys
import os
#path = os.getcwd()
# adding Folder_2 to the system path
#sys.path.insert(0, "{}/{}".format(path,"../"))

path = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/plots"
sys.path.insert(0, "{}/{}".format(path,"../"))
from utils import read_plot_pandas

Test_path = "{}/{}".format(path,"../Test")

base_Test = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/Test/"

paths = [base_Test+"Golden_LMSE_BER_SNR-14_11_2022-22_19.csv",
         base_Test+"BER_SNR_(45_5)_(BOTH)_-15_11_2022-23_13.csv",
         base_Test+"BER_SNR_(45_5)_(BOTH)_-15_11_2022-23_34.csv"]

labels = ["LSME","NN_light_MSE","NN_light_HARD"]

read_plot_pandas(paths,labels)
