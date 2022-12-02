import sys
import os
#TODO

main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

Month = "December/"
Test_path = Test_path+"/"+Month

paths = [Test_path+"BER_SNR_(45_5)_(BOTH)_-2_12_2022-11_46.csv",
         Test_path+"Golden_LMSE_BER_SNR-2_12_2022-11_43.csv"]

labels = ["LMMSE","NN_Phase"]

read_plot_pandas(paths,labels)
