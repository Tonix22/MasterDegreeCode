import sys
import os
#TODO

main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

Month = "Golden/QPSK/"
Test_path = Test_path+"/"+Month

paths = [Test_path+"Golden_LMSE_BER_SNR-11_1_2023-16_25.csv",
         Test_path+"Golden_MSE_BER_SNR-11_1_2023-16_25.csv"
         ]

labels = ["LMMSE","MSE"]

read_plot_pandas(paths,labels,"Golden")
