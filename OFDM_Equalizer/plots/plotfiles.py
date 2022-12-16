import sys
import os
#TODO

main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

Month = "December/16QAM/"
Test_path = Test_path+"/"+Month

paths = [Test_path+"Golden_LMSE_BER_SNR-8_12_2022-13_9.csv",
         Test_path+"BER_SNR_(45_5)_(COMPLETE)_-9_12_2022-13_36.csv",
         Test_path+"BER_SNR_(45_5)_(FOUR)_-9_12_2022-12_8.csv",
         Test_path+"BER_SNR_(45_5)_(POLAR)_-9_12_2022-14_12.csv",
         Test_path+"BER_SNR_(45_5)_(REAL_IMAG)_-9_12_2022-12_26.csv",
         ]

labels = ["LMMSE","L2_distance","Symbol","Polar","Real_imag"]

read_plot_pandas(paths,labels," 16-QAM")
