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

paths = [Test_path+"Golden_LMSE_BER_SNR-8_12_2022-0_7.csv",
         Test_path+"BER_SNR_(45_5)_(BOTH)_-7_12_2022-23_44.csv"]

labels = ["LMMSE","Polar"]

read_plot_pandas(paths,labels)
