import sys
import os
#TODO

main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

Golden   = Test_path+"/Golden/QPSK/"
QPSK_res = Test_path+"/January/QPSK/"


paths = [Golden+"Golden_LMSE_BER_SNR-11_1_2023-16_25.csv",
         Golden+"Golden_MSE_BER_SNR-11_1_2023-16_25.csv",
         QPSK_res+"BER_SNR_(45_5)_(BOTH)_-11_1_2023-17_25.csv",
         QPSK_res+"BER_SNR_(45_5)_(REAL_IMAG)_-11_1_2023-20_55.csv"
         ]
labels = ["LMMSE","MSE","NN_Phase","Real_imag"]

read_plot_pandas(paths,labels,"Golden vs NN_Phase vs Real_imag")
