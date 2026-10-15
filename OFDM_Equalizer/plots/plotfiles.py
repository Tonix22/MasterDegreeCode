import sys
import os
#TODO
from PlotsPaths import *
main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

def plot_files(labels,Btype,Title,QAM=16):
    paths = []
    for n in labels:
        if n == "LMMSE" or n == "MSE" or n == "ZERO" or n=="OSIC" or n == "NML":
            paths.append(Golden_plot(Btype,QAM,n))
        else:
            paths.append(Net_plot(Btype,n))
            
    read_plot_pandas(paths,labels,Title,BER_BLER=Btype)
    
def get_last_document(folder_path):
    # List all files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        return None  # Return None if the folder is empty or contains no files

    # Find the file with the latest modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

labels = ["LMMSE","MSE","NML","OSIC","GridNet","ComNet","PhaseNet"]

mode ="BER"
constelation = "4QAM"
BasePath = f'/home/tonixscarlet/Documents/MasterDegreeCode/OFDM_Equalizer/Test/Golden/{constelation}/'


LMSEPath = BasePath+f"LMSE/{mode}/2024/dic/10/"
MSEPath  = BasePath+f"MSE/{mode}/2024/dic/10/"
NMLPath  = BasePath+f"NML/{mode}/2025/ene/8/"
OSICPath = BasePath+f"OSIC/{mode}/2024/dic/10/"

paths = [ get_last_document(LMSEPath),
          get_last_document(MSEPath),
          get_last_document(NMLPath),
          get_last_document(OSICPath),
          f'/home/tonixscarlet/Documents/MasterDegreeCode/OFDM_Equalizer/Test/Golden/4QAM/GridNet/{mode}/2026/Mar/17/Complete_1841.csv',
          f"/home/tonixscarlet/Documents/MasterDegreeCode/OFDM_Equalizer/Test/Golden/4QAM/ZeroForcing/{mode}/2025/May/Complete_0956.csv",
          f"/home/tonixscarlet/Documents/MasterDegreeCode/OFDM_Equalizer/Test/Golden/4QAM/PhaseNet/{mode}/2024/dic/10/Complete_1826.csv"  
        ]

#print("Paths:")
#for path in paths:
#    print(path)

read_plot_pandas(paths,labels,"GridNet",BER_BLER=mode)
