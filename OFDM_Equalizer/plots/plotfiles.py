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


#labels = ["LMMSE","OSIC","NML","PolarNet","ComplexNet","MobileNet","GridNet Square","GridNet Polar"]
labels   = [#"NML",
            #"NML+QR+DFT",
            #"OSIC",
            #"OSIC+QR+DFT"
            #"LMSE",
            #"LSME+DFT",
            "MSE",
            "MSE+DFT"
            ]

BasePath = '/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/Test/Golden/16QAM/'

NMLPath  = BasePath+"NML/BER/2024/jun/16/"
OSICPath = BasePath+"OSIC/BER/2024/jun/18/"

LMSEPath = BasePath+"LMSE/BER/2024/jun/18/"
MSEPath  = BasePath+"MSE/BER/2024/jun/18/"

paths = [ #NMLPath+'Complete_2151.csv',
          #NMLPath+'DFT_spreading_2338.csv',
          #OSICPath+'Complete_0940.csv',
          #OSICPath+'DFT_spreading_0958.csv',
          #LMSEPath+'Complete_0908.csv',
          #LMSEPath+'DFT_spreading_0922.csv',
          MSEPath+'Complete_0842.csv',
          MSEPath+'DFT_spreading_0855.csv'
        ]

read_plot_pandas(paths,labels,"MSE",BER_BLER="BER")
