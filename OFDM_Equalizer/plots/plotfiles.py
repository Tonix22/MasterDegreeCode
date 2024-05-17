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
labels = ["MSE","LMMSE","OSIC","NML"]
plot_files(labels,"BER","Non_linear_Golden vs GridNetSquare")

