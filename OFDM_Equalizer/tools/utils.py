import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d
import datetime

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"conf")

from config import Test_PAHT
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP, PLOTS_PATH
import matplotlib.pyplot as plt
from datetime import datetime

MONTH = datetime.now().strftime('%b')
    
def get_time_string():
    current_time = datetime.now()
    day  = current_time.day
    mon  = current_time.month
    year = current_time.year
    hr   = current_time.time().hour
    mn   = current_time.time().minute
    return "-{}_{}_{}-{}_{}".format(day,mon,year,hr,mn)

def read_plot_pandas(BER_list,labels,title="",BER_BLER = 'BER'):
    indexValues = np.arange(GOLDEN_WORST_SNR,GOLDEN_BEST_SNR+1,GOLDEN_STEP)
    index = 0
    width = 2 if len(BER_list) <= 3 else 1
    for BER_it in BER_list:
        df  = pd.read_csv(BER_it)
        BER = df.iloc[:, 1].to_numpy()
        f = interp1d(indexValues, BER, kind='linear')
        indexValues_interp = np.linspace(GOLDEN_WORST_SNR, GOLDEN_BEST_SNR, 21)
        BER_interp = f(indexValues_interp)
        plt.grid(True, which="both")
        plt.semilogy(indexValues_interp, BER, label=labels[index],linewidth=width)
        if(index > 2):
            # Add dashed line
            plt.gca().get_lines()[-1].set_linestyle("--")
            # Add star marker
            if(index == 3):
                plt.gca().get_lines()[-1].set_marker("v")
            elif(index == 4):
                plt.gca().get_lines()[-1].set_marker("h")
            elif(index == 5):
                plt.gca().get_lines()[-1].set_marker("+")
            elif(index == 6):
                plt.gca().get_lines()[-1].set_marker("s")
            else:
                plt.gca().get_lines()[-1].set_marker("*")
            
            plt.gca().get_lines()[-1].set_markersize(4)
            plt.gca().get_lines()[-1].set_markevery(10)  # set marker every 10th data point
        
        index += 1

    plt.legend()
    plt.title('SNR and {} '.format(BER_BLER)+title)
    plt.xlabel('SNR')
    plt.ylabel(BER_BLER)
    
    plt.show()
    #plt.savefig('{}/{}/Results_{}_{}.png'.format(PLOTS_PATH,BER_BLER,title,get_time_string()))
           

def vector_to_pandas(name,BER,path=Test_PAHT):
    BER = np.asarray(BER)
    BER = np.flip(BER)
    df = pd.DataFrame(BER)
    
    if(path == Test_PAHT):
        path = "{}/{}".format(path,MONTH)
    
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            
        df.to_csv(path+"/"+name)
    else:
        df.to_csv("{}/{}".format(path,name))