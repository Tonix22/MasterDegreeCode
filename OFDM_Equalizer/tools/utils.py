import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"conf")

from config import Test_PAHT
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP, PLOTS_PATH
import matplotlib.pyplot as plt
from datetime import datetime

MONTH = "January"
    
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
    for BER_it in BER_list:
        df  = pd.read_csv(BER_it)
        BER = df.iloc[:, 1].to_numpy()
        f = interp1d(indexValues, BER, kind='cubic')
        indexValues_interp = np.linspace(GOLDEN_WORST_SNR, GOLDEN_BEST_SNR, 200)
        BER_interp = f(indexValues_interp)
        plt.grid(True, which="both")
        plt.semilogy(indexValues_interp, BER_interp, label=labels[index])
        index += 1

    plt.legend()
    plt.title('SNR and {} '.format(BER_BLER)+title)
    plt.xlabel('SNR')
    plt.ylabel(BER_BLER)
    
    #plt.show()
    plt.savefig('{}/{}/Results_{}_{}.png'.format(PLOTS_PATH,BER_BLER,title,get_time_string()))
           

def vector_to_pandas(name,BER,path=Test_PAHT):
    BER = np.asarray(BER)
    BER = np.flip(BER)
    df = pd.DataFrame(BER)
    if(path == Test_PAHT):
        df.to_csv("{}/{}/{}".format(path,MONTH,name))
    else:
        df.to_csv("{}/{}".format(path,name))