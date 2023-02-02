import numpy as np
import pandas as pd
import sys
import os

main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"conf")

from config import Test_PAHT
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP, PLOTS_PATH
import matplotlib.pyplot as plot
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

def read_plot_pandas(BER_list,labels,title=""):
    indexValues = np.arange(GOLDEN_WORST_SNR,GOLDEN_BEST_SNR+1,GOLDEN_STEP)
    index = 0
    for BER_it in BER_list:
        df  = pd.read_csv(BER_it)
        BER = df.iloc[:, 1].to_numpy()
        plot.grid(True, which ="both")
        plot.semilogy(indexValues,BER, label= labels[index])
        index+=1
    
    plot.legend()  
    plot.title('SNR and BER'+title)
    # Give x axis label for the semilogy plot
    plot.xlabel('SNR')
    # Give y axis label for the semilogy plot
    plot.ylabel('BER')
    
    #plot.show()
    plot.savefig('{}/Results_{}.png'.format(PLOTS_PATH,get_time_string()))
           

def vector_to_pandas(name,BER,path=Test_PAHT):
    BER = np.asarray(BER)
    BER = np.flip(BER)
    df = pd.DataFrame(BER)
    if(path == Test_PAHT):
        df.to_csv("{}/{}/{}".format(path,MONTH,name))
    else:
        df.to_csv("{}/{}".format(path,name))