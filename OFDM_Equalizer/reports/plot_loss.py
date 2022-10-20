# import pandas module 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
    
# making dataframe 
"""
df = pd.read_csv("Train_Loss_SNR_-19_10_2022-22_18.csv") 
x = np.arange(5,60,1)
plot.stem(x,np.flip(df.iloc[15000].values))
plot.savefig('Trainning_SNR_60_5_row.png')
"""
#Testing

df = pd.read_csv("Train_Loss_SNR-19_10_2022-23_1.csv")
plot.plot(df.iloc[:,50])
plot.savefig('Testing_SNR_10_Col.png')
