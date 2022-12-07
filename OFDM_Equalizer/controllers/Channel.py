import scipy.io
import numpy as np
import os
import sys
main_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.insert(0, main_path+"conf")
sys.path.insert(0, main_path+"tools")
from config import MATH_PATH
from DownloadFiles import Download_Mat_files

class Channel():
    #Data Set size of 10,000 for each data set
    def __init__(self,LOS=True):
        #LOS stand for Line of sight
        #This class convert mat file into numpy 
        Mat = None
        import os
        directory_path = MATH_PATH
        if(os.path.exists(directory_path+"/v2v80211p_LOS.mat") == False):
            Download_Mat_files()
            
        if(LOS):
            Mat = scipy.io.loadmat("{}{}".format(directory_path,'v2v80211p_LOS.mat'))
        else:
            Mat = scipy.io.loadmat("{}{}".format(directory_path,'v2v80211p_NLOS.mat'))
        self.con_list = Mat['vectReal32b']
        
    def __getitem__(self, index):
        return self.con_list[:,:,index]
    