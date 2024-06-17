import matlab.engine
import numpy as np

eng = None

def near_ml(yp,R,conste,index):
    global eng
    if(eng == None):
        eng = matlab.engine.start_matlab()
        eng.cd(r'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/Matlab/Puerto', nargout=0)
    
    return np.array(eng.QRM_Det4b(yp,R,conste,index)).squeeze()
    