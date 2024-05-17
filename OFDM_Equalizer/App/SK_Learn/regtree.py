from NN_lab import NetLabs
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from Recieved import RX
from math import pi
from  tqdm import tqdm
from config import GOLDEN_BEST_SNR, GOLDEN_WORST_SNR, GOLDEN_STEP
from utils import get_time_string,vector_to_pandas
import pickle

class RegTree(NetLabs):
    def __init__(self,depth,best=45,worst=5,step=-5):
        self.data      = RX()
        #self.regtree   = DecisionTreeRegressor(max_depth=depth)
        self.regtree   = RandomForestClassifier(n_estimators = depth)
        #self.regtree   = QuadraticDiscriminantAnalysis()
        #self.regtree   = GaussianNB()
        self.TLen      = int(self.data.total*.6)
        self.BEST_SNR  = best
        self.WORST_SNR = worst
        self.step      = step
        self.filename = 'tree_model.sav'
    
    def LMSE(self,H,Y,SNR):
        return np.linalg.inv(H.H@H+np.eye(48)*(10**(-SNR/10)))@H.H@Y
    
    def Generate_SNR_ANGLE(self,SNR):
        r = None
        self.data.AWGN(SNR)
        #right side of equalizer
        Entry = np.empty((self.data.sym_no,self.data.total,1))
        for i in range(0,self.data.total):
            Y = self.data.Qsym.r[:,i]
            H = np.matrix(self.data.H[:,:,i])
            X_hat = self.LMSE(H,Y,SNR)
            Entry[:,i] = np.angle(X_hat)/pi
            #Entry[:,i,0]= X_hat.real
            #Entry[:,i,1]= X_hat.imag
            
        r = Entry
        del Entry
        return r
    
    def train(self):
        for SNR in range(self.BEST_SNR,self.WORST_SNR-1,self.step):
            self.r = self.Generate_SNR_ANGLE(SNR)
            #self.data.AWGN(SNR)
            #self.r = np.angle(self.data.Qsym.r)
            loop  = tqdm(range(0,self.TLen),desc="Progress")
            for i in loop:
                self.regtree.fit(self.r[:,i],np.ravel(self.data.Qsym.bits[:,i]))
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
        # save the model to disk
        pickle.dump(self.regtree, open(self.filename, 'wb'))
        
    def test(self):
        frames = self.data.total
        BER    = []
        self.regtree = pickle.load(open(self.filename, 'rb'))
        
        for SNR in range(GOLDEN_BEST_SNR,GOLDEN_WORST_SNR-1,GOLDEN_STEP*-1):
            self.r = self.Generate_SNR_ANGLE(SNR)
            loop   = tqdm(range(0,self.data.total),desc="Progress")
            errors = 0
            for i in loop:
                rxbits = self.regtree.predict(self.r[:,i]).astype(int)
                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
            
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
            
        formating = "Tree_SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,"REGTREE",get_time_string())
        vector_to_pandas("BER_{}.csv".format(formating),BER)
        
        
t = RegTree(16,best=45,worst=5,step=-5)
t.train()
print("TEST")
t.test()