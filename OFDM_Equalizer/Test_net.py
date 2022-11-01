from cmath import pi
from NN_lab import NetLabs
import torch
import torch.nn as nn
import numpy as np
from Constants import *
import pandas as pd
from  tqdm import tqdm
import GPUtil
import matplotlib.pyplot as plot


class TestNet(NetLabs):
    def __init__(self,path = None,pth_real=None,pth_imag=None,loss_type=MSE,best_snr = 60,worst_snr = 5):
        super().__init__(loss_type,best_snr,worst_snr)
        if(loss_type == MSE_COMPLETE):
            self.model = self.Generate_Network_Model()
            self.model.load_state_dict(torch.load(path))
        
        if(loss_type == MSE):
            self.model_real = None
            self.model_imag = None
            self.model_real = self.Generate_Network_Model()
            self.model_imag = self.Generate_Network_Model()
            self.model_real.load_state_dict(torch.load(pth_real))
            self.model_imag.load_state_dict(torch.load(pth_imag))
            
        #ground truth
        if(self.loss_type == CROSSENTROPY or self.loss_type == MSE_COMPLETE):
            self.gt = self.Get_ground_truth(self.data.Qsym.bits)
        else:
            self.gt = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        
    #************************
    #*******TESTING**********
    #************************
    def TestQAM(self):
        self.loss_type=CROSSENTROPY
        
        df = pd.DataFrame()
        BER    = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-5):
            losses = []
            self.r = self.Generate_SNR(SNR,BOTH)
            loop   = tqdm(range(0,self.data.total),desc="Progress")
            frames = self.data.total-0 #self.training_data
            errors = 0
            
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                #Ground truth are constelation bits
                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                Y      = torch.from_numpy(txbits).cuda()
                
                pred = self.model(X[0:self.data.sym_no].float(),X[self.data.sym_no:].float()) 

                loss = self.criterion(pred,Y.float())
                losses.append(loss.cpu().detach().numpy())
                #BER
                rxbits = pred.cpu().detach().numpy()*3
                rxbits = np.around(rxbits)
                rxbits = rxbits.astype(np.uint8)
                errors += ((txbits^rxbits)&1).sum()+((txbits^rxbits)&2>>1).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(loss=torch.mean(loss).cpu().detach().numpy())
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            Ber_DF = errors/((self.data.bitsframe*self.data.sym_no)*frames)
            df["SNR{}".format(SNR)]= Ber_DF
            losses.clear()
            
            #Calculate BER
            BER.append(Ber_DF)
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,BOTH,self.get_time_string())
        df.to_csv('reports/Test_BER_{}.csv'.format(formating), header=True, index=False)
        
        indexValues = np.arange(self.WORST_SNR,self.BEST_SNR,)
        BER = np.asarray(BER)
        plot.grid(True, which =BOTH)
        plot.semilogy(indexValues,BER)
        plot.title('SNR and BER')
        # Give x axis label for the semilogy plot
        plot.xlabel('SNR')
        # Give y axis label for the semilogy plot
        plot.ylabel('BER')
        plot.savefig('plots/Test_BER_{}.png'.format(formating))
        
    
    def Test(self):
        df = pd.DataFrame()
        BER    = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,-2):
            losses = []
            self.r = self.Generate_SNR(SNR,BOTH)
            loop   = tqdm(range(0,self.data.total),desc="Progress")
            errors = 0
            passed = 0
            frames = self.data.total # self.training_data
            for i in loop:
                
                X  = torch.squeeze(self.r[:,i],1)  # input
                Y  = torch.squeeze(self.gt[:,i],1) # ground thruth
                
                pred_real = self.model_real(X[0:self.data.sym_no].float())
                pred_imag = self.model_imag(X[self.data.sym_no:].float())
                                
                loss_real = self.criterion(pred_real,Y[0:self.data.sym_no].float())
                loss_imag = self.criterion(pred_imag,Y[self.data.sym_no:].float())
                
                loss = (loss_real + loss_imag)/2

                losses.append(loss.cpu().detach().numpy())
                
                #BER
                if(self.loss_type == MSE):
                    #demodulate prediction data
                    real   = pred_real.cpu().detach().numpy()
                    imag   = pred_imag.cpu().detach().numpy()
                    res    = real + 1j*imag
                    rxbits = self.data.Qsym.Demod(res)

                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    #print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,BOTH,self.get_time_string())
        df.to_csv('reports/Testing_Loss_{}.csv'.format(formating), header=True, index=False)
        
        indexValues = np.arange(self.WORST_SNR,self.BEST_SNR,2)
        BER = np.asarray(BER)
        BER = np.flip(BER)
        plot.grid(True, which =BOTH)
        plot.semilogy(indexValues,BER)
        plot.title('SNR and BER')
        # Give x axis label for the semilogy plot
        plot.xlabel('SNR')
        # Give y axis label for the semilogy plot
        plot.ylabel('BER')
        plot.savefig('plots/Test_BER_{}.png'.format(formating))
        
        
class TestNet_Angle_Phase(NetLabs):
    def __init__(self,pth_angle,pth_mag,loss_type=MSE,best_snr = 60,worst_snr = 5,step = -1):
        super().__init__(loss_type,best_snr,worst_snr,step=step)
        
        self.model_angle = self.Generate_Network_Model()
        self.model_mag   = self.Generate_Network_Model()
        self.model_angle.load_state_dict(torch.load(pth_angle))
        self.model_mag.load_state_dict(torch.load(pth_mag))
        #angle
        self.real_imag = ANGLE
        self.gt_angle = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        #Mag
        self.real_imag = ABS
        self.gt_abs = self.Get_ground_truth(self.data.Qsym.GroundTruth)
        
    def Test(self):
        df  = pd.DataFrame()
        BER = []
        
        for SNR in range(self.BEST_SNR,self.WORST_SNR,self.step):
            losses = []
            
            self.r_abs   = self.Generate_SNR(SNR,ABS)
            self.r_angle = self.Generate_SNR(SNR,ANGLE)
            
            loop   = tqdm(range(self.training_data,self.data.total),desc="Progress")
            errors = 0
            passed = 0
            frames = self.data.total-self.training_data # self.training_data
            for i in loop:
                
                X_ang  = torch.squeeze(self.r_angle[:,i],1)  # input
                X_abs  = torch.squeeze(self.r_abs[:,i],1) 
                Y_angle  = torch.squeeze(self.gt_angle[:,i],1) # ground thruth
                Y_mag    = torch.squeeze(self.gt_abs[:,i],1)
                
                pred_ang = self.model_angle(X_ang.float())
                pred_abs = self.model_mag(X_abs.float())
                                
                loss_ang = self.criterion(pred_ang,Y_angle.float())
                loss_abs = self.criterion(pred_abs,Y_mag.float())
                
                loss = (loss_ang + loss_abs)/2

                losses.append(loss.cpu().detach().numpy())
                

                pred_ang = pred_ang*pi
                theta    = pred_ang.cpu().detach().numpy()
                radius   = pred_abs.cpu().detach().numpy()
                res      = radius*np.exp(1j*theta)
                rxbits   = self.data.Qsym.Demod(res)

                txbits = np.squeeze(self.data.Qsym.bits[:,i],axis=1)
                errors+=np.unpackbits((txbits^rxbits).view('uint8')).sum()
                                
                #Status bar and monitor  
                if(i % 500 == 0):
                    loop.set_description(f"SNR [{SNR}]")
                    loop.set_postfix(ber=errors/((self.data.bitsframe*self.data.sym_no)*frames))
                    #print(GPUtil.showUtilization())
                    
            #Apend report to data frame
            df["SNR{}".format(SNR)]= losses
            losses.clear()
            
            #Calculate BER
            BER.append(errors/((self.data.bitsframe*self.data.sym_no)*frames))
        
        
        formating = "SNR_({}_{})_({})_{}".format(self.BEST_SNR,self.WORST_SNR,real_imag_str[BOTH],self.get_time_string())
        df.to_csv('reports/Testing_Loss_{}.csv'.format(formating), header=True, index=False)
        
        indexValues = np.arange(self.WORST_SNR,self.BEST_SNR,self.step*-1)
        BER = np.asarray(BER)
        BER = np.flip(BER)
        plot.grid(True, which ="both")
        plot.semilogy(indexValues,BER)
        plot.title('SNR and BER')
        # Give x axis label for the semilogy plot
        plot.xlabel('SNR')
        # Give y axis label for the semilogy plot
        plot.ylabel('BER')
        plot.savefig('plots/Test_BER_{}.png'.format(formating))
                