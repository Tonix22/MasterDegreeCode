import sys
import os
#TODO

main_path  = os.path.dirname(os.path.abspath(__file__))+"/../"
utils_path = main_path+"tools"
Test_path  = main_path+"Test"
sys.path.insert(0, utils_path)

from utils import read_plot_pandas

Month = "Golden/QPSK/"
Test_path = Test_path+"/"+Month

#BEST PhaseNET
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/BER_csv/BER_Test_(Golden_4QAM_PhaseNet)_-24_2_2023-11_6.csv'

#GRID 7
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/GridNet/BER_csv/BER_Test_(Golden_16QAM_GridTransformer)_-23_2_2023-19_14.csv'
#GRID 8 
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/GridNet/BER_csv/BER_Test_(Golden_16QAM_GridTransformer)_-23_2_2023-18_13.csv'
#Best GRID
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/GridNet/BER_csv/BER_Test_(Golden_16QAM_GridTransformer)_-22_2_2023-16_59.csv'

#Polar net 95%
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarNet/BER_csv/BER_Test_(Golden_16QAM_PolarNet)_-22_2_2023-21_21.csv',
#Polar Net 5%
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarNet/BER_csv/BER_Test_(Golden_16QAM_PolarNet)_-23_2_2023-22_52.csv'
#Polar Net no-zcore
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarNet/BER_csv/BER_Test_(Golden_16QAM_PolarNet)_-23_2_2023-23_7.csv'
#Polar net with preprocesing stage


#GOLDEN 4QAM LMMSE
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/GoldenModels/BER_csv/BER_Test_(Golden_4QAM_LMSE)_-6_2_2023-19_2.csv'
#GOLDEN 16QAM LMMSE
#'/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/GoldenModels/BER_csv/BER_Test_(Golden_16QAM_LMSE)_-1_2_2023-20_22.csv'




paths = ['/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/GoldenModels/BER_csv/BER_Test_(Golden_16QAM_LMSE)_-1_2_2023-20_22.csv',
         '/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarNet/models/16QAM/version_80/BER_Test_(Golden_16QAM_PolarNet)_-16_2_2023-12_26.csv',
         '/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarPreTrained/BER_csv/BER_Test_(Golden_16QAM_PolarMixed)_-24_2_2023-15_12.csv']

labels = ["Golden LMMSE","Polar Net Together","Polar Net Separate Train","Grid Net 7"]

read_plot_pandas(paths,labels," 16-QAM Polar Models")
