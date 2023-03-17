GOLDEN_PATH  = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/GoldenModels"
PHASE_PATH   = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet"
POLAR_PATH   = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PolarPreTrained"
COMPLEX_PATH = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/Real_imag"
MOBILE_PATH  = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Autoencoder/MobileNet"
GRID_PATH    = "/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/Transformers/GridNet"


Golden_BER_type_4 = {
    "LMMSE": 'BER_Test_(Golden_4QAM_LMSE)_-10_3_2023-14_18.csv',
    "MSE"  : 'BER_Test_(Golden_4QAM_MSE)_-10_3_2023-14_14.csv',
    "ZERO" : 'BER_Test_(Golden_4QAM_ZERO)_-10_3_2023-14_11.csv',
    "OSIC" : 'BER_Test_(Golden_4QAM_OSIC)_-17_3_2023-0_12.csv',
    "NML"  : 'BER_Test_(Golden_4QAM_NML)_-17_3_2023-0_11.csv'
 }

Golden_BER_type_16 = {
    "LMMSE": 'BER_Test_(Golden_16QAM_LMSE)_-10_3_2023-14_19.csv',
    "MSE"  : 'BER_Test_(Golden_16QAM_MSE)_-10_3_2023-14_19.csv',
    "ZERO" : 'BER_Test_(Golden_16QAM_ZERO)_-10_3_2023-14_19.csv',
    "OSIC" : 'BER_Test_(Golden_16QAM_OSIC)_-16_3_2023-15_39.csv',
    "NML"  : 'BER_Test_(Golden_16QAM_NML)_-16_3_2023-17_21.csv'
}

Golden_BLER_type_4 = {
    "LMMSE": 'BLER_Test_(Golden_4QAM_LMSE)_-10_3_2023-14_18.csv',
    "MSE"  : 'BLER_Test_(Golden_4QAM_MSE)_-10_3_2023-14_14.csv',
    "ZERO" : 'BLER_Test_(Golden_4QAM_ZERO)_-10_3_2023-14_11.csv',
    "OSIC" : 'BLER_Test_(Golden_4QAM_OSIC)_-17_3_2023-0_12.csv',
    "NML"  : 'BLER_Test_(Golden_4QAM_NML)_-17_3_2023-0_11.csv'
}

Golden_BLER_type_16 = {
    "LMMSE": 'BLER_Test_(Golden_16QAM_LMSE)_-10_3_2023-14_19.csv',
    "MSE"  : 'BLER_Test_(Golden_16QAM_MSE)_-10_3_2023-14_19.csv',
    "ZERO" : 'BLER_Test_(Golden_16QAM_ZERO)_-10_3_2023-14_19.csv',
    "OSIC" : 'BLER_Test_(Golden_16QAM_OSIC)_-16_3_2023-15_39.csv',
    "NML"  : 'BLER_Test_(Golden_16QAM_NML)_-16_3_2023-17_21.csv'
}

Golden_dict_BER = {
    4 :Golden_BER_type_4,
    16:Golden_BER_type_16}

Golden_dict_BLER = {
    4 :Golden_BLER_type_4,
    16:Golden_BLER_type_16}

Golden_dict = {
    "BER":Golden_dict_BER,
    "BLER":Golden_dict_BLER
}

def Golden_plot(Btype,QAM,model):
    prefix ="/BER_csv/" if Btype == "BER" else "/BLER_csv/"
    return GOLDEN_PATH+prefix+Golden_dict[Btype][QAM][model]

def Net_plot(Btype,model):
    prefix ="/BER_csv/" if Btype == "BER" else "/BLER_csv/"
    if(model == "PhaseNet"):
        return PHASE_PATH+prefix+"{}_Test_(Golden_4QAM_PhaseNet)_-10_3_2023-16_17.csv".format(Btype)
    elif(model == "PolarNet"):
        return POLAR_PATH+prefix+"{}_Test_(Golden_16QAM_PolarMixed)_-10_3_2023-16_33.csv".format(Btype)
    elif(model == "ComplexNet"):
        return COMPLEX_PATH+prefix+"{}_Test_(Golden_16QAM_ComplexNet)_-10_3_2023-16_49.csv".format(Btype)
    elif(model == "MobileNet"):
        return MOBILE_PATH+prefix+"{}_Test_(Golden_16QAM_ZeroForceMobileNet)_-10_3_2023-16_54.csv".format(Btype)
    elif(model == "GridNet Square"):
        if(Btype == "BER"):
            return GRID_PATH+prefix+"{}_Test_(SquareGrid)_-13_3_2023-17_2.csv".format(Btype)
        else:
            return GRID_PATH+prefix+"{}_Test_(SquareGrid)_-11_3_2023-5_51.csv".format(Btype)

    elif(model == "GridNet Polar"):
        if(Btype == "BER"):
            return GRID_PATH+prefix+"PolarGrid.csv"
        else:
            return GRID_PATH+prefix+"PolarGrid.csv"
        
