# Recomendations
sudo apt install nvidia-cuda-toolkit
pip install complexPyTorch

# Terminal steps 
conda create --name thesis
conda install --yes --file requirements.txt
conda install -c conda-forge pytorch-gpu

# Dependendencies
conda env create -f environment.yml

## Notes

10,000 Samples of LOS and NLOS
Total = 20,000

16,000 Training 80%
4,000  Testing  20%

**Batches** = 16 batches of 1000 samples

Init=50 dB, End = 5, 

SNR from {20 to 3}

## Check last values added in folder linux 
ls -Art | tail -n 1

## Install Kaggle
pip install kaggle
