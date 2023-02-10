import torch
checkpoint_file = torch.load('/home/tonix/Documents/MasterDegreeCode/OFDM_Equalizer/App/NeuronalNet/PhaseNet/lightning_logs/version_5/checkpoints/epoch=79-step=96000.ckpt')
print(checkpoint_file.keys())
torch.save(checkpoint_file['state_dict'],"PhaseNet.pth")
#model.load_state_dict(checkpoint_file['state_dict'])