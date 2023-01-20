#%% 
import torch

#%% take symetrically diagonals
def vect_diagonals(matrix,diagonals):
    #Concatenate diagonals into a tensor
    off_diag = torch.cat([torch.diag(matrix, i) for i in range(-diagonals, diagonals+1)])
    return off_diag


# Create a 4D tensor with dimensions
# Batch size = 10, Channels 2, Matrix 3x3
tensor_4d   = torch.randint(1,100,(10, 2, 4, 4))
diagonals   = 1
dummy_diag  = vect_diagonals(tensor_4d[0, 0, :, :],diagonals)
dummy_shape = len(dummy_diag)

proccesed = torch.zeros((10,2,dummy_shape))
#%%
##Extract a matrix at index (0,1,:,:)
for image in range(0,10):
    
    for channel in range(0,2):
        proccesed[image,channel,:] = vect_diagonals(tensor_4d[image, channel, :, :],diagonals)


