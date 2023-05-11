import os
import time
import numpy as np
import DaMAT as dmt

from scipy.io import savemat

### This python script creates random tensors for TT-FOA
### benchmark case study. Tensors created with this script
### will be compressed using either "randTensorTTFOATest.py"
### or "randTensorTTICETest.py" scripts

cwd = os.getcwd()
tensorSaveLocation = "/randomTensors/"

### User defined parameters ###
nDims=4 # Number of dimensions of tensor
rep=20 # Number of tensors created
dims=[10,15,20,500] # Dimensions of the tensor, note that last dimension is the temporal index
ranks=[1,4,10,30,1] # Underlying TT-ranks
fileName=f'random4d_'+'_'.join(map(str,ranks[1:-1]))+f'_Tensors'
tensorDict={}

for ten in range(rep):
    ttCores=[]
    ## Create random cores
    for dim in range(nDims):
        ttCores.append(np.random.rand(ranks[dim],dims[dim],ranks[dim+1]))

    ## Contract cores
    xTrue=dmt.utils.coreContraction(ttCores)
    tensorDict[f'ten{ten}']=xTrue
    ## Write 4d tensor into .mat file.
    ## We do this to be able to compare with the TT-FOA repository
savemat(cwd+tensorSaveLocation+f'{fileName}.mat',tensorDict)
print(f"{rep} random tensors saved at location {cwd}{tensorSaveLocation}")


