import os
import time
import numpy as np
import DaMAT as dmt

from scipy.io import savemat

## Initialize/set parameters
nDims=4
rep=20
dims=[10,15,20,500]
ranks=[1,4,10,30,1]
fileName=f'random4d_{ranks[1]}_{ranks[2]}_{ranks[3]}_Tensors'
tensorDict={}

for ten in range(rep):
    ttCores=[]
    ## Create random cores
    for dim in range(nDims):
        ttCores.append(np.random.rand(ranks[dim],dims[dim],ranks[dim+1]))

    ## Contract cores
    xTrue=dmt.utils.coreContraction(ttCores)
    tensorDict[f'ten{ten}']=xTrue
    ## Write 4d tensor into .mat file
savemat(f'{fileName}.mat',tensorDict)


