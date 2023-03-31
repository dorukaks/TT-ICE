import os
import time

import numpy as np
import DaMAT as dmt

from random import sample
from datetime import datetime
from scipy.io import loadmat

rep=20
tenIdx=0
epsilon=1e-8
lines2print=[]
errors=np.zeros((20,500))
alltime=0
print('TT-ICE* on random tensor')
for tenIdx in range(rep):
    print(f'Tensor:{tenIdx}')
    totalTime=0
    # xTrue=loadmat('random4d_2_3_5_Tensors.mat')[f'ten{tenIdx}']
    xTrue=loadmat('random4d_4_10_30_Tensors.mat')[f'ten{tenIdx}']
    dims=list(xTrue.shape)
    dataSet=dmt.ttObject(xTrue[...,0][...,None],epsilon=epsilon,keepData=False,method='ttsvd')

    # print(f'Sample: {0}')
    stTime=time.time()
    dataSet.ttDecomp(dtype=np.float64)
    stepTime=time.time()-stTime
    totalTime+=stepTime
    xRec=dmt.utils.coreContraction(dataSet.ttCores[:-1]+[dataSet.ttCores[-1][:,0,:]])
    err=xTrue[...,0]-xRec
    relErr=np.linalg.norm(err)/np.linalg.norm(xTrue[...,0])
    errors[tenIdx,0]=relErr
    # print(f'Compressed in: {round(stepTime,4)}')

    for sampleIdx in range(1,dims[-1]):
        # print(f'Sample: {sampleIdx}')
        stTime=time.time()
        dataSet.ttICEstar(xTrue[...,sampleIdx][...,None],epsilon=epsilon,heuristicsToUse=['skip'])
        stepTime=time.time()-stTime
        # print(f'Compressed in: {round(stepTime,4)}')
        totalTime+=stepTime
        xRec=dmt.utils.coreContraction(dataSet.ttCores[:-1]+[dataSet.ttCores[-1][:,sampleIdx,:]])
        err=xTrue[...,sampleIdx]-xRec
        relErr=np.linalg.norm(err)/np.linalg.norm(xTrue[...,sampleIdx])
        errors[tenIdx,sampleIdx]=relErr
        lines2print.append(f'{relErr}')
        # print(f'Relative error: {relErr}')
    print(f'Final tt-ranks: {dataSet.ttRanks}')
    # for core in dataSet.ttCores:
    #     print(core.shape)
    print(f'Total time: {totalTime}')
    lines2print.append(f'\n')
    alltime+=totalTime
# with open('./tticePythonErrors.txt','a') as txt:
# 	txt.writelines(' '.join(lines2print))
lines2print=[]
print(alltime)
print(alltime/20)
# np.savetxt("ttice.txt",errors.mean(0))