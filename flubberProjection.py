import os
import time
import glob

import numpy as np
import DaMAT as dmt


cwd=os.getcwd()
dataDir="/home/doruk/incrementalTensorTrain/catgelTestData/"
saveDir="/home/doruk/incrementalTensorTrain/compressedFlubber/"
saveName="flubberCore"
errMetrFile = "flubberProjection001.txt"
lines2Print=[]

method='ttsvd'
epsilon=0.01
heurs=['skip','occupancy']
occThreshold=1

nRuns =1600
nTimesteps = 10

dataSet=dmt.ttObject.loadData(saveDir+"flubberCore.npy",5)
snapshotCtr=0
projectedSnapshots=[]
for runIdx in range(nRuns):
    for timeStep in range(nTimesteps):
        print(f"Simulation {runIdx} timestep {timeStep}")

        streamedSnapshot=np.load(dataDir+f"catgel_testData{snapshotCtr}.cgf")[:,:,:,timeStep][:,:,:,:,None]
        projSnp=dataSet.projectTensor(streamedSnapshot)
        projectedSnapshots.append(projSnp)
        relErr=dataSet.computeRelError(streamedSnapshot).item()
        snapshotNorm=np.linalg.norm(streamedSnapshot)


        # streamedData = vtk_to_numpy(array).reshape(dataSet.reshapedShape[:-1]+[-1])
        # stTime=time.time()
        # dataSet.ttICEstar(streamedData,epsilon=epsilon,heuristicsToUse=heurs,occupancyThreshold=occThreshold)
        # stepTime=time.time()-stTime
        # totTime+=stepTime
        lines2Print.append(f"{runIdx}")
        lines2Print.append(f"{timeStep}")
        lines2Print.append(f"{snapshotCtr}")
        lines2Print.append(f"{relErr}")
        lines2Print.append(f'{snapshotNorm}')
        lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2Print.append("\n")
        with open(saveDir+errMetrFile,'a') as txt:
            txt.writelines(' '.join(lines2Print))
        lines2Print=[]
        snapshotCtr+=1
    # print(f"TT-ranks {dataSet.ttRanks}")
    # print(f"Compression ratio {dataSet.compressionRatio}")
    # print(f"Total elapsed time (so far) {round(totTime,4)}")