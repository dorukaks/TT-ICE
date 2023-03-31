import os
import time
import glob
import numpy as np
import DaMAT as dmt

cwd=os.getcwd()
dataDir="/home/doruk/incrementalTensorTrain/catgelTrainData/"
saveDir="/home/doruk/incrementalTensorTrain/compressedFlubber/"
saveName="flubberCore"
compMetrFile = "flubberMetrics001.txt"
lines2Print=[]

numSims=6400
numTimesteps=10

method='ttsvd'
epsilon=0.01
heurs=['skip','occupancy']
occThreshold=1


snapshot=np.load(dataDir+"catgel_trainData0.cgf")[:,:,:,0][:,:,:,:,None]


dataSet=dmt.ttObject(snapshot,epsilon=epsilon,keepData=False,samplesAlongLastDimension=True,method=method)
dataSet.ttDecomp(dtype=np.float64)
print(f"Snapshot {0} timestep {0} compressed in {round(dataSet.compressionTime,4)}")
print(f"TT-ranks {dataSet.ttRanks}")
lines2Print.append(f"{0}")
lines2Print.append(f"{0}")
lines2Print.append(f"{0}")
lines2Print.append(f"{dataSet.compressionTime}")
lines2Print.append(f'{dataSet.compressionRatio}')
lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
lines2Print.append("\n")
with open(saveDir+compMetrFile,'a') as txt:
    txt.writelines(' '.join(lines2Print))
lines2Print=[]

totTime=dataSet.compressionTime

iterStep=1
for timeStep in range(1,numTimesteps,1):
    streamedSnapshot=np.load(dataDir+"catgel_trainData0.cgf")[:,:,:,timeStep][:,:,:,:,None]
    stTime=time.time()
    dataSet.ttICEstar(streamedSnapshot,epsilon=epsilon,heuristicsToUse=heurs,occupancyThreshold=occThreshold)
    stepTime=time.time()-stTime
    # print(f"Snapshot {0} timestep {timeStep} compressed in {round(stepTime,4)}")
    totTime+=stepTime
    lines2Print.append(f"{0}")
    lines2Print.append(f"{timeStep}")
    lines2Print.append(f"{iterStep}")
    lines2Print.append(f"{stepTime}")
    lines2Print.append(f'{dataSet.compressionRatio}')
    lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
    lines2Print.append("\n")
    with open(saveDir+compMetrFile,'a') as txt:
        txt.writelines(' '.join(lines2Print))
    lines2Print=[]
    iterStep+=1
print(f"TT-ranks {dataSet.ttRanks}")
print(f"Compression ratio {dataSet.compressionRatio}")
print(f"Total elapsed time (so far) {round(totTime,4)}")

for simIdx in range(1,numSims,1):
    print(simIdx)
    for timeStep in range(numTimesteps):
        # if runIdx==151:
        #     print(f"{runIdx}")
        streamedSnapshot=np.load(dataDir+f"catgel_trainData{simIdx}.cgf")[:,:,:,timeStep][:,:,:,:,None]
        stTime=time.time()
        dataSet.ttICEstar(streamedSnapshot,epsilon=epsilon,heuristicsToUse=heurs,occupancyThreshold=occThreshold)
        stepTime=time.time()-stTime
        # print(f"Simulation {simIdx} timestep {timeStep} compressed in {round(stepTime,4)}")
        totTime+=stepTime
        # if dataSet.computeRelError(streamedSnapshot)>epsilon:
        #     print("neler oluyor yau")
        #     2+2
        lines2Print.append(f"{simIdx}")
        lines2Print.append(f"{timeStep}")
        lines2Print.append(f"{iterStep}")
        lines2Print.append(f"{stepTime}")
        lines2Print.append(f'{dataSet.compressionRatio}')
        lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2Print.append("\n")
        with open(saveDir+compMetrFile,'a') as txt:
            txt.writelines(' '.join(lines2Print))
        lines2Print=[]
        iterStep+=1
    print(f"TT-ranks {dataSet.ttRanks}")
    print(f"Compression ratio {dataSet.compressionRatio}")
    print(f"Total elapsed time (so far) {round(totTime,4)}")

dataSet.saveData(saveName,directory=saveDir,justCores=True,outputType="npy")


2+2