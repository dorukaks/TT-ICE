import os
import time

import numpy as np
import DaMAT as dmt

method = "ttsvd"
heuristics = ["skip", "occupancy"]
occThreshold = 1
compMetrFile = "compressionMetrics.txt" # This file is for me 
lines2Print=[]
"""
Don't modify those 3 lines above they are settings to the compression algorithm.
"""

cwd = os.getcwd()
epsilon = 0.05 # This is the relative approximation error threshold
nTrainingRuns = 640 # Modify this number accordingly
stpIdx = 300 # If you downsample before saving data, change this to an appropriate number
step = 10 # If you downsample before saving data, change this to 1
iterStep = 1

dataDir  = "./" # Folder that the 2048x1528x3x300 numpy arrays are stored
                # I suggest you to put all the training runs in one folder
saveDir  = "./" # Folder that you will save the TT-cores
saveName = "trainedCore" # Name of the saved TT-core files
dataName = "run_" # I assumed that you name all the run files "run_<runIdx>"

"""
Pick one of the two loops below and proceed
"""
# OPTION 1
# If you use the following loop, have all the train runs at one place 
for runIdx in os.listdir():
    data=np.load(dataDir+runIdx,mmap_mode="r")
# OPTION 2
# If you use the following loop, name the training runs with consecutive numbers starting from 0
for runIdx in range(nTrainingRuns):
    print(f"Run: {runIdx}")
    data=np.load(dataDir+dataName+f"{runIdx}",mmap_mode="r")
    
    # After you pick one of the loops above, comment the other.
    # The rest should be in the same loop since we are compressing.

    stIdx=0
    if runIdx==0: #I'm checking here if we are at the first run. Please modify this if statement accordingly
        dataSet=dmt.ttObject(
            data[:,:,:,stIdx][:,:,:,None],
            epsilon=epsilon,
            keepData=False,
            samplesAlongLastDimension=True,
            method=method,
        )
        dataSet.changeShape([16,32,32,191,3,1])
        dataSet.ttDecomp()
        lines2Print.append(f"{0}")
        lines2Print.append(f"{dataSet.compressionTime}")
        lines2Print.append(f'{dataSet.compressionRatio}')
        lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2Print.append("\n")

        stIdx=9 #If you end up downsampling the timesteps before saving the data, change this to 1
    else:
        stIdx=0
    
    for iterIdx in range(stIdx,stpIdx,step):
        stTime = time.time()
        streamedTensor = data[:,:,:,iterIdx][:,:,:,None].reshape(dataSet.reshapedShape[:-1] + [-1])
        dataSet.ttICEstar(
            streamedTensor,
            epsilon=epsilon,
            heuristicsToUse=heuristics,
            occupancyThreshold=occThreshold,
        )
        stepTime=time.time()-stTime
        lines2Print.append(f"{iterStep}")
        lines2Print.append(f"{stepTime}")
        lines2Print.append(f'{dataSet.compressionRatio}')
        lines2Print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2Print.append("\n")
        print(f" Run {runIdx} timestep {iterIdx} (overall step: {iterStep}) done in {round(stepTime,4)}s")
        with open(compMetrFile,'a') as txt:
            txt.writelines(' '.join(lines2Print))
        lines2Print=[]
        iterStep += 1
    """
    I'm saving after each simulation here, it will slow down compression time a little bit 
    but it will save us a lot of valuable time if compression fails prematurely for some reason
    """
    dataSet.saveData(saveName,directory=saveDir,justCores=True,outputType="npy")