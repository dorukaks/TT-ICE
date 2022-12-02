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


data = np.load(
    # "/home/doruk/Downloads/hurricane_data/data/hurricane_image_train.npy", mmap_mode="r"
    # "/home/doruk/data/run_0.npy",
    dataDir+"testArr0.npy",
    mmap_mode="r"
    # "/home/doruk/data/hurricane_image_train.npy"
)


# anan=data[:,:,:,0][:,:,:,None]
# data = data.reshape(-1, 10, 128, 257, 6)
# data = data.reshape(128,128,191,3,-1)
# dataSet=dmt.ttObject(data[0,:,:,:,:][None,:],epsilon=epsilon,keepData=False,samplesAlongLastDimension=True,method=method)
dataSet = dmt.ttObject(
    # data[0, 0, :, :, :][None, :],
    data[:,:,:,0][:,:,:,None],
    # anan,
    epsilon=epsilon,
    keepData=False,
    samplesAlongLastDimension=True,
    method=method,
)
# [1, 8, 94, 1280, 265, 1] 116.5493s -> naive
# [1, 8, 94, 1280, 265, 1] 95.1593s -> memmap
# dataSet.computeTranspose([2,3,1,4,0])
# dataSet.computeTranspose([1, 2, 3, 0])
# dataSet.computeTranspose([1, 0, 2, 3])
# dataSet.changeShape([191,8,4,16,32,3,1])  # 257 is a prime number..

# dataSet.changeShape([8, 16, 257, 6, 1])  # 257 is a prime number..
# dataSet.changeShape([128,128,191,3,1])  # 257 is a prime number..
dataSet.changeShape([16,32,32,191,3,1])  # 257 is a prime number..
# dataSet.changeShape([4,16,32,191,8,3,1])  # 257 is a prime number..

overallSt = time.time()
dataSet.ttDecomp()
snapshotIdx = 1
for idx1 in range(0, 1):
    for idx2 in range(1, data.shape[-1]):
    # for idx2 in range(1, data.shape[-1]//batchSize):
        # deneme=data[:,:,:,idx2*batchSize:(idx2+1)*batchSize]#[:,:,:,None]
        # for snapshotIdx in range(1,100):
        stTime = time.time()
        streamedTensor = (
            # deneme
            data[:,:,:,idx2][:,:,:,None]
            # data[idx1, idx2, :, :, :][None, :]
            # .transpose(dataSet.indexOrder)
            .reshape(dataSet.reshapedShape[:-1] + [-1])
        )
        dataSet.ttICEstar(
            streamedTensor,
            epsilon=epsilon,
            heuristicsToUse=heuristics,
            occupancyThreshold=occThreshold,
        )
        print(
            f" Update {snapshotIdx} ({idx1,idx2}) done in {round(time.time()-stTime,4)}s"
            # f" Update {snapshotIdx} {idx2*batchSize,(idx2+1)*batchSize} done in {round(time.time()-stTime,4)}s"
        )
        print(dataSet.compressionRatio)
        print(dataSet.ttRanks)
        snapshotIdx += 1
#
# for idx1 in range(1, data.shape[0]):
# for idx1 in range(1, 20):
#     for idx2 in range(0, data.shape[1]):
#         # for snapshotIdx in range(1,100):
#         stTime = time.time()
#         streamedTensor = (
#             data[idx1, idx2, :, :, :][None, :]
#             .transpose(dataSet.indexOrder)
#             .reshape(dataSet.reshapedShape[:-1] + [-1])
#         )
#         dataSet.ttICEstar(
#             streamedTensor,
#             epsilon=epsilon,
#             heuristicsToUse=heuristics,
#             occupancyThreshold=occThreshold,
#         )
#         print(
#             f" Update {snapshotIdx} ({idx1,idx2}) done in {round(time.time()-stTime,4)}s"
#         )
#         snapshotIdx += 1
#
print(dataSet.ttRanks)
print(f"total process took {round(time.time()-overallSt,4)}s")
dataSet.saveData("ttCoresTrainData", outputType="txt")
dataSet.saveData("ttCoresTrainData", outputType="ttc")

# testData = np.load(
#     # "/home/doruk/Downloads/hurricane_data/data/hurricane_image_train.npy", mmap_mode="r"
#     # "/home/doruk/data/hurricane_image_test.npy", mmap_mode="r"
#     "/home/doruk/data/hurricane_image_test.npy"
# )

# testData = testData.transpose(dataSet.indexOrder)
# testData = testData.reshape(dataSet.reshapedShape[:-1] + [-1])
# testMatrix = dataSet.projectTensor(testData)


# print(testMatrix.shape)

# np.savetxt(
#     directory + f"{fileName}_{coreIdx}.txt",
#     testMatrix.reshape(-1, core.shape[-1]),
#     header=f"{core.shape[0]} {core.shape[1]} {core.shape[2]}",
#     delimiter=" ",
#     )
