import os
import time

import numpy as np
import DaMAT as dmt

cwd = os.getcwd()
epsilon = 0.01
method = "ttsvd"
heuristics = ["skip", "occupancy"]
occThreshold = 1

data = np.load(
    # "/home/doruk/Downloads/hurricane_data/data/hurricane_image_train.npy", mmap_mode="r"
    "/home/doruk/data/hurricane_image_train.npy",
    mmap_mode="r"
    # "/home/doruk/data/hurricane_image_train.npy"
)
data = data.reshape(-1, 10, 128, 257, 6)
# dataSet=dmt.ttObject(data[0,:,:,:,:][None,:],epsilon=epsilon,keepData=False,samplesAlongLastDimension=True,method=method)
dataSet = dmt.ttObject(
    data[0, 0, :, :, :][None, :],
    epsilon=epsilon,
    keepData=False,
    samplesAlongLastDimension=True,
    method=method,
)
# [1, 8, 94, 1280, 265, 1] 116.5493s -> naive
# [1, 8, 94, 1280, 265, 1] 95.1593s -> memmap
# dataSet.computeTranspose([2,3,1,4,0])
dataSet.computeTranspose([1, 2, 3, 0])
dataSet.changeShape([8, 16, 257, 6, 1])  # 257 is a prime number..

dataSet.ttDecomp()
snapshotIdx = 1
for idx1 in range(0, 1):
    for idx2 in range(1, data.shape[1]):
        # for snapshotIdx in range(1,100):
        stTime = time.time()
        streamedTensor = (
            data[idx1, idx2, :, :, :][None, :]
            .transpose(dataSet.indexOrder)
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
        )
        snapshotIdx += 1
#
overallSt = time.time()
# for idx1 in range(1, data.shape[0]):
for idx1 in range(1, 20):
    for idx2 in range(0, data.shape[1]):
        # for snapshotIdx in range(1,100):
        stTime = time.time()
        streamedTensor = (
            data[idx1, idx2, :, :, :][None, :]
            .transpose(dataSet.indexOrder)
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
        )
        snapshotIdx += 1
#
print(dataSet.ttRanks)
print(f"total process took {round(time.time()-overallSt,4)}s")
dataSet.saveData("ttCoresTrainData", outputType="txt")

testData = np.load(
    # "/home/doruk/Downloads/hurricane_data/data/hurricane_image_train.npy", mmap_mode="r"
    # "/home/doruk/data/hurricane_image_test.npy", mmap_mode="r"
    "/home/doruk/data/hurricane_image_test.npy"
)

testData = testData.transpose(dataSet.indexOrder)
testData = testData.reshape(dataSet.reshapedShape[:-1] + [-1])
testMatrix = dataSet.projectTensor(testData)


print(testMatrix.shape)

# np.savetxt(
#     directory + f"{fileName}_{coreIdx}.txt",
#     testMatrix.reshape(-1, core.shape[-1]),
#     header=f"{core.shape[0]} {core.shape[1]} {core.shape[2]}",
#     delimiter=" ",
#     )
