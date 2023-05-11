import os
import sys
import time
# sys.path.append('../')

import numpy as np
import DaMAT as dmt

from PIL import Image
from functools import partial
from concurrent.futures import ThreadPoolExecutor

### Function to load image files in parallel
def parallelLoader(imageDir, runIdximgIdx):
    return np.array(
        Image.open(f"{imageDir}{runIdximgIdx[0]}/{runIdximgIdx[1]}.png"), dtype="int16"
    )
### Helper function to write metrics to .txt files
def writer(fileName,fileDir,lines2Print):
    with open(fileDir+fileName,'a') as txt:
        txt.writelines(' '.join(lines2Print))
    lines2Print=[]
    return lines2Print

cwd = os.getcwd()

game = str(sys.argv[1]) # Take name of the atari game as input
# imgLocation = str(sys.argv[2]) # Parent directory that contains the images
# imgLocation = f"/home/doruk/incrementalTensorTrain/"
imgLocation = cwd+f"/data/"

epsilon = 0.10 # Relative error upper bound for compression

# TODO: Implement input checking for games
allowedGames = [
    "BeamRider",
    "Breakout",
    "Enduro",
    "MsPacman",
    "Pong",
    "Qbert",
    "Seaquest",
    "SpaceInvaders",
    ]
if game not in allowedGames:
    raise ValueError(f"Entered game {game} is not in the list of expected games. Please select one from "+", ".join(allowedGames))

# Number of runs used to train the TT-cores.
# We used 40 for Enduro and Pong, and 60 for all other games.
trainRunCounts = {
    "BeamRider" : 60,
    "Breakout" : 60,
    "Enduro" : 40,
    "MsPacman" : 60,
    "Pong" : 40,
    "Qbert" : 60,
    "Seaquest" : 60,
    "SpaceInvaders" : 60,
}
trainRunCount = trainRunCounts[game]
# Feel free to explore by changing the following line.
# trainRunCount = 10

# Number of runs used to test the generalization capability of the TT-cores.
# We used 40 for Enduro and Pong, 160 for MsPacman, and 100 for all other games.
testRunCounts = {
    "BeamRider" : 100,
    "Breakout" : 100,
    "Enduro" : 40,
    "MsPacman" : 160,
    "Pong" : 40,
    "Qbert" : 100,
    "Seaquest" : 100,
    "SpaceInvaders" : 100,
}
testRunCount = testRunCounts[game]
# Feel free to explore by changing the following line.
testRunCount = 20

# Method to compute the initial set of TT-cores.
# Currently only ttsvd is supported
method = "ttsvd" 

initialize = 1 # How many batches of frames we want to increment the TT-cores. Set to 1 in the manuscript.
increment = 1 # How many batches of frames we want to train the initial set of TT-cores. Set to 1 in the manuscript.
batchMultiplier = 5 # Determines the number of batches we will split each compressed batch in testing (to reduce memory) 

printMetrics = True # Boolean flag to print output metrics to terminal
printMetrics2File = True # Boolean flag to print output metrics to a file
saveTrainedCores = False # Boolean flag to save trained TT-cores

metricsCompressedData = True # Boolean flag to compute metrics using all compressed data up to that point
metricsTestData = True # Boolean flag to compute metrics using test data

if saveTrainedCores:
    savedCoreDir = cwd+"/trainedCores/"
    savedCoreName = "Catgel_TTICE_epsilon"+"".join(format(epsilon,'.2f').split("."))+f"_{trainRunCount}train_{testRunCount}test"
    saveType = "npy"

if printMetrics2File:
    metricsFileName = game+"_TTICE_epsilon"+"".join(format(epsilon,'.2f').split("."))+f"_{trainRunCount}train_{testRunCount}test.txt"
    metricsDirectory=cwd+'/textOutputs/'
    if printMetrics:
        print(f"Metrics will be saved under: {metricsDirectory}")
        print(f"Metrics will be saved in file: {metricsFileName}")
    lines2print=[]
    lines2print.append(f"{epsilon}")
    lines2print.append("\n")
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)
    lines2print.append("epsilon")
    lines2print.append("numFrames")
    lines2print.append("usedFrames")
    lines2print.append("errBeforeUpdate")
    lines2print.append("errAfterUpdate")
    lines2print.append("stepTime")
    if metricsCompressedData:
        lines2print.append("cumulRecError")
    if metricsTestData:
        lines2print.append("cumulTestError")
    lines2print.append("compressionRatio")
    lines2print.append("ttRanks")
    lines2print.append("\n")
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)

trainRunIndices = np.arange(trainRunCount).tolist()
testRunIndices = np.arange(trainRunCount,trainRunCount+testRunCount).tolist()

if printMetrics:
    print(f"Numer of training batches: {len(trainRunIndices)}")
    print(f"Numer of test batches    : {len(testRunIndices)}")
    print(f"TT-cores will be initialized with (batches): {(initialize)}")
    print(f"Increment size (batches)                   : {(increment)}")

runs2Compress=[]
compressedRuns=[]

for _ in range(initialize):
    run=trainRunIndices.pop(0)
    runs2Compress.append(run)
    compressedRuns.append(run)
imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'

indexes=[]
numIms = np.zeros(initialize)
for idx,runIdx in enumerate(runs2Compress):
    os.chdir(imgDir+f'{runIdx}')
    numIms[idx]=int(len(os.listdir()))
    indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
numImgs=int(numIms.sum())
imageDimension=np.array(Image.open('0.png')).shape
with ThreadPoolExecutor() as tpe:
    images=list(tpe.map(partial(parallelLoader,imgDir),indexes))
images=np.array(images).transpose(1,2,3,0)
totalNorm=np.linalg.norm(images)

curIms=[0]
prevIms=[images.shape[-1]]

dataSet=dmt.ttObject(images,method=method,epsilon=epsilon)
dataSet.changeShape((30,28,40,3,-1))
dataSet.ttDecomp(totalNorm)
dataSet.originalData=None
stepError=dataSet.computeRecError(images,curIms[-1],prevIms[-1],useExact=True)
previousRanks=dataSet.ttRanks.copy()

if printMetrics2File:
    lines2print.append(f'{dataSet.ttEpsilon}')
    lines2print.append(f'{numImgs}')# total data
    lines2print.append(f'{numImgs}')# used data
    lines2print.append(f'-') # error before ttsvd
    lines2print.append(f'{stepError}') # error after ttsvd 
    lines2print.append(f'{dataSet.compressionTime}')

if printMetrics:
    print(f"Number of images in batch: {numImgs}")
    print(f"Number of images used    : {numImgs}")
    print(f"Approximation error of the batch: {stepError}")
    print(f"Compression ratio               : {dataSet.compressionTime}")

if metricsCompressedData:
    numIms=np.zeros(len(compressedRuns))
    imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
    indexes=[]
    for idx,runIdx in enumerate(compressedRuns):
        os.chdir(imgDir+f'{runIdx}')
        numIms[idx]=int(len(os.listdir()))
        indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
    numImgs=int(numIms.sum())
    indexes=np.array_split(indexes,batchMultiplier*len(compressedRuns))
    recErrors=[]
    ### Create temporary indices to track the image counts in mini batches
    tempImCt=[indexes[idx].shape[0] for idx in range(len(indexes))]
    tempLB=np.cumsum([0]+tempImCt[:-1])
    tempUB=np.cumsum(tempImCt)
    for batchIdx,batch in enumerate(indexes):
        with ThreadPoolExecutor() as tpe:
            images=list(tpe.map(partial(parallelLoader,imgDir),batch))
        images=np.array(images).transpose(1,2,3,0)
        recErrors.append([dataSet.computeRecError(images,tempLB[batchIdx],tempUB[batchIdx],useExact=True),np.linalg.norm(images)])
    totalRecNorm=np.sqrt(np.square(list(zip(*recErrors))[1]).sum())
    totalRecError=np.sqrt(np.square(list(map(np.prod,recErrors))).sum())/totalRecNorm # Total reconstruction error of all compressed data
    if printMetrics:
        print(f'Overall approximation error ({numImgs} images): {round(totalRecError,4)}')
    if printMetrics2File:
        lines2print.append(f'{totalRecError}') # total reconstruction error

if metricsTestData:
    numIms=np.zeros(testRunCount)
    imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
    indexes=[]
    for idx,runIdx in enumerate(testRunIndices):
        os.chdir(imgDir+f'{runIdx}')
        numIms[idx]=int(len(os.listdir()))
        indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
    numImgs=int(numIms.sum())
    indexes=np.array_split(indexes,batchMultiplier*len(testRunIndices))
    relErrors=[]
    for batch in indexes:
        with ThreadPoolExecutor() as tpe:
            images=list(tpe.map(partial(parallelLoader,imgDir),batch))
        images=np.array(images).transpose(1,2,3,0)
        relErrors.append([dataSet.computeRelError(images),np.linalg.norm(images)])
    totalRelNorm=np.sqrt(np.square(list(zip(*relErrors))[1]).sum())
    totalRelError=np.sqrt(np.square(list(map(np.prod,relErrors))).sum())/totalRelNorm # Total projection error of all unseen data
    if printMetrics:
        print(f'Unseen data relative error: {round(totalRelError,4)}')
    if printMetrics2File:
        lines2print.append(f'{totalRelError}') # total projection error

if printMetrics2File:
    lines2print.append(f'{dataSet.compressionRatio}')
    lines2print.append(' '.join(map(str,dataSet.ttRanks)))
    lines2print.append('\n')
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)
if printMetrics:
    print(f"Compression ratio:{dataSet.compressionRatio}")
    print(f"TT-ranks : "+" ".join(map(str,dataSet.ttRanks)))

if saveTrainedCores:
    dataSet.saveData(savedCoreName,savedCoreDir,outputType=saveType)


while trainRunIndices:
    curIms.append(prevIms[-1])
    runs2Compress=[]
    for _ in range(increment):
        run=trainRunIndices.pop(0)
        runs2Compress.append(run)
        compressedRuns.append(run)
    imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
    indexes=[]
    numIms=np.zeros(increment)
    for idx,runIdx in enumerate(runs2Compress):
        os.chdir(imgDir+f'{runIdx}')
        numIms[idx]=int(len(os.listdir()))
        indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
    numImgs=int(numIms.sum())
    with ThreadPoolExecutor() as tpe:
        images=list(tpe.map(partial(parallelLoader,imgDir),indexes))
    images=np.array(images).transpose(1,2,3,0)
    prevIms.append(prevIms[-1]+images.shape[-1])
    imagesNorm=np.linalg.norm(np.linalg.norm(np.linalg.norm(images,axis=0),axis=0),axis=0)
    totalNorm=np.sqrt(totalNorm**2+np.linalg.norm(imagesNorm)**2)
    relErrorBeforeUpdate=dataSet.computeRelError(images,useExact=True)
    elementwiseRelErrorBeforeUpdate=dataSet.computeRelError(images,useExact=False)

    if printMetrics2File:
        lines2print.append(f'{dataSet.ttEpsilon}')
        lines2print.append(f'{numImgs}') 

    iterTime=time.time()
    dataSet.ttICE(
        images,
        epsilon=epsilon
    )
    updTime=time.time()-iterTime

    stepError=dataSet.computeRecError(images,curIms[-1],prevIms[-1],useExact=True)
    if printMetrics:
        print(f"Number of images in batch: {numImgs}")
    if printMetrics2File:
        lines2print.append(f'{numImgs}')
        lines2print.append(f'{relErrorBeforeUpdate}') # error before update 
        lines2print.append(f'{stepError}') # error after ttsvd 
        lines2print.append(f'{updTime}')
    if printMetrics:
        print(f"Number of images used    : {numImgs}")
        print(f'Overall approximation error before update: {round(relErrorBeforeUpdate,4)}')
        print(f"Approximation error of the batch: {stepError}")
        print(f"Compression ratio               : {dataSet.compressionTime}")

    if metricsCompressedData:
        numIms=np.zeros(len(compressedRuns))
        print(compressedRuns)
        imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
        indexes=[]
        for idx,runIdx in enumerate(compressedRuns):
            os.chdir(imgDir+f'{runIdx}')
            numIms[idx]=int(len(os.listdir()))
            indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
        numImgs=int(numIms.sum())
        indexes=np.array_split(indexes,batchMultiplier*len(compressedRuns))
        recErrors=[]
        ### Create temporary indices to track the image counts in mini batches
        tempImCt=[indexes[idx].shape[0] for idx in range(len(indexes))]
        tempLB=np.cumsum([0]+tempImCt[:-1])
        tempUB=np.cumsum(tempImCt)
        for batchIdx,batch in enumerate(indexes):
            with ThreadPoolExecutor() as tpe:
                images=list(tpe.map(partial(parallelLoader,imgDir),batch))
            images=np.array(images).transpose(1,2,3,0)
            recErrors.append([dataSet.computeRecError(images,tempLB[batchIdx],tempUB[batchIdx],useExact=True),np.linalg.norm(images)])
        totalRecNorm=np.sqrt(np.square(list(zip(*recErrors))[1]).sum())
        totalRecError=np.sqrt(np.square(list(map(np.prod,recErrors))).sum())/totalRecNorm # Total reconstruction error of all compressed data
        if printMetrics:
            print(f'Overall relative error ({numImgs} images): {round(totalRecError,4)}')
        if printMetrics2File:
            lines2print.append(f'{totalRecError}') # total reconstruction error   
    
    
    if metricsTestData and dataSet.ttRanks!=previousRanks:
        numIms=np.zeros(testRunCount)
        imgDir=imgLocation+f'{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
        indexes=[]
        for idx,runIdx in enumerate(testRunIndices):
            os.chdir(imgDir+f'{runIdx}')
            numIms[idx]=int(len(os.listdir()))
            indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
        numImgs=int(numIms.sum())
        indexes=np.array_split(indexes,batchMultiplier*len(testRunIndices))
        relErrors=[]
        for batch in indexes:
            with ThreadPoolExecutor() as tpe:
                images=list(tpe.map(partial(parallelLoader,imgDir),batch))
            images=np.array(images).transpose(1,2,3,0)
            relErrors.append([dataSet.computeRelError(images),np.linalg.norm(images)])
        totalRelNorm=np.sqrt(np.square(list(zip(*relErrors))[1]).sum())
        totalRelError=np.sqrt(np.square(list(map(np.prod,relErrors))).sum())/totalRelNorm # Total projection error of all unseen data
        if printMetrics:
            print(f'Unseen data relative error: {round(totalRelError,4)}')
        if printMetrics2File:
            lines2print.append(f'{totalRelError}') # total projection error
    elif dataSet.ttRanks==previousRanks:
        if printMetrics:
            print(f'Update skipped, using the previous unseen error. Unseen data relative error: {round(totalRelError,4)}')
        if printMetrics2File:
            lines2print.append(f'{totalRelError}') # total projection error
    if printMetrics2File:
        lines2print.append(f'{dataSet.compressionRatio}')
        lines2print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2print.append('\n')
        lines2print=writer(metricsFileName,metricsDirectory,lines2print)

    if printMetrics:
        print(f"Compression ratio:{dataSet.compressionRatio}")
        print(f"TT-ranks : "+" ".join(map(str,dataSet.ttRanks)))

    if saveTrainedCores:
        dataSet.saveData(savedCoreName,savedCoreDir,outputType=saveType)

    previousRanks=dataSet.ttRanks.copy()


    
