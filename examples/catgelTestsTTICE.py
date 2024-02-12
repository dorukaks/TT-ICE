import os
import sys
import time

import numpy as np
import DaMAT as dmt

def writer(fileName,fileDir,lines2Print):
    with open(fileDir+fileName,'a') as txt:
        txt.writelines(' '.join(lines2Print))
    lines2Print=[]
    return lines2Print

cwd = os.getcwd()

method = "ttsvd"
epsilon = 0.1
initialize = 1 # How many batches of frames we want to increment the TT-cores. Set to 1 in the manuscript.
increment = 1 # How many batches of frames we want to train the initial set of TT-cores. Set to 1 in the manuscript.
numberSimulations = 6400
simulationDir = cwd+"/data/catgelTrainData/"
simulationFileName = "catgel_trainData"

printMetrics = True # Boolean flag to print output metrics to terminal
saveTrainedCores = False # Boolean flag to save trained TT-cores
printMetrics2File = True # Boolean flag to print output metrics to a file
metricsCompressedData = True # Boolean flag to compute metrics using all compressed data up to that point

if saveTrainedCores:
    savedCoreDir = cwd+"/trainedCores/"
    savedCoreName = "Catgel_TTICE_epsilon"+"".join(format(epsilon,'.2f').split("."))+f"_{numberSimulations}train"
    saveType = "npy"

if printMetrics2File:
    metricsFileName = "Catgel_TTICE_epsilon"+"".join(format(epsilon,'.2f').split("."))+f"_{numberSimulations}train.txt"
    metricsDirectory=cwd+'/textOutputs/'
    if printMetrics:
        print(f"Metrics will be saved under: {metricsDirectory}")
        print(f"Metrics will be saved in file: {metricsFileName}")
    lines2print=[]
    lines2print.append(f"{epsilon}")
    lines2print.append("\n")
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)
    lines2print.append("startIdx")
    lines2print.append("endIdx")
    lines2print.append("stepTime")
    lines2print.append("stepError")
    if metricsCompressedData:
        lines2print.append("cumulRecError")
    lines2print.append("compressionRatio")
    lines2print.append("ttRanks")
    lines2print.append("\n")
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)


snapshots=[]
compressedSnapshots=[]
for snapshotIdx in range(initialize):
    snapshot = np.load(simulationDir+simulationFileName+f"{snapshotIdx}.cgf")[...,None]
    print(snapshot.shape)
    snapshots.append(snapshot)
    compressedSnapshots.append(snapshotIdx)
snapshots=np.concatenate(snapshots,axis=-1)
print(snapshots.shape)
print(compressedSnapshots)
dataSet = dmt.ttObject(snapshots,epsilon=epsilon,method=method)
dataSet.ttDecomp()
stepError=dataSet.computeRelError(snapshots)

if printMetrics:
    print(f"Numer of training batches: {numberSimulations}")
    print(f"TT-cores will be initialized with (batches): {(initialize)}")
    print(f"Increment size (batches)                   : {(increment)}")

if printMetrics2File:
    lines2print.append(f'{0}')
    lines2print.append(f'{increment-1}')
    lines2print.append(f'{dataSet.compressionTime}')
    lines2print.append(f'{stepError}')
    

if metricsCompressedData:
    recErrors=[]
    for compressedIdx in compressedSnapshots:
        snapshot = np.load(simulationDir+simulationFileName+f"{compressedIdx}.cgf")[...,None]
        recErrors.append([dataSet.computeRecError(snapshot,compressedIdx,useExact=True),np.linalg.norm(snapshot)])
    totalRecNorm=np.sqrt(np.square(list(zip(*recErrors))[1]).sum())
    totalRecError=np.sqrt(np.square(list(map(np.prod,recErrors))).sum())/totalRecNorm # Total reconstruction error of all compressed data
    if printMetrics2File:
        lines2print.append(f'{totalRecError}')

if printMetrics2File:
    lines2print.append(f'{dataSet.compressionRatio}')
    lines2print.append(' '.join(map(str,dataSet.ttRanks)))
    lines2print.append('\n')
    lines2print=writer(metricsFileName,metricsDirectory,lines2print)

if printMetrics:
    print(f"Indices of the first-last simulation: {0}, {initialize}")
    print(f"Approximation error of the batch: {stepError}")
    print(f"Compression time                : {dataSet.compressionTime}")
    print(f'Overall approximation error ({compressedIdx} images): {round(totalRecError,4)}')
    print(f"Compression ratio: {dataSet.compressionRatio}")
    print(f"TT-ranks : "+" ".join(map(str,dataSet.ttRanks)))


if saveTrainedCores:
    dataSet.saveData(savedCoreName,savedCoreDir,outputType=saveType)


for snapshotIdx in range(initialize,numberSimulations,increment):
    snapshots=[]
    if snapshotIdx+increment<numberSimulations:
        for batchIdx in range(snapshotIdx,snapshotIdx+increment):
            snapshot = np.load(simulationDir+simulationFileName+f"{batchIdx}.cgf")[...,None]
            snapshots.append(snapshot)
            compressedSnapshots.append(batchIdx)
    else:
        for batchIdx in range(snapshotIdx,numberSimulations):
            snapshot = np.load(simulationDir+simulationFileName+f"{batchIdx}.cgf")[...,None]
            snapshots.append(snapshot)
            compressedSnapshots.append(batchIdx)
    snapshots=np.concatenate(snapshots,axis=-1)
    snapshotsNorm=np.linalg.norm(np.linalg.norm(np.linalg.norm(np.linalg.norm(np.linalg.norm(snapshots,axis=0),axis=0),axis=0),axis=0),axis=0)
    elementwiseRelErrorBeforeUpdate=dataSet.computeRelError(snapshots,useExact=False)
    iterTime=time.time()
    dataSet.ttICE(
        snapshots,
        epsilon=epsilon,
        )
    updTime=time.time()-iterTime
    stepError=dataSet.computeRelError(snapshots)
    if printMetrics2File:
        lines2print.append(f'{snapshotIdx}')
        lines2print.append(f'{batchIdx}')
        lines2print.append(f'{updTime}')
        lines2print.append(f'{stepError}')

    if printMetrics:
        print(f"Indices of the first-last simulation: {snapshotIdx}, {batchIdx}")
        print(f"Approximation error of the batch: {stepError}")
        print(f"Compression time                : {updTime}")

    if metricsCompressedData:
        recErrors=[]
        for compressedIdx in compressedSnapshots:
            snapshot = np.load(simulationDir+simulationFileName+f"{compressedIdx}.cgf")[...,None]
            recErrors.append([dataSet.computeRecError(snapshot,compressedIdx,useExact=True),np.linalg.norm(snapshot)])
        totalRecNorm=np.sqrt(np.square(list(zip(*recErrors))[1]).sum())
        totalRecError=np.sqrt(np.square(list(map(np.prod,recErrors))).sum())/totalRecNorm # Total reconstruction error of all compressed data
        if printMetrics2File:
            lines2print.append(f'{totalRecError}')

    if printMetrics2File:
        lines2print.append(f'{dataSet.compressionRatio}')
        lines2print.append(' '.join(map(str,dataSet.ttRanks)))
        lines2print.append('\n')
        lines2print=writer(metricsFileName,metricsDirectory,lines2print)

    if printMetrics:
        print(f'Overall approximation error ({compressedIdx} images): {round(totalRecError,4)}')
        print(f"Compression ratio: {dataSet.compressionRatio}")
        print(f"TT-ranks : "+" ".join(map(str,dataSet.ttRanks)))

    if saveTrainedCores:
        dataSet.saveData(savedCoreName,savedCoreDir,outputType=saveType)





