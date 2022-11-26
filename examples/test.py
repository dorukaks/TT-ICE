import os
import time

import numpy as np
import python.src.DaMAT as dmt

from PIL import Image

# from random import sample
# from datetime import datetime
from functools import partial

# from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def parallelLoader(imageDir, runIdximgIdx):
    return np.array(
        Image.open(f"{imageDir}{runIdximgIdx[0]}/{runIdximgIdx[1]}.png"), dtype="int16"
    )


cwd = os.getcwd()


heuristicsToUse = ["subselect", "skip", "occupancy"]
epsilon = 0.01
method = "ttsvd"
game = "Qbert"
runs = 10
increment = 1
initialize = 1

runIndices = np.arange(runs).tolist()
allRuns = runIndices.copy()

startRuns = []
runPool = []  # to keep track of used runs
for _ in range(initialize):
    run = runIndices.pop(0)
    runPool.append(run)
    startRuns.append(run)
if game == "MsPacman":
    imgDir = (
        "/home/dorukaks/Desktop/GameDataProject"
        + f"/{game}NoFrameskip-v4/training/full_image_all_lives/\
            {game}NoFrameskip-v4-recorded_images-"
    )
else:
    imgDir = (
        "/home/dorukaks/Desktop/GameDataProject"
        + f"/{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-"
    )

indexes = []
numIms = np.zeros(initialize)
for idx, runIdx in enumerate(startRuns):
    os.chdir(imgDir + f"{runIdx}")
    numIms[idx] = int(len(os.listdir()))
    indexes.extend(zip([runIdx] * len(os.listdir()), range(len(os.listdir()))))
numImgs = int(numIms.sum())
imageDimension = np.array(Image.open("0.png")).shape

with ThreadPoolExecutor() as tpe:
    images = list(tpe.map(partial(parallelLoader, imgDir), indexes))
images = np.array(images).transpose(1, 2, 3, 0)
totalNorm = np.linalg.norm(images)

curIms = [0]
prevIms = [images.shape[-1]]


dataSet = dmt.ttObject(images, epsilon=epsilon)
dataSet.changeShape((30, 28, 40, 3, -1))
dataSet.ttDecomp(totalNorm)
# dataSet.originalData=None

# stepError=dataSet.computeRecError(images,curIms[-1],prevIms[-1])
while runIndices:
    # curIms.append(prevIms[-1])
    print(runIndices)
    incIndices = []
    # lines2Print=[]
    for _ in range(increment):
        run = runIndices.pop(0)
        incIndices.append(run)
        runPool.append(run)
    numIms = np.zeros(increment)
    if game == "MsPacman":
        imgDir = (
            "/home/dorukaks/Desktop/GameDataProject"
            + f"/{game}NoFrameskip-v4/training/full_image_all_lives/\
                {game}NoFrameskip-v4-recorded_images-"
        )
    else:
        imgDir = (
            "/home/dorukaks/Desktop/GameDataProject"
            + f"/{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-"
        )
    indexes = []
    for idx, runIdx in enumerate(incIndices):
        os.chdir(imgDir + f"{runIdx}")
        numIms[idx] = int(len(os.listdir()))
        indexes.extend(zip([runIdx] * len(os.listdir()), range(len(os.listdir()))))
    numImgs = int(numIms.sum())
    imageDimension = np.array(Image.open("0.png")).shape
    with ThreadPoolExecutor() as tpe:
        images = list(tpe.map(partial(parallelLoader, imgDir), indexes))
    images = np.array(images).transpose(1, 2, 3, 0)

    prevIms.append(prevIms[-1] + images.shape[-1])

    imagesNorm = np.linalg.norm(
        np.linalg.norm(np.linalg.norm(images, axis=0), axis=0), axis=0
    )
    totalNorm = np.sqrt(totalNorm**2 + np.linalg.norm(imagesNorm) ** 2)
    stTime = time.time()
    # lines2Print.append(f'{incMethod}')
    # lines2Print.append(f'{dataSet.ttEpsilon}')
    # lines2Print.append(f'{numImgs}') #total data
    # lines2Print.append(f'{numImgs}') #used data

    relErrorBeforeUpdate = dataSet.computeRelError(images)
    # imageCount+=relErrorBeforeUpdate.shape[0]
    norm = np.linalg.norm(imagesNorm)
    stTime = time.time()
    # dataSet.ttICE(images,tenNorm=np.linalg.norm(imagesNorm))
    # print(f'ttICE completed in {round(time.time()-stTime,4)}s')
    # dataSet.ttICEstar(images,tenNorm=np.linalg.norm(imagesNorm),heuristicsToUse=heuristicsToUse,elementwiseNorm=imagesNorm)
    dataSet.ttICEstar(
        images, tenNorm=norm, heuristicsToUse=heuristicsToUse, elementwiseNorm=imagesNorm
    )
    # dataSet.ttICEstar(images,tenNorm=norm,heuristicsToUse=heuristicsToUse,elementwiseNorm=imagesNorm,simpleEpsilonUpdate=True)
    print(f"ttICE* completed in {round(time.time()-stTime,4)}s")
    print(dataSet.ttRanks)
