import os

# import time

import numpy as np
import DaMAT as dmt

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


epsilon = 0.01
method = "ttsvd"
game = "Qbert"
runs = 10
increment = 1
initialize = 1

np.random.seed(31)

dims = 5
rank = 8
dimensions = [20, 35, 10, 4, 67]


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
        # "/home/dorukaks/Desktop/GameDataProject"
        "/home/doruk/gameDataProject"
        + f"/{game}NoFrameskip-v4/training/full_image_all_lives/\
            {game}NoFrameskip-v4-recorded_images-"
    )
else:
    imgDir = (
        # "/home/dorukaks/Desktop/GameDataProject"
        "/home/doruk/gameDataProject"
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
