import os
import time
import cv2
import glob

import numpy as np
import DaMAT as dmt

cwd = os.getcwd()
videoDir = "/home/doruk/minecraftDataset/"
method = "ttsvd"
heuristics = ["skip", "occupancy", "subselect"]
occThreshold = 1
# compMetrFile = "compressionMetrics.txt"  # This file is for me
# lines2Print = []
epsilon = 0.01

videoFiles = glob.glob(videoDir + "*.mp4")
frames = []
print(videoFiles[0])
video = cv2.VideoCapture(videoFiles[0])
success, image = video.read()
newShape = []
newShape.extend(dmt.utils.primes(image.shape[0]))
newShape.extend(dmt.utils.primes(image.shape[1])[::-1])
newShape.append(image.shape[2])
newShape += [-1]
frames.append(image[:, :, :, None])
ctr = 0
while success:
    success, image = video.read()
    try:
        frames.append(image[:, :, :, None])
    except TypeError:
        pass
    ctr += 1
frames = np.concatenate(frames, axis=-1)
dataSet = dmt.ttObject(
    frames, epsilon=epsilon, keepData=False, samplesAlongLastDimension=True, method=method
)
dataSet.changeShape(newShape=newShape)
dataSet.ttDecomp()
totTime = dataSet.compressionTime
print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
print(f"{ctr} frames compressed in {round(dataSet.compressionTime,4)}")
print(f"Elapsed time (so far) {round(totTime,4)}")

for videoFile in videoFiles[1:]:
    print(videoFile)
    frames = []
    video = cv2.VideoCapture(videoFile)
    success, image = video.read()
    frames.append(image[:, :, :, None])
    ctr = 0
    while success:
        success, image = video.read()
        try:
            frames.append(image[:, :, :, None])
        except TypeError:
            pass
        ctr += 1
    frames = np.concatenate(frames, axis=-1)
    stTime = time.time()
    dataSet.ttICEstar(
        frames,
        epsilon=epsilon,
        heuristicsToUse=heuristics,
        occupancyThreshold=occThreshold,
    )
    updTime = time.time() - stTime
    totTime += updTime
    print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
    print(f"{ctr} frames compressed in {round(updTime,4)}")
    print(f"Elapsed time (so far) {round(totTime,4)}")
    2 + 2
