import os
import time
import glob

import cv2 as cv
import numpy as np
import DaMAT as dmt
import cupy as cp

cwd = os.getcwd()
videoDir = "/home/doruk/minecraftDataset/"
videoDir = "/media/dorukaks/Database/minecraftData/"
saveDir = "./"
saveDir = "/media/dorukaks/Database/minecraftCores/"
method = "ttsvd"
heuristics = ["occupancy", "skip"]
occThreshold = 1
lines2Print = []
epsilon = 0.01
spatialDS = 1
tempDS = 1
saveName = (
    "minecraftOFMetricsCPE"
    # "minecraftOFCoresE"
    + "".join(str(epsilon).split("."))
    + f"sDS{spatialDS}tDS{tempDS}_f32"
)
compMetrFile = (
    "minecraftOFMetricsCPE"
    # "minecraftOFMetricsE"
    + "".join(str(epsilon).split("."))
    + f"sDS{spatialDS}tDS{tempDS}_f32.txt"
)  # This file is for me

newShape = [12, 30, 40, 16, 2, -1]


videoFiles = glob.glob(videoDir + "*.mp4")
frames = []
print(videoFiles[0])
video = cv.VideoCapture(videoFiles[0])

ret, frame = video.read()
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

ret, frame = video.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.75, 3, 120, 1, 5, 1.2, 0)

# dataSet = dmt.ttObject(
#     flow, epsilon=epsilon, keepData=False, samplesAlongLastDimension=True, method=method
# )
dataSet = dmt.ttObjectCP(
    flow, epsilon=epsilon, keepData=False, samplesAlongLastDimension=True, method=method
)

dataSet.changeShape(newShape=newShape)
dataSet.ttDecomp(dtype=cp.float32)
# dataSet.ttDecomp(dtype=np.float32)
# dataSet.ttDecomp()
videoFrameCtr = 0
frameCtr = 2
videoCtr = 0
lines2Print.append(" ".join(map(str, newShape)))
lines2Print.append("\n")
lines2Print.append(f"{videoCtr}")
lines2Print.append(f"{frameCtr}")
lines2Print.append(f"{dataSet.compressionTime}")
lines2Print.append(f"{dataSet.compressionRatio}")
lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
lines2Print.append("\n")
frameCtr = 1
# for core in dataSet.ttCores:
#     print(type(core))
with open(compMetrFile, "a") as txt:
    txt.writelines(" ".join(lines2Print))
lines2Print = []
totTime = dataSet.compressionTime
# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
# mask = np.zeros_like(frame)

# Sets image saturation to maximum
# mask[..., 1] = 255
prev_gray = gray
# while(video.isOpened()):
while ret:
    # print(frameCtr)
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = video.read()
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except Exception:
        continue
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.75, 3, 120, 1, 5, 1.2, 0)
    flow = flow.reshape(newShape)
    stTime = time.time()
    dataSet.ttICEstar(
        # dataSet.ttICEstarCuPy(
        flow,
        epsilon=epsilon,
        heuristicsToUse=heuristics,
        occupancyThreshold=occThreshold,
    )
    stepTime = time.time() - stTime
    totTime += stepTime
    lines2Print.append(f"{videoCtr}")
    lines2Print.append(f"{frameCtr}")
    lines2Print.append(f"{stepTime}")
    lines2Print.append(f"{dataSet.compressionRatio}")
    lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
    lines2Print.append("\n")
    with open(compMetrFile, "a") as txt:
        txt.writelines(" ".join(lines2Print))
    lines2Print = []
    frameCtr += 1
    prev_gray = gray
videoFrameCtr += frameCtr
# frameCtr += ctr
print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
print(f"{videoFrameCtr} frames compressed in {round(totTime,4)}")
print(f"Elapsed time (so far) {round(totTime,4)}")
dataSet.saveData(saveName, saveDir, justCores=True, outputType="npy")
for videoFile in videoFiles[1:]:
    frameCtr = 0
    videoCtr += 1
    print(videoFile)
    video = cv.VideoCapture(videoFile)
    ret, frame = video.read()
    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.75, 3, 120, 1, 5, 1.2, 0)
    flow = flow.reshape(newShape)
    stTime = time.time()
    dataSet.ttICEstar(
        # dataSet.ttICEstarCuPy(
        flow,
        epsilon=epsilon,
        heuristicsToUse=heuristics,
        occupancyThreshold=occThreshold,
    )
    stepTime = time.time() - stTime
    totTime += stepTime
    lines2Print.append(f"{videoCtr}")
    lines2Print.append(f"{frameCtr}")
    lines2Print.append(f"{stepTime}")
    lines2Print.append(f"{dataSet.compressionRatio}")
    lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
    lines2Print.append("\n")
    with open(compMetrFile, "a") as txt:
        txt.writelines(" ".join(lines2Print))
    lines2Print = []
    frameCtr += 1
    prev_gray = gray
    while ret:
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = video.read()
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        except Exception:
            continue
        flow = cv.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.75, 3, 120, 1, 5, 1.2, 0
        )
        flow = flow.reshape(newShape)
        stTime = time.time()
        dataSet.ttICEstar(
            # dataSet.ttICEstarCuPy(
            flow,
            epsilon=epsilon,
            heuristicsToUse=heuristics,
            occupancyThreshold=occThreshold,
        )
        stepTime = time.time() - stTime
        totTime += stepTime
        lines2Print.append(f"{videoCtr}")
        lines2Print.append(f"{frameCtr}")
        lines2Print.append(f"{stepTime}")
        lines2Print.append(f"{dataSet.compressionRatio}")
        lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
        lines2Print.append("\n")
        with open(compMetrFile, "a") as txt:
            txt.writelines(" ".join(lines2Print))
        lines2Print = []
        frameCtr += 1
        prev_gray = gray
    # frameCtr += ctr
    videoFrameCtr += frameCtr
    print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
    print(f"{videoFrameCtr} frames compressed in {round(totTime,4)}")
    print(f"Elapsed time (so far) {round(totTime,4)}")
    dataSet.saveData(saveName, saveDir, justCores=True, outputType="npy")
2 + 2
