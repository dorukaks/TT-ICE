import os
import time
import cv2
import glob

import numpy as np
import DaMAT as dmt

cwd = os.getcwd()
videoDir = "/media/dorukaks/Database/minecraftData/"
saveDir =  "/media/dorukaks/Database/minecraftCores/"
method = "ttsvd"
heuristics = ["occupancy","skip"]
occThreshold = 1
lines2Print = []
epsilon = 0.1
downSampleRatio=4
tempDS=5
saveName=f'minecraftCoresE'+"".join(str(epsilon).split("."))+f'sDS{downSampleRatio}tDS{tempDS}_f32'
compMetrFile = f"minecraftMetricsE"+"".join(str(epsilon).split("."))+f"sDS{downSampleRatio}tDS{tempDS}_f32.txt"  # This file is for me


videoFiles = glob.glob(videoDir + "*.mp4")
frames = []
print(videoFiles[0])
video = cv2.VideoCapture(videoFiles[0])

success, image = video.read()
# newShape = []
# newShape.extend(dmt.utils.primes(image.shape[0]))
# newShape.extend(dmt.utils.primes(image.shape[1])[::-1])
# newShape.append(image.shape[2])
# newShape += [-1]
# newShape=[8,8,10,10,6,6,3,-1]
# newShape=[16,40,20,18,3,-1]
# newShape=[64,10,20,18,3,-1]
newShape=[360,640,3,-1]
# newShape=[180,320,3,-1]
# newShape=[6,6,10,10,8,8,3,-1]
# newShape=[12,30,40,16,3,-1]
# frames.append(image[:, :, :, None])
xDs=([True]+[False]*(downSampleRatio-1))*(360//downSampleRatio)
yDs=([True]+[False]*(downSampleRatio-1))*(640//downSampleRatio)
newShape[0]=newShape[0]//downSampleRatio
newShape[1]=newShape[1]//downSampleRatio

image=image[xDs,:,:]
image=image[:,yDs,:]

# frames = np.concatenate(image, axis=-1)
dataSet = dmt.ttObject(
    image, epsilon=epsilon, keepData=False, samplesAlongLastDimension=True, method=method
)
dataSet.changeShape(newShape=newShape)
dataSet.ttDecomp(dtype=np.float32)
videoFrameCtr=0
frameCtr = 0
videoCtr=0
lines2Print.append(" ".join(map(str, newShape)))
lines2Print.append("\n")
lines2Print.append(f"{videoCtr}")
lines2Print.append(f"{frameCtr}")
lines2Print.append(f"{dataSet.compressionTime}")
lines2Print.append(f"{dataSet.compressionRatio}")
lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
lines2Print.append("\n")
ctr=1
with open(compMetrFile, "a") as txt:
    txt.writelines(" ".join(lines2Print))
lines2Print = []
totTime = dataSet.compressionTime
while success:
    # print(frameCtr)
    success, image = video.read()
    if ctr % tempDS ==0:
        try:
            image=image[xDs,:,:]
            image=image[:,yDs,:]
            image=image.reshape(newShape)
            stTime=time.time()
            dataSet.ttICEstar(
            # dataSet.ttICEstarCuPy(
            image,
            epsilon=epsilon,
            heuristicsToUse=heuristics,
            occupancyThreshold=occThreshold,
        )
            stepTime=time.time()-stTime
            totTime+=stepTime
            lines2Print.append(f"{videoCtr}")
            lines2Print.append(f"{frameCtr}")
            lines2Print.append(f"{stepTime}")
            lines2Print.append(f"{dataSet.compressionRatio}")
            lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
            lines2Print.append("\n")
            with open(compMetrFile, "a") as txt:
                txt.writelines(" ".join(lines2Print))
            lines2Print = []
            frameCtr += tempDS
            videoFrameCtr += tempDS
        except TypeError:
            pass
    ctr+=1
print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
print(f"{videoFrameCtr} frames compressed in {round(totTime,4)}")
print(f"Elapsed time (so far) {round(totTime,4)}")
dataSet.saveData(saveName,saveDir,justCores=True,outputType='npy')
for videoFile in videoFiles[1:]:
    ctr=0
    videoCtr+=1
    print(videoFile)
    # frames = []
    video = cv2.VideoCapture(videoFile)
    success, image = video.read()
    # frames.append(image[:, :, :, None])
    videoFrameCtr=0
    frameCtr=0
    while success:
        # print(ctr)
        success, image = video.read()
        if ctr % tempDS==0:
            stTime = time.time()
            try:
                image=image[xDs,:,:]
                image=image[:,yDs,:]
                image=image.reshape(newShape)
                dataSet.ttICEstar(
                # dataSet.ttICEstarCuPy(
                image,
                epsilon=epsilon,
                heuristicsToUse=heuristics,
                occupancyThreshold=occThreshold,
            ) 
                stepTime=time.time()-stTime
                totTime+=stepTime
                lines2Print.append(f"{videoCtr}")
                lines2Print.append(f"{frameCtr}")
                lines2Print.append(f"{stepTime}")
                lines2Print.append(f"{dataSet.compressionRatio}")
                lines2Print.append(" ".join(map(str, dataSet.ttRanks)))
                lines2Print.append("\n")
                with open(compMetrFile, "a") as txt:
                    txt.writelines(" ".join(lines2Print))
                lines2Print = []
                frameCtr += tempDS
                videoFrameCtr += tempDS
            except TypeError:
                pass
            updTime = time.time() - stTime
        ctr+=1
        # totTime += updTime
    # frames = np.concatenate(frames, axis=-1)
    # dataSet.ttICEstar(
    #     frames,
    #     epsilon=epsilon,
    #     heuristicsToUse=heuristics,
    #     occupancyThreshold=occThreshold,
    # )
    print(round(dataSet.compressionRatio, 2), dataSet.ttRanks)
    print(f"{videoFrameCtr} frames compressed in {round(updTime,4)}")
    print(f"Elapsed time (so far) {round(totTime,4)}")
    dataSet.saveData(saveName,saveDir,justCores=True,outputType='npy')
    2 + 2
