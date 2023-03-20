import os
import time

import numpy as np
import DaMAT as dmt
import scipy.io as sio

from random import sample
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def parallelLoader(dataDir,dataM,infoTuple):
	# dataName=dataDir+'CA_'+DM+f'DM_Pts_{dimension}{instance}_type{motion}.mat'
	# np.concatenate(sio.loadmat(dataName)['CA_Gelsdata'][paramIdx,:].squeeze()).transpose()
	dataName=dataDir+'CA_'+dataM+f'DM_Pts_{infoTuple[2]}{infoTuple[0]}_type{infoTuple[1]}.mat'
	return np.expand_dims(np.concatenate(sio.loadmat(dataName)['CA_Gelsdata'][infoTuple[3],:]).squeeze(),axis=(-1,-2))


cwd=os.getcwd()

outputFileName='catgelttFoaTestIUA001.txt'
outputFilePath=cwd
lines2Print=[]

forgettingFactor=0.7
ttRanks=[1,7,88,656,2745,1469,1] #Final ttRanks for IUA0.01
incMethod='TT-FOA'
scaleData=False
stackData=True
message="Starting ttfoaTest.py with method "+incMethod+f" and ranks {' '.join(map(str,ttRanks))} on fxb-2031"
dmt.dcping(message,'botaks','dorukaks')

timeProbe=time.time()

streamTimeSteps=10
paramSize=6400
if streamTimeSteps==1:
	increment=3
elif streamTimeSteps==10:
	increment=30
else:
	raise ValueError(f'The stream timestep={streamTimeSteps} is not known!')
# paramBatch=4
# paramSize=6400
# batchSize=6400//4


lines2Print.append(f'{time.time()}')
lines2Print.append('ttFOA')
lines2Print.append(' '.join(map(str,ttRanks)))
lines2Print.append('\n')
with open(outputFilePath+outputFileName,'a') as txt:
		txt.writelines(' '.join(lines2Print))
lines2Print=[]



DM='3rd'
latticeDimensions = ['x','y','z']#['x']#, 'y']#, 'z']
motionTypes = [1]
instanceNumber = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
# instanceNumber = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
# dataLocation= '/media/dorukaks/Database/Nathan_GelsData'+'/'+DM+'_DM/'
dataLocation= '/home/dorukaks'+'/'+DM+'_DM/' #for doruk-fxb2031
dataLocation= '/home/dorukaks/Desktop/GameDataProject/catgelTrainData/' #for doruk-fxb2031

dataStream=[]

for motion in motionTypes:
	for idx in range(paramSize):
		for instance in instanceNumber:
			for dimension in latticeDimensions:
				dataStream.append((instance,motion,dimension,idx))
				

runPool=[]
startRuns=[]
nCompressedSims=0

for _ in range(increment):
	run = dataStream.pop(0)
	runPool.append(run)
	startRuns.append(run)
print(startRuns)
# prelimData=[]
# for instance,motion,dimension,start,end in startRuns:
loopTime=time.time()

with open(dataLocation+'catgel_trainData0.cgf','rb') as gg:
	prelimData=np.expand_dims(np.load(gg),-1)
dataNorm=np.linalg.norm(prelimData)
stepError=[]
simNorm=[]
curErr=[]
simNorm.append(dataNorm)


stepProbe=time.time()
S,ttCores=dmt.ttObject.ttfoa6d(prelimData,ttRanks,forgettingFactor=forgettingFactor)
stepTime=time.time()-stepProbe
print(f'Initial TT-FOA took {round(stepTime,4)} s')
nCompressedSims+=1
lines2Print.append('ttFOA')
lines2Print.append(f'{stepTime}') #Time to compute TT-FOA
lines2Print.append(f'{dataNorm}') #Norm of the simulation
stepError.append(
	np.linalg.norm(
		dmt.utils.coreContraction(ttCores[:-1]+ttCores[-1][:,-1]) - prelimData
		)
	) #Norm of the error
for idx in range(nCompressedSims):
	origData=np.load(dataLocation+f'catgel_trainData{idx}.cgf')
	curErr.append(
		np.linalg.norm(
			dmt.utils.coreContraction(ttCores[:-1]+ttCores[-1][idx])-origData
			)
		)
	



