import os
import time

import numpy as np
import DaMAT as dmt

from random import sample
from datetime import datetime
from scipy.io import loadmat

cwd = os.getcwd()
tensorSaveLocation = "/randomTensors/"

rep=20
dims=[10,15,20,500]
ranks=[4,10,30]
# ranks=[3,5,10]
forgettingFactor=0.7

tenIdx=0
totalTime=0
lines2print=[]
alltime=0

ttCores=None
S=None
errors=np.zeros((20,dims[-1]))
print('TT-FOA on random tensor')
print(f"Estimated ranks: "+" ".join(map(str,ranks)))
for tenIdx in range(rep):
	totalTime=0
	print(f'Tensor:{tenIdx}')
	ttCores=None
	S=None
	# xTrue=loadmat('random4d_2_3_5_Tensors.mat')[f'ten{tenIdx}']
	xTrue=loadmat(cwd+tensorSaveLocation+'random4d_4_10_30_Tensors.mat')[f'ten{tenIdx}']
	for sampleIdx in range(dims[-1]):
		# cumErr=[]
		# cumRelErr=[]
		# print(f'Sample: {sampleIdx}')
		stTime=time.time()
		S,ttCores=dmt.ttObject.ttfoa4d(xTrue[...,sampleIdx][...,None],ranks,ttCores=ttCores,S=S,forgettingFactor=forgettingFactor)
		stepTime=time.time()-stTime
		totalTime+=stepTime
		# print(f'Compressed in: {round(stepTime,4)}')
		xRec=dmt.utils.coreContraction(ttCores[:-1]+[ttCores[-1][:,sampleIdx]])
		err=xTrue[...,sampleIdx]-xRec
		relErr=np.linalg.norm(err)/np.linalg.norm(xTrue[...,sampleIdx])
		errors[tenIdx,sampleIdx]=relErr
		lines2print.append(f'{relErr}')
		# print(f'Relative error: {relErr}')

		# for idx in range(sampleIdx+1):
		# # origData=np.load(dataLocation+f'catgel_trainData{idx}.cgf')
		#     elemErr=np.linalg.norm(
		# 		dmt.utils.coreContraction(ttCores[:-1]+[ttCores[-1][:,idx]])-xTrue[...,idx].squeeze()
		#         )
		#     cumRelErr.append(
		# 	elemErr/np.linalg.norm(xTrue[...,idx])
		# 	) # Cumulative relative error of the compressed simulations (Because of the forgetting factor)
		#     cumErr.append(elemErr*elemErr)
		# # Cumulative relative error of the compressed simulations (Because of the forgetting factor)
		# print(f'{np.sqrt(np.sum(cumErr))}') # Cumulative error
		# print(f'{np.sqrt(np.sum(cumErr))/np.linalg.norm(xTrue[...,:sampleIdx+1])}')
		# print(f'{np.mean(cumRelErr)}') # mean relative cumulative error
	print(f'Compression time: {totalTime}')
	alltime+=totalTime
	lines2print.append(f'\n')

# with open('./ttfoaPythonErrors.txt','a') as txt:
# 	txt.writelines(' '.join(lines2print))
lines2print=[]
print(f"A total of {rep} random tensors compressed with TT-FOA: {alltime} seconds")
print(f"Average compression time per tensor                : {alltime/rep} seconds")
# np.savetxt("ttfoaOR.txt",errors.mean(0))

# tt_dim = [10,15,20,25,29,500]
# tt_rank= [ 1, 2, 3, 5, 6, 10,1]
# forgettingFactor=0.7

## Create tensors ##

# orig_cores=[]
# for idx in range(len(tt_dim)):
#     orig_cores.append((np.random.rand(tt_rank[idx],tt_dim[idx],tt_rank[idx+1])))
# xTrue=np.memmap('randxTrue.npmmp',dtype='float32',mode='w+',shape=tuple(tt_dim))
# for idx in range(tt_dim[-1]):
#   print(idx)
#   xTrue[...,idx]=dmt.utils.coreContraction(orig_cores[:-1]+[orig_cores[-1][:,idx,:]]).squeeze()
# orig_cores=[]

# ttCores=None
# S=None
# for sampleIdx in range(tt_dim[-1]):
#     cumErr=[]
#     cumRelErr=[]
#     print(f'Sample {sampleIdx}')
#     stTime=time.time()
#     S,ttCores=dmt.ttObject.ttfoa6d(xTrue[...,sampleIdx][...,None],tt_rank[1:-1],ttCores=ttCores,S=S,forgettingFactor=forgettingFactor)
#     stepTime=time.time()-stTime
#     print(f'Compressed in {round(stepTime,4)}')
#     xRec=dmt.utils.coreContraction(ttCores[:-1]+[ttCores[-1][:,sampleIdx]])
#     err=xTrue[...,sampleIdx]-xRec
#     relErr=np.linalg.norm(err)/np.linalg.norm(xTrue[...,sampleIdx])
#     print(f'Relative error {relErr}')
#     for idx in range(sampleIdx+1):
# 		# origData=np.load(dataLocation+f'catgel_trainData{idx}.cgf')
#         elemErr=np.linalg.norm(
# 				dmt.utils.coreContraction(ttCores[:-1]+[ttCores[-1][:,idx]])-xTrue[...,idx].squeeze()
#             )
#         cumRelErr.append(
# 			elemErr/np.linalg.norm(xTrue[...,idx])
# 			) # Cumulative relative error of the compressed simulations (Because of the forgetting factor)
#         cumErr.append(elemErr*elemErr)
# 		# Cumulative relative error of the compressed simulations (Because of the forgetting factor)
#     print(f'{np.sqrt(np.sum(cumErr))}') # Cumulative error
#     print(f'{np.sqrt(np.sum(cumErr))/np.linalg.norm(xTrue[...,:sampleIdx+1])}')
#     print(f'{np.mean(cumRelErr)}') # mean relative cumulative error