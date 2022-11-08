import warnings
import time

import numpy as np

# from ast import Pass, Try
from logging import warning
# from numpy import core
from DaMATutils import * 
from pickle import dump,load
from datetime import datetime
# from discord_webhook import DiscordWebhook as dcmessage

class ttObject:
    def __init__(self,data,epsilon=None,keepData=False,samplesAlongLastDimension=True,method='ttsvd') -> None:
        self.inputType=type(data)
        self.keepOriginal = keepData ##boolean variable to determine if you want to store the original data along with the compression (for some weird reason)
        self.nCores=None
        self.samplesAlongLastDimension=samplesAlongLastDimension
        # self.ttRanks=ranks
        self.ttCores=None
        self.originalData=data
        self.nElements=None
        self.method=method

        if self.inputType==np.ndarray:
            self.ttEpsilon=epsilon
            self.originalShape=data.shape
            self.reshapedShape=self.originalShape
            self.indexOrder=[idx for idx in range(len(self.originalShape))]
        elif self.inputType==list:
            self.nCores=len(data)
            self.ttCores=data
            self.reshapedShape=[core.shape[1] for core in self.ttCores]
        else:
            raise TypeError("Unknown input type!")
        

    ## List of required class methods
    def changeShape(self,newShape:tuple) -> tuple: ##function to change shape of input tensors and keeping track
        # a simple numpy.reshape was sufficient for this function but in order to keep track of the shape changes the self.reshapedShape also needs to be updated
        if self.ttCores!=None: 
            warning("Warning! You are reshaping the original data after computing a TT-decomposition! We will proceed without reshaping self.originalData!!")
            return None
        self.reshapedShape=newShape
        self.originalData=np.reshape(self.originalData,self.reshapedShape)
        self.reshapedShape=self.originalData.shape
        if self.samplesAlongLastDimension:
            self.singleDataShape=self.reshapedShape[:-1] #This line assumes we keep the last index as the samples index and don't interfere with its shape

    def computeTranspose(self,newOrder:list) -> list: ##function to transpose the axes of input tensor -> might be unnecessary
        assert (self.inputType==np.ndarray and self.ttCores==None)
        if len(newOrder)!=len(self.indexOrder): raise ValueError('size of newOrder and self.indexOrder does not match. Maybe you forgot reshaping?')
        self.indexOrder=[self.indexOrder[idx] for idx in newOrder]
        self.originalData=self.originalData.transpose(newOrder)

    def indexMap(self) -> None: ## function to map the original indices to the reshaped indices
        self.A=2

    def primeReshaping(self) -> None: ## function to reshape the first d dimensions to prime factors
        self.A=2

    def saveData(self,fileName:str,directory="./",justCores=True,outputType='ttc') -> None: ## function to write the computed tt-cores to a .ttc file -> should provide alternative output formats such as .txt
        saveFile=open(fileName+'.ttc','wb')
        if justCores:
            if outputType=='ttc':
                temp=ttObject(self.ttCores)
                for attribute in vars(self):
                    if attribute !='originalData':
                        setattr(temp,attribute,eval(f'self.{attribute}'))
                dump(temp,saveFile)
                saveFile.close()
            elif outputType=='txt':
                for coreIdx,core in enumerate(self.ttCores):
                    np.savetxt(f'{fileName}_{coreIdx}.txt',core.reshape(-1,core.shape[-1]),header=f'{core.shape[0]} {core.shape[1]} {core.shape[2]}', delimiter=' ')
            else:
                raise ValueError(f'Output type {outputType} is not supported!')
        else:
            if outputType=='txt': raise ValueError(".txt type outputs are only supported for justCores=True!!")
            if self.method=='ttsvd':# or self.method=='ttcross': ## TT-cross support might be omitted
                dump(self,saveFile)
                saveFile.close()
            else:
                raise ValueError('Unknown Method!')
    @staticmethod
    def loadData(fileName:str,numCores=None) -> "ttObject": ## function to load data from a .ttc file -> additional support may be included for .txt files with a certain format?
        ''' Static method to load TT-cores into a ttObject object. 
            Note that if data is stored in {coreFile}_{coreIdx}.txt format, the input fileName should just be coreFile.txt '''
        fileExt=fileName.split('.')[-1]
        if fileExt=='ttc':
            with open(fileName,'rb') as f:
                dataSetObject=load(f)
            return dataSetObject
        elif fileExt=="txt":
            if numCores==None: raise ValueError("Number of cores are not defined!!")
            fileBody=fileName.split('.')[0]
            coreList=[]
            for coreIdx in range(numCores):
                with open(f'{fileBody}_{coreIdx}.{fileExt}'):
                    coreShape=f.readline()[2:-1]
                    coreShape=[int(item) for item in coreShape.split(" ")]
                coreList.append(np.loadtxt(f'{fileBody}_{coreIdx}.{fileExt}').reshape[coreShape])
            return coreList
        else:
            raise ValueError(f'{fileExt} files are not supported!')
    @staticmethod
    def ttDot(tt1,tt2) -> float():
        if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            if isinstance(tt2,list):
                tt2=ttObject(tt2)
            if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
                raise AttributeError('One of the passed objects is not in TT-format!')
        v=np.kron(tt1.ttCores[0][:,0,:],tt2.ttCores[0][:,0,:])
        for i1 in range(1,tt1.ttCores[0].shape[1]):
            v+=np.kron(tt1.ttCores[0][:,i1,:],tt2.ttCores[0][:,i1,:])
        for coreIdx in range(1,len(tt1.ttCores)):
            p=[]
            for ik in range(tt1.ttCores[coreIdx].shape[1]):
                p.append(v@(np.kron(tt1.ttCores[coreIdx][:,ik,:],tt2.ttCores[coreIdx][:,ik,:])))
            v=np.sum(p,axis=0)
        return v.item()

    @staticmethod
    def ttNorm(tt1) -> float:
        if not isinstance(tt1,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            else:
                raise AttributeError('Passed object is not in TT-format!')
        try:
            norm=ttObject.ttDot(tt1,tt1)
            norm=np.sqrt(norm)
        except MemoryError:
            norm=np.linalg.norm(tt1.ttCores[-1])
        return norm

    def projectTensor(self,newData:np.array,upTo=None) -> np.array: ## function to project tensor onto basis spanned by tt-cores
        for coreIdx,core in enumerate(self.ttCores):
            if (coreIdx==len(self.ttCores)-1) or coreIdx==upTo :
                break
            newData=(core.reshape(np.prod(core.shape[:2]),-1).transpose())@newData.reshape(self.ttRanks[coreIdx]*self.ttCores[coreIdx].shape[1],-1)
        return newData
    def reconstruct(self,projectedData,upTo=None):
        if upTo==None:
            upTo=len(self.ttCores)-1 #Write the core index in 1-indexed form!!
        for core in self.ttCores[:upTo][::-1]:
            projectedData=np.tensordot(core,projectedData,axes=(-1,0))
        return projectedData

    @property
    def coreOccupancy(self) -> None: ## function to return core occupancy
        try:
            return  [core.shape[-1]/np.prod(core.shape[:-1]) for core in self.ttCores[:-1]]
        except ValueError:
            warnings.warn('No TT cores exist, maybe forgot calling object.ttDecomp?',Warning)
            return None
    
    @property
    def compressionRatio(self) -> float: ## function to compute compression ratio of existing cores
        originalNumEl=1
        compressedNumEl=0
        for core in self.ttCores:
            originalNumEl*=core.shape[1]
            compressedNumEl+=np.prod(core.shape)
        return originalNumEl/compressedNumEl

    def updateRanks(self) -> None: ## function to update the ranks of the ttobject after incremental updates
        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def computeRelError(self,data:np.array) -> np.array: ## computes relative error by projecting data
        elementwiseNorm=np.linalg.norm(data,axis=0)
        for idx in range(len(data.shape)-2):
            elementwiseNorm=np.linalg.norm(elementwiseNorm,axis=0)
        projectedData=self.projectTensor(data)
        reconstructedData=self.reconstruct(projectedData).reshape(data.shape)
        difference=data-reconstructedData
        differenceNorm=np.linalg.norm(difference,axis=0)
        for idx in range(len(difference.shape)-2):
            differenceNorm=np.linalg.norm(differenceNorm,axis=0)
        relError=differenceNorm/elementwiseNorm
        return relError

    def computeRecError(self,data:np.array,start=None,finish=None) -> None: ## computes relative error by reconstructing data
        self.A=2


    
    #List of methods that will compute a decomposition
    def ttDecomp(self,norm=None,dtype=np.float32) -> "ttObject.ttCores": ##tt decomposition to initialize the cores, will support ttsvd for now but will be open to other computation methods
        if norm==None: norm=np.linalg.norm(self.originalData)
        if self.method=='ttsvd':
            startTime=time.time()
            self.ttRanks,self.ttCores=ttsvd(self.originalData,norm,self.ttEpsilon,dtype=dtype)
            self.compressionTime=time.time()-startTime
            self.nCores=len(self.ttCores)
            self.nElements=0
            for cores in self.ttCores:
                self.nElements+=np.prod(cores.shape)
            if not self.keepOriginal:
                self.originalData=None
            return None
        else:
            raise ValueError('Method unknown. Please select a valid method!')

    def ttICE(self,newTensor,epsilon=None,tenNorm=None) -> None: ##TT-ICE algorithmn without heuristics
        if tenNorm==None: tenNorm=np.linalg.norm(newTensor)
        if epsilon==None: epsilon=self.ttEpsilon
        newTensorSize=len(newTensor.shape)-1
        newTensor=newTensor.reshape(list(self.reshapedShape[:-1])+[-1])[None,:]
        newTensor=newTensor.reshape(self.reshapedShape[0],-1)
        # newTensor=newTensor.reshape(tuple(list(self.reshapedShape[:-1])+[-1]))[None,:] # if line above does not work, use this one instead  
        # Test the code, not sure if this actually works
        Ui=self.ttCores[0].reshape(self.reshapedShape[0],-1)
        Ri=newTensor-Ui@(Ui.T@newTensor)
        for coreIdx in range(0,len(self.ttCores)-2):
            URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
            self.ttCores[coreIdx]=np.hstack((Ui,URi)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)
            # Need to check these following three lines
            Ui=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0]*self.reshapedShape[coreIdx],-1)
            newTensor=(Ui.T@newTensor).reshape(np.prod(self.ttCores[coreIdx+1].shape[:-1]),-1)
            Ui=self.ttCores[coreIdx+1].reshape(self.ttCores[coreIdx].shape[-1]*self.reshapedShape[coreIdx+1],-1)
            Ri=newTensor-Ui@(Ui.T@newTensor)
        coreIdx=len(self.ttCores)-2
        URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
        self.ttCores[coreIdx]=np.hstack((Ui,URi))#.reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
        self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)
        newTensor=self.ttCores[coreIdx].T@newTensor
        self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
        coreIdx+=1
        Ui=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],-1)
        self.ttCores[coreIdx]=np.hstack((Ui,newTensor)).reshape(self.ttCores[coreIdx].shape[0],-1,1)
        '''
        # Remove this after testing the method above

        for coreIdx in range(len(self.ttCores)):
            Ui=self.ttCores[coreIdx].reshape(np.prod(self.ttCores[coreIdx].shape[:-1]),-1)
            if coreIdx==0:
                newTensor=newTensor.reshape(Ui.shape[0],-1)
            elif coreIdx==len(self.ttCores)-1:
                pass
            else:
                newTensor=self.ttCores[coreIdx-1].reshape(np.prod(self.ttCores[coreIdx-1].shape[:-1]),-1).T@newTensor
                newTensor=newTensor.reshape(Ui.shape[0],-1)
            if coreIdx==len(self.ttCores)-1:
                newTensor=self.ttCores[coreIdx-1].reshape(-1,self.ttCores[coreIdx-1].shape[-1]).T@newTensor
                self.ttCores[coreIdx]=np.concatenate((tempCore,newTensor),axis=1)[:,:,None]
                break
            else:
                Ri=newTensor-Ui@(Ui.T@newTensor)
            URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
            self.ttCores[coreIdx]=np.hstack((Ui,URi)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            if coreIdx!=len(self.ttCores)-2:
                self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)
            else:
                tempCore=self.ttCores[coreIdx+1].squeeze(-1)
                tempCore=np.concatenate((tempCore,np.zeros((URi.shape[1],tempCore.shape[1]))),axis=0)
        ''' 
        self.updateRanks()
        # self.ttRanks=[1]
        # for core in self.ttCores:
        #     self.ttRanks.append(core.shape[-1])
        return None


    def ttICEstar(self,newTensor,epsilon=None,tenNorm=None,heuristicsToUse=['skip','subselect','occupancy'],occupancyThreshold=0.8) -> None: ##TT-ICE* algorithmn with heuristics -> support of heuristic modifications at various level
        if tenNorm==None: tenNorm=np.linalg.norm(newTensor)
        if epsilon==None: epsilon=self.ttEpsilon
        if ('subselect' in heuristicsToUse) and (newTensor.shape[-1]==1): warning('The streamed tensor has only 1 observation in it. Subselect heuristic will not be useful!!')
        updEpsilon=epsilon
        newTensorSize=len(newTensor.shape)-1

        elementwiseEpsilon=self.computeRelError(newTensor)
        if 'skip' in heuristicsToUse:
            if mean(elementwiseEpsilon)<=epsilon:
                newTensor=self.projectTensor(newTensor)
                self.ttCores[-1]=np.hstack((self.ttCores[-1].reshape(self.ttRanks[-2],-1),newTensor)).reshape(self.ttRanks[-2],-1,1)
                return None
        select=[True]*newTensor.shape[-1]
        discard=[False]*newTensor.shape[-1]
        if 'subselect' in heuristicsToUse:
            elementwiseNorm=np.linalg.norm(newTensor,axis=0)
            for _ in range(len(self.ttCores)-2):
                elementwiseNorm=np.linalg.norm(elementwiseNorm,axis=0)
            allowedError=(dataSet.ttEpsilon*np.linalg.norm(elementwiseNorm))**2
            select=elementwiseEpsilon>epsilon
            discard=elementwiseEpsilon<=epsilon
            discardedEpsilon=np.sum((elementwiseEpsilon[select]*elementwiseNorm[select])**2)/np.sum(np.linalg.norm(elementwiseNorm[discard])**2)
            discardedError=discardedEpsilon*(np.linalg.norm(elementwiseNorm[discard])**2)
            updEpsilon=np.sqrt((allowedError-discardedError)/(np.linalg.norm(elementwiseNorm[select])**2))
        
        indexString='['
        for _ in range(len(self.reshapedShape)-1): #this heuristic assumes that the last dimension is for observations
            indexString+=':,'
        selectString=indexString+"select]"
        discardString=indexString+"discard]"
        selected=eval('newTensor'+indexString)
        # discarded=eval('newTensor'+indexString) #looks unnecessary, might get rid of this line#
        
        selected=selected.reshape(list(self.reshapedShape[:-1])+[-1])[None,:]
        # selected=selected.reshape(tuple(list(self.reshapedShape[:-1])+[-1]))[None,:] # if line above does not work, use this one instead  

        # Test the code, not sure if this actually works

        # Ui=self.ttCores[0].reshape(self.reshapedShape[0],-1)
        for coreIdx in range(0,len(self.ttCores)-1):
            if ('occupancy' in heuristicsToUse) and (self.coreOccupancy[coreIdx]>=occupancyThreshold):
                #It seems like you don't need to do anything else here, but check and make sure!#
                # continue
                pass
            else:
                Ui=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx]*self.reshapedShape[coreIdx+1],-1)
                Ri=selected-Ui@(Ui.T@selected)
                URi,_,_=deltaSVD(Ri,np.linalg.norm(elementwiseNorm[select]),newTensorSize,updEpsilon)
                self.ttCores[coreIdx]=np.hstack((Ui,URi))#.reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
                self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)


            selected=(self.ttCores[coreIdx].T@selected).reshape(self.ttRanks[coreIdx+1]*self.reshapedShape[coreIdx+1]) #project onto existing core and reshape for next core
            self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1) #fold back the previous core
        coreIdx+=1 # coreIdx=len(self.ttCores), i.e working on the last core
        self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],-1)
        self.ttCores[coreIdx]=np.hcat((self.ttCores[coreIdx],selected)).reshape(self.ttCores[coreIdx].shape[0],-1,1)

        # # Need to check these following three lines
        # Ui=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0]*self.reshapedShape[coreIdx],-1)
        # newTensor=Ui.T@newTensor
        # Ri=newTensor-Ui@(Ui.T@newTensor)

        # coreIdx=len(self.ttCores)-2
        # URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
        # self.ttCores[coreIdx]=np.hstack((Ui,URi)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
        # self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)
        # newTensor=Ui.T@newTensor
        # coreIdx+=1
        # Ui=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0]*self.reshapedShape[coreIdx],-1)
        # self.ttCores[coreIdx]=np.hstack((Ui,newTensor)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
        self.updateRanks()
        return none

    def ttRound(self,norm,epsilon=0) -> None: ##tt rounding as per oseledets 2011 -> Might be implemented as a utility
        d=[core.shape[1] for core in self.ttCores]
        for coreIdx in np.arange(len(self.ttCores))[::-1]:
            currCore=self.ttCores[coreIdx]
            currCore=currCore.reshape(currCore.shape[0],-1) ## Compute mode 1 unfolding of the tt-core
            q,r=np.linalg.qr(currCore.T) ## Using the transpose results in a row orthogonal Q matrix !!!
            q,r=q.T,r.T
            self.ttCores[coreIdx]=q
            self.ttCores[coreIdx-1]=np.tensordot(self.ttCores[coreIdx-1],r,axes=(-1,0)) #contract previous tt-core and R matrix
            if coreIdx==1:
                break
                # pass ##TODO: write the case for the last core here.
        ### Compression of the orthogonalized representation ###
        ranks=[1]
        for coreIdx in range(len(self.ttCores)-1):
            core=self.ttCores[coreIdx]
            core=core.reshape(np.prod(core.shape[:-1]),-1)
            self.ttCores[coreIdx],sigma,v=deltaSVD(core,norm,len(self.ttCores),epsilon)
            ranks.append(self.ttCores[coreIdx].shape[-1])
            self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(ranks[coreIdx],d[coreIdx],-1) #fold matrices into tt-cores
            self.ttCores[coreIdx+1]=((np.diag(sigma)@v)@self.ttCores[coreIdx+1]).reshape(ranks[-1]*d[coreIdx+1],-1)
        self.ttCores[-1]=self.ttCores[-1].reshape(-1,d[-1],1)
        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    @staticmethod
    def ittd(tt1,tt2,rounding=True,epsilon=0) -> "ttObject": ##ittd method from liu2018 for comparison
        if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            if isinstance(tt2,list):
                tt2=ttObject(tt2)
            if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
                raise AttributeError('One of the passed objects is not in TT-format!')
        tt1.ttCores[-1]=tt1.ttCores[-1].squeeze(-1)
        tt2.ttCores[-1]=tt2.ttCores[-1].squeeze(-1)
        rank1,numObs1=tt1.ttCores[-1].shape
        rank2,numObs2=tt2.ttCores[-1].shape
        tt1.ttCores[-1]=np.concatenate((tt1.ttCores[-1],np.zeros((rank1,numObs2))),axis=1)
        tt2.ttCores[-1]=np.concatenate((np.zeros((rank2,numObs1)),tt2.ttCores[-1]),axis=1)
        ## Addition in tt-format ##
        for coreIdx in range(len(tt1.ttCores)):
            if coreIdx==0:
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx].squeeze(0),tt2.ttCores[coreIdx].squeeze(0)),axis=1)[None,:,:]
            elif coreIdx == len(tt1.ttCores)-1:
                # tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx].squeeze(-1),tt2.ttCores[coreIdx].squeeze(-1)),axis=0)[:,:,None]
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],tt2.ttCores[coreIdx]),axis=0)[:,:,None]
            else:
                s11,s12,s13=tt1.ttCores[coreIdx].shape
                s21,s22,s23=tt2.ttCores[coreIdx].shape
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],np.zeros((s11,s12,s23))),axis=2)
                tt2.ttCores[coreIdx]=np.concatenate((np.zeros((s21,s22,s13)),tt2.ttCores[coreIdx]),axis=2)
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],tt2.ttCores[coreIdx]),axis=0)
        tt1.ttRanks=[1]
        for core in tt1.ttCores:
            tt1.ttRanks.append(core.shape[-1])
        if rounding:
            tenNorm=ttObject.ttNorm(tt1)
            tt1.rounding(tenNorm,epsilon=epsilon)
        return tt1
    


class ttObjectLegacy:
    def __init__(self,data,method:str='ttsvd',epsilon=None,ranks=None,crossIter=1,originalShape=None,transposeMap=None,imageOrder=None) -> None:
        self.inputType=type(data)
        if self.inputType==np.ndarray: #if you are starting with original data (tensor form) you will give numpy arrays as input at the class initialization
            self.nCores=None
            self.ttRanks=ranks
            self.ttCores=None
            self.originalData=data
            self.ttEpsilon=epsilon
            self.originalShape=self.originalData.shape
            self.reshapedShape=self.originalShape
            self.imageWidth=self.originalShape[0] ## Remove
            self.imageHeight=self.originalShape[1] ## Remove
            self.imageDepth=self.originalShape[2] ## Remove
            self.nElements=None
            self.imageCount=np.prod(self.originalShape[3:])
            if transposeMap==None:
                self.transposeMap=[idx for idx in range(len(data.shape))]
            else:
                self.transposeMap=transposeMap
        elif self.inputType==list: #if you are starting with compressed data (tt form) you will give list of arrays as input at the class initialization
            self.nCores=len(data)
            self.originalData=None
            self.ttEpsilon=epsilon
            self.nElements=0
            self.ttCores=data.copy()
            for cores in self.ttCores:
                self.nElements+=np.prod(cores.shape)
            self.ttRanks=[1]+[cores.shape[2] for cores in data]
            if originalShape == None:
                self.originalShape=[cores.shape[1] for cores in data]
                self.reshapedShape=self.originalShape
            elif type(originalShape) == tuple or type(originalShape) == list:
                self.originalShape=tuple(originalShape)
                self.reshapedShape=[cores.shape[1] for cores in data]
            else:
                raise TypeError('Unknown original shape!')
            self.originalShape=[cores.shape[1] for cores in data]
            self.reshapedShape=self.originalShape
            self.imageCount=np.prod(self.originalShape)//(np.prod(self.originalShape[:3]))
            # self.imageWidth=84
            # self.imageHeight=84
            # self.imageDepth=3
            if transposeMap==None:
                self.transposeMap=[idx for idx in range(len(data))]
            else:
                self.transposeMap=transposeMap

        else:
            raise TypeError("Unknown input type!")
        # These attributes will be used later for other functionalities, do not hesitate to suggest anything you find useful
        # self.singleReshapedImage=(self.imageWidth,self.imageHeight,self.imageDepth)
        self.method=method
        
        self.crossVerbose=True
        self.crossRankAdapt=False
        self.maxCrossIter=crossIter
        self.recoveryIndices=None
        self.imageCount=np.prod(self.originalShape[3:])
        self.imageOrder=imageOrder
        self.compressionTime=None

    def ttDecomp(self,norm,dtype=np.float32):
        startTime=time.time()
        self.ttRanks,self.ttCores=ttsvd(self.originalData,norm,self.ttEpsilon,dtype=dtype)
        self.compressionTime=time.time()-startTime
        self.nCores=len(self.ttCores)
        self.nElements=0
        for cores in self.ttCores:
            self.nElements+=np.prod(cores.shape)
        return None
    def ttDecompLarge(self):
        #compression function, you will prepare most of the inputs/options to this function while initializing the object
        if self.method=="ttsvd":
            startTime=time.time()
            self.ttRanks,self.ttCores=ttsvdLarge(self.originalData,self.ttEpsilon)
            self.compressionTime=time.time()-startTime
            self.nCores=len(self.ttCores)
            self.nElements=0
        elif self.method=="ttcross":
            self.recoveryIndices=self.recoveryMap()
            self.ttCores=c3py.TensorTrain.cross_from_numpy_tensor(self.originalData,self.ttRanks,self.crossVerbose,self.crossRankAdapt,maxiter=self.maxCrossIter)
            self.nCores=len(self.ttCores.cores)
            self.nElements=0
            self.ttCores=self.ttCores.cores
        else:
            raise KeyError('Method unknown. Please select a valid method!')
        for cores in self.ttCores:
            self.nElements+=np.prod(cores.shape)
    def changeShape(self,newShape:tuple):
        # a simple numpy.reshape was sufficient for this function but in order to keep track of the shape changes the self.reshapedShape also needs to be updated
        self.reshapedShape=newShape
        self.originalData=np.reshape(self.originalData,self.reshapedShape)
        self.reshapedShape=self.originalData.shape
        self.singleReshapedImage=self.reshapedShape[:-1] #This line assumes we keep the last index as the samples index and don't interfere with its shape

    def createTransposeMap(self):
        #function created specifically to transpose the tensor axes according to the suggested "football" shape and keep track of the changes automatically
        currentIndices=self.originalData.shape
        rankingIndices=[index for element, index in sorted(zip(currentIndices, range(len(currentIndices))))]
        magIndices=np.array([index for element,index in sorted(zip(rankingIndices,range(len(rankingIndices))))],dtype=int)
        posIndices=np.arange(0,len(currentIndices),1,dtype=int)
        combIndices=np.concatenate((posIndices.reshape(-1,1),magIndices.reshape(-1,1)),axis=1)
        evenMags=combIndices[combIndices[:,1]%2==0]
        oddMags=combIndices[combIndices[:,1]%2==1]
        evenMags=evenMags[np.argsort(evenMags[:,1])]
        oddMags=oddMags[np.argsort(oddMags[:,1])][::-1]
        orderedIndices=np.concatenate((evenMags,oddMags),axis=0)
        self.transposeMap=tuple(orderedIndices[:,0])
        self.originalData=np.transpose(self.originalData,axes=self.transposeMap)
        self.reshapedShape=self.originalData.shape

    @staticmethod
    def revertTranspose(input,transposeMap):
        #function created to revert any given transposed tensor
        mapType=type(transposeMap)
        if mapType == tuple or mapType ==list:
            transposeMap=list(transposeMap)
            transposeMap=np.argsort(transposeMap)
            input=np.transpose(input,axes=tuple(transposeMap))
            return input
        else:
            raise TypeError('Unknown transpose map type')


    @staticmethod
    def indexMap(initialShape:tuple,desiredShape:tuple,indexWanted,baseDimension=None,transposeMap=None,fixDimensions=3):
        #function created to find any given image with a given index in transposed and/or reshaped configurations.
        #  This function specifically needs the transpose map and the original/current shape of the tensor 
        initialShape=list(initialShape)
        desiredShape=list(desiredShape)
        if baseDimension == None:
            baseDimension=initialShape[:fixDimensions]
            del initialShape[:fixDimensions],desiredShape[:fixDimensions]
        else:
            map=list(transposeMap)
            # map.sort()
            desiredShape=list(np.array(desiredShape)[list(np.argsort(map))])
            initialShape=list(initialShape)
            del desiredShape[:len(baseDimension)],initialShape[:fixDimensions] #this line assumes that the original matrix has width*height*RGB as FIRST 3 dimensions
            desiredShape=tuple(desiredShape)
            initialShape=tuple(initialShape)
        if len(initialShape)>len(desiredShape):
            mappedIndex=list(np.ravel_multi_index(tuple(indexWanted),desiredShape))
            mappedIndex=tuple(baseDimension+mappedIndex)
        elif len(initialShape)<len(desiredShape):
            mappedIndex=list(np.unravel_index(indexWanted,tuple(desiredShape)))
            mappedIndex=baseDimension+mappedIndex
        else:
            warnings.warn('Input and output have the same size, returning the same index...')
            mappedIndex=tuple(list(baseDimension)+list([indexWanted]))
        if baseDimension ==None:
            return mappedIndex
        else:
            return list(np.array(mappedIndex)[list(transposeMap)])

    def primeReshaping(self,ordered=False,full:bool=False):
        # This is a function used for create reshapings with the prime factorizations
        originalImageShape=(self.imageWidth,self.imageHeight,self.imageDepth) ## Modify this
        if full:
            self.singleReshapedImage=[]
            for dimensions in originalImageShape: self.singleReshapedImage.extend(primes(dimensions))
            factorization=[]
            for dimension in self.originalData.shape: factorization.extend(primes(dimension))
            self.changeShape(tuple(factorization))
        else:
            factorization=primes(self.imageCount)
            if ordered:
                factorization.sort()
                factorization.reverse()
            newShape=tuple(list(self.originalShape[:3])+factorization)
            self.changeShape(newShape)


    def coresOfImage(self,imageIndex):
        desiredImage=ttObject.indexMap(self.originalShape,self.reshapedShape,imageIndex,self.singleReshapedImage,self.transposeMap)
        maps=list(self.transposeMap)
        desiredImage=list(np.array(desiredImage)[list(np.argsort(maps))])
        for idx,indices in enumerate(desiredImage):
            if idx>=len(self.singleReshapedImage):
                desiredImage[idx]=str(indices)
            else:
                desiredImage[idx]=':'
        desiredImage=list(np.array(desiredImage)[list(self.transposeMap)])
        imageCores=[]
        for coreIdx,indices in enumerate(desiredImage):
            if self.method=='ttcross' or self.method=='ttsvd':
                eval(f'imageCores.append(self.ttCores[{coreIdx}][:,'+indices+',:])')
            else:
                raise ValueError('Unknown decomposition method!')
        return imageCores
    
    def returnImage(self,imageCores):
        # This function reconstructs image array from the cores passed into the function handle
        imageArray=coreContraction(imageCores)
        coreLength=len(imageArray.shape)
        if self.method=='ttcross' or self.method=='ttsvd':
            indices2Remove=list(np.arange(coreLength,len(self.ttCores),1,int))
        else:
            raise ValueError('Unknown decomposition method!')
        transposeMap=list(self.transposeMap)
        for index in indices2Remove: transposeMap.remove(index)
        imageArray=np.transpose(imageArray,axes=tuple(np.argsort(transposeMap)))
        imageArray=imageArray.reshape(self.originalShape[:3])
        return imageArray
    
    def returnOriginalImage(self,imageIndex): ### Replace the functionality of this code appropriately
        if type(self.originalData)!=np.ndarray:
            return None
        desiredImage=ttObject.indexMap(self.originalShape,self.reshapedShape,imageIndex,self.singleReshapedImage,self.transposeMap)
        maps=list(self.transposeMap)
        desiredImage=list(np.array(desiredImage)[list(np.argsort(maps))])
        for idx,indices in enumerate(desiredImage):
            if idx>=len(self.singleReshapedImage):
                desiredImage[idx]=str(indices)
            else:
                desiredImage[idx]=':'
        desiredImage=list(np.array(desiredImage)[list(self.transposeMap)])
        index=''
        for idx in desiredImage: index+=idx+','
        index=index[:-1]
        imageArray=eval(f'self.originalData[{index}]')
        transposeMap=list(self.transposeMap)
        indices2Remove=list(np.arange(len(imageArray.shape),len(self.originalData.shape),1,int))
        for index in indices2Remove: transposeMap.remove(index)
        imageArray=np.transpose(imageArray,axes=tuple(np.argsort(transposeMap)))
        return imageArray.reshape(self.imageWidth,self.imageHeight,self.imageDepth)

    def saveData(self,fileName:str,justCores=False):
        # Function handle to save the compressed object to current working directory
        saveFile=open(fileName+'.svf','wb')
        if justCores:
            temp=ttObject(self.ttCores)
            for attribute in vars(self):
                if attribute !='originalData':
                    setattr(temp,attribute,eval(f'self.{attribute}'))
            dump(temp,saveFile)
            saveFile.close()
        else:
            if self.method=='ttsvd' or self.method=='ttcross':
                dump(self,saveFile)
                saveFile.close()
            else:
                raise KeyError('Unknown Method!')
        
        # return 
    @staticmethod
    def loadData(fileName:str):
        # Function to load data from current working directory. Please note that loading is a static method!
        loadFile=open(fileName,'rb')
        dataSetObject=load(loadFile)
        loadFile.close()
        return dataSetObject

    def projectData(self,newData,upTo=None):
        for coreIdx,core in enumerate(self.ttCores):
            if (coreIdx==len(self.ttCores)-1) or coreIdx==upTo :
                break
            newData=(core.reshape(np.prod(core.shape[:2]),-1).transpose())@newData.reshape(self.ttRanks[coreIdx]*self.ttCores[coreIdx].shape[1],-1)
        return newData
    @property
    def coreOccupancy(self): #Returns a list of core occupancy
        try:
            return  [core.shape[-1]/np.prod(core.shape[:-1]) for core in self.ttCores[:-1]]
        except ValueError:
            warnings.warn('No TT cores exist, maybe forgot calling object.ttDecomp?',Warning)
            return None
    @property
    def compressionRatio(self):
        originalNumEl=1
        compressedNumEl=0
        for core in self.ttCores:
            originalNumEl*=core.shape[1]
            compressedNumEl+=np.prod(core.shape)
        return originalNumEl/compressedNumEl

    def updateRanks(self):
        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def reconstruct(self,projectedData,upTo=None):
        if upTo==None:
            upTo=len(self.ttCores)-1 #Write the core index in 1-indexed form!!
        for core in self.ttCores[:upTo][::-1]:
            projectedData=np.tensordot(core,projectedData,axes=(-1,0))
        return projectedData
    def inflateRank(self):

        return None

    # @profile
    def incrementalUpdate(self,newTensor,epsilon=None,tenNorm=None,useHeuristics=False,heuristicsToUse='none',occupancyThreshold=0.8):
        if tenNorm==None:
            tenNorm=np.linalg.norm(newTensor)
        newTensor=newTensor.reshape(tuple(list(self.reshapedShape[:-1])+[-1]))[None,:]
        newTensorSize=len(newTensor.shape)-1
        if epsilon==None: epsilon=self.ttEpsilon

        for coreIdx in range(len(self.ttCores)):
            Ui=self.ttCores[coreIdx].reshape(np.prod(self.ttCores[coreIdx].shape[:-1]),-1)
            if coreIdx==0:
                newTensor=newTensor.reshape(Ui.shape[0],-1)
            elif coreIdx==len(self.ttCores)-1:
                pass
            else:
                newTensor=self.ttCores[coreIdx-1].reshape(np.prod(self.ttCores[coreIdx-1].shape[:-1]),-1).T@newTensor
                newTensor=newTensor.reshape(Ui.shape[0],-1)
            if useHeuristics and (heuristicsToUse=='all' or heuristicsToUse=='occupancy'):
                if (coreIdx!=len(self.ttCores)-1) and self.coreOccupancy[coreIdx]>=occupancyThreshold:
                    continue
            if coreIdx==len(self.ttCores)-1:
                if useHeuristics and (heuristicsToUse=='all' or heuristicsToUse=='subselect'):
                    try:
                        self.ttCores[coreIdx]=tempCore[:,:,None]
                    except NameError:
                        pass
                    break #If data is subselected break at update of the last core and update the last core outside with projection!!
                newTensor=self.ttCores[coreIdx-1].reshape(-1,self.ttCores[coreIdx-1].shape[-1]).T@newTensor
                self.ttCores[coreIdx]=np.concatenate((tempCore,newTensor),axis=1)[:,:,None]
                break
            # elif coreIdx==0:
            #     2+2
            else:
                Ri=newTensor-Ui@(Ui.T@newTensor)
                # 2+2
            URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
            self.ttCores[coreIdx]=np.hstack((Ui,URi)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            if coreIdx!=len(self.ttCores)-2:
                self.ttCores[coreIdx+1]=np.concatenate((self.ttCores[coreIdx+1],np.zeros((URi.shape[-1],self.reshapedShape[coreIdx+1],self.ttRanks[coreIdx+2]))),axis=0)
            else:
                tempCore=self.ttCores[coreIdx+1].squeeze(-1)
                tempCore=np.concatenate((tempCore,np.zeros((URi.shape[1],tempCore.shape[1]))),axis=0)


        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def incrementalUpdate2(self,newTensor,epsilon=None,tenNorm=None,useHeuristics=False,heuristicsToUse='none'):
        if tenNorm==None:
            tenNorm=np.linalg.norm(newTensor)
        newTensor=newTensor.reshape(tuple(list(self.reshapedShape[:-1])+[-1]))[None,:]
        newTensorSize=len(newTensor.shape)-1
        if epsilon==None: epsilon=self.ttEpsilon
        for coreIdx,core in enumerate(self.ttCores):
            if useHeuristics and (heuristicsToUse=='all' or heuristicsToUse=='occupancy'):
                if (coreIdx==len(self.ttCores)-1) or self.coreOccupancy[coreIdx]>0.8:
                    if coreIdx!=len(self.ttCores)-1:
                        continue
            if coreIdx==len(self.ttCores)-1:
                if useHeuristics and (heuristicsToUse=='all' or heuristicsToUse=='subselect'):
                    self.ttCores[coreIdx]=tempCore[:,:,None]
                    break #If data is subselected break at update of the last core and update the last core outside with projection!!
                newTensor=self.ttCores[coreIdx-1].reshape(-1,self.ttCores[coreIdx-1].shape[-1]).T@newTensor.reshape((newTensor.shape[0]*self.ttCores[coreIdx-1].shape[1],-1))
                self.ttCores[coreIdx]=np.concatenate((tempCore,newTensor),axis=1)[:,:,None]
                break ## Bu if clause'un ici kopyalandi, bug olma ihtimali yuksek!
            elif coreIdx==0:
                Ri=(np.eye(np.prod(core.shape[:-1]))-core.reshape(np.prod(core.shape[:-1]),-1)@core.reshape(np.prod(core.shape[:-1]),-1).T)@newTensor.reshape((np.prod(newTensor.shape[:2]),-1))
            else:
                newTensor=self.ttCores[coreIdx-1].reshape(-1,core.shape[0]).T@newTensor.reshape((newTensor.shape[0]*self.ttCores[coreIdx-1].shape[1],-1))
                Ri=(np.eye(np.prod(core.shape[:-1]))-core.reshape(np.prod(core.shape[:-1]),-1)@core.reshape(np.prod(core.shape[:-1]),-1).T)@newTensor.reshape(np.prod(self.ttCores[coreIdx].shape[:-1]),-1)
            URi,_,_=deltaSVD(Ri,tenNorm,newTensorSize,epsilon)
            self.ttCores[coreIdx]=np.hstack((core.reshape(np.prod(core.shape[:-1]),-1),URi)).reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            if coreIdx!=len(self.ttCores)-2:
                Uip1=self.ttCores[coreIdx+1].reshape(np.prod(self.ttCores[coreIdx+1].shape[:-1]),-1)
                Uip1=np.concatenate((Uip1,np.zeros(((self.ttCores[coreIdx].shape[-1]-self.ttCores[coreIdx+1].shape[0])*self.ttCores[coreIdx+1].shape[1],self.ttCores[coreIdx+1].shape[-1]))),axis=0)
                self.ttCores[coreIdx+1]=Uip1.reshape(self.ttCores[coreIdx].shape[-1],self.ttCores[coreIdx+1].shape[1],-1)
            else:
                tempCore=self.ttCores[coreIdx+1].squeeze()
                tempCore=np.concatenate((tempCore,np.zeros((URi.shape[1],tempCore.shape[1]))),axis=0)

        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def incrementalUpdateOld(self,newTensor,epsilon=None,tenNorm=None,heuristicsToUse='all'):
        newTensorSize=len(self.reshapedShape)
        if tenNorm==None:
            tenNorm=np.linalg.norm(newTensor)
        if epsilon==None: epsilon=self.ttEpsilon
        for coreIdx, core in enumerate(self.ttCores):
            if heuristicsToUse=='all' and (coreIdx==len(self.ttCores)-1 or self.coreOccupancy[coreIdx]>0.8):
                continue      
            tempData=self.projectData(newTensor,upTo=coreIdx+1)
            tempData=self.reconstruct(tempData,upTo=coreIdx+1).reshape(tuple(list(self.originalShape[:-1])+[-1]))
            tempData=newTensor-tempData
            tempData=self.projectData(tempData,upTo=coreIdx).reshape(np.prod(self.ttCores[coreIdx].shape[:2]),-1)
            u,_,_=deltaSVD(tempData,tenNorm,newTensorSize,eps=epsilon)
            tempCore=np.concatenate((self.ttCores[coreIdx].reshape(np.prod(self.ttCores[coreIdx].shape[:-1]),-1), u),axis=1)
            self.ttCores[coreIdx]=tempCore.reshape(tuple(list(self.ttCores[coreIdx].shape[:2])+[-1]))
            self.ttRanks[coreIdx+1]=self.ttCores[coreIdx].shape[-1]
            tempCore=self.ttCores[coreIdx+1].reshape(np.prod(self.ttCores[coreIdx+1].shape[:-1]),-1)
            if coreIdx!=len(self.ttCores)-2:
                v=np.zeros((u.shape[-1]*self.ttCores[coreIdx+1].shape[1],self.ttCores[coreIdx+1].shape[-1]))
            else:
                tempCore=self.ttCores[coreIdx+1].squeeze()
                self.ttCores[coreIdx+1]=np.concatenate((tempCore,np.zeros((self.ttCores[coreIdx].shape[-1]-self.ttCores[coreIdx+1].shape[0],self.ttCores[coreIdx+1].shape[1]))),axis=0).reshape(-1,self.ttCores[coreIdx+1].shape[1],1)
                break
            tempCore=np.concatenate((tempCore,v),axis=0)
            self.ttCores[coreIdx+1]=tempCore.reshape(self.ttCores[coreIdx].shape[-1],self.ttCores[coreIdx+1].shape[1],-1)
        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def computeRelError(self,data:np.array):
        elementwiseNorm=np.linalg.norm(data,axis=0)
        for idx in range(len(data.shape)-2):
            elementwiseNorm=np.linalg.norm(elementwiseNorm,axis=0)
        projectedData=self.projectData(data)
        reconstructedData=self.reconstruct(projectedData).reshape(data.shape)
        difference=data-reconstructedData
        differenceNorm=np.linalg.norm(difference,axis=0)
        for idx in range(len(difference.shape)-2):
            differenceNorm=np.linalg.norm(differenceNorm,axis=0)
        relError=differenceNorm/elementwiseNorm
        return relError
    
    def computeRecError(self,data:np.array,start=None,finish=None):#computes relative error by reconstructing data
        rec=self.reconstruct(self.ttCores[-1][:,start:finish,:]).reshape(data.shape)
        elementwiseNorm=np.linalg.norm(data,axis=0)
        for idx in range(len(data.shape)-2):
            elementwiseNorm=np.linalg.norm(elementwiseNorm,axis=0)
        difference=data-rec
        differenceNorm=np.linalg.norm(difference,axis=0)
        for idx in range(len(difference.shape)-2):
            differenceNorm=np.linalg.norm(differenceNorm,axis=0)
        recError=differenceNorm/elementwiseNorm
        return recError
    
    

    ###  TODO: write code for TT rounding ###
    def rounding(self,norm,epsilon=0):
        d=[core.shape[1] for core in self.ttCores]
        for coreIdx in np.arange(len(self.ttCores))[::-1]:
            currCore=self.ttCores[coreIdx]
            currCore=currCore.reshape(currCore.shape[0],-1) ## Compute mode 1 unfolding of the tt-core
            q,r=np.linalg.qr(currCore.T) ## Using the transpose results in a row orthogonal Q matrix !!!
            q,r=q.T,r.T
            self.ttCores[coreIdx]=q
            self.ttCores[coreIdx-1]=np.tensordot(self.ttCores[coreIdx-1],r,axes=(-1,0)) #contract previous tt-core and R matrix
            if coreIdx==1:
                break
                # pass ##TODO: write the case for the last core here.
        ### Compression of the orthogonalized representation ###
        ranks=[1]
        for coreIdx in range(len(self.ttCores)-1):
            core=self.ttCores[coreIdx]
            core=core.reshape(np.prod(core.shape[:-1]),-1)
            self.ttCores[coreIdx],sigma,v=deltaSVD(core,norm,len(self.ttCores),epsilon)
            ranks.append(self.ttCores[coreIdx].shape[-1])
            self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(ranks[coreIdx],d[coreIdx],-1) #fold matrices into tt-cores
            self.ttCores[coreIdx+1]=((np.diag(sigma)@v)@self.ttCores[coreIdx+1]).reshape(ranks[-1]*d[coreIdx+1],-1)
        self.ttCores[-1]=self.ttCores[-1].reshape(-1,d[-1],1)
        self.ttRanks=[1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None
    
    ### TODO: write dot product for tensor in TT format ### 
    @staticmethod
    def ttDot(tt1,tt2):
        if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            if isinstance(tt2,list):
                tt2=ttObject(tt2)
            if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
                raise AttributeError('One of the passed objects is not in TT-format!')
        v=np.kron(tt1.ttCores[0][:,0,:],tt2.ttCores[0][:,0,:])
        for i1 in range(1,tt1.ttCores[0].shape[1]):
            v+=np.kron(tt1.ttCores[0][:,i1,:],tt2.ttCores[0][:,i1,:])
        for coreIdx in range(1,len(tt1.ttCores)):
            p=[]
            for ik in range(tt1.ttCores[coreIdx].shape[1]):
                p.append(v@(np.kron(tt1.ttCores[coreIdx][:,ik,:],tt2.ttCores[coreIdx][:,ik,:])))
            v=np.sum(p,axis=0)
        return v.item()
    @staticmethod
    def ttNorm(tt1):
        if not isinstance(tt1,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            else:
                raise AttributeError('Passed object is not in TT-format!')
        try:
            norm=ttObject.ttDot(tt1,tt1)
            norm=np.sqrt(norm)
        except MemoryError:
            norm=np.linalg.norm(tt1.ttCores[-1])
        return norm

    ### TODO: write ITTD for comparison ###
    @staticmethod
    def ITTD(tt1,tt2,rounding=True,epsilon=0): ## Ensure that tt1 and tt2 are instances of ttObject
        if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
            if isinstance(tt1,list):
                tt1=ttObject(tt1)
            if isinstance(tt2,list):
                tt2=ttObject(tt2)
            if not isinstance(tt1,ttObject) or not isinstance(tt2,ttObject):
                raise AttributeError('One of the passed objects is not in TT-format!')
        tt1.ttCores[-1]=tt1.ttCores[-1].squeeze(-1)
        tt2.ttCores[-1]=tt2.ttCores[-1].squeeze(-1)
        rank1,numObs1=tt1.ttCores[-1].shape
        rank2,numObs2=tt2.ttCores[-1].shape
        tt1.ttCores[-1]=np.concatenate((tt1.ttCores[-1],np.zeros((rank1,numObs2))),axis=1)
        tt2.ttCores[-1]=np.concatenate((np.zeros((rank2,numObs1)),tt2.ttCores[-1]),axis=1)
        ## Addition in tt-format ##
        for coreIdx in range(len(tt1.ttCores)):
            if coreIdx==0:
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx].squeeze(0),tt2.ttCores[coreIdx].squeeze(0)),axis=1)[None,:,:]
            elif coreIdx == len(tt1.ttCores)-1:
                # tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx].squeeze(-1),tt2.ttCores[coreIdx].squeeze(-1)),axis=0)[:,:,None]
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],tt2.ttCores[coreIdx]),axis=0)[:,:,None]
            else:
                s11,s12,s13=tt1.ttCores[coreIdx].shape
                s21,s22,s23=tt2.ttCores[coreIdx].shape
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],np.zeros((s11,s12,s23))),axis=2)
                tt2.ttCores[coreIdx]=np.concatenate((np.zeros((s21,s22,s13)),tt2.ttCores[coreIdx]),axis=2)
                tt1.ttCores[coreIdx]=np.concatenate((tt1.ttCores[coreIdx],tt2.ttCores[coreIdx]),axis=0)
        tt1.ttRanks=[1]
        for core in tt1.ttCores:
            tt1.ttRanks.append(core.shape[-1])
        if rounding:
            tenNorm=ttObject.ttNorm(tt1)
            tt1.rounding(tenNorm,epsilon=epsilon)
        return tt1