from ast import Pass, Try
from logging import warning
import numpy as np
from discord_webhook import DiscordWebhook as dcmessage
from numpy import core
import warnings
from pickle import dump,load
from datetime import datetime
import time


def arrcheck(A):
    if isinstance(A,np.ndarray):
        return A
    else:
        return np.array(A,order='F')

def dot(X,Y):
    X=arrcheck(X)
    Y=arrcheck(Y)
    X_dim=X.shape
    Y_dim=Y.shape
    if len(X_dim)==len(Y_dim):
        dim=1
        for x,y in zip(X_dim,Y_dim):
            if x!=y:
                raise ValueError(f"Dimension mismatch in tensors at dim={dim}: Tensor 1={x}, Tensor 2={y}!!")
            dim+=1
    else:
        raise ValueError(f"Tensor 1 is {len(X_dim)}-Way, Tensor 2 is {len(Y_dim)}-Way!!")
    result=X*Y
    for _ in range(len(X_dim)):
        result=sum(result)
    return result

def norm(X):
    return(pow(dot(X,X),0.5))


def unfold(tensor, mode ,mult_mode=False):
    """Unfolds a tensors following the Kolda and Bader definition

        Moves the `mode` axis to the beginning and reshapes in Fortran order

        found at https://stackoverflow.com/questions/49970141/using-numpy-reshape-to-perform-3rd-rank-tensor-unfold-operation
        on 09.21.2020 19.01 GMT+3
    """
    tens_ndim=tensor.ndim

    orig_shape=tensor.shape #original tensor's shape just in case it's needed
    if mode>tens_ndim:
        raise ValueError("Mode exceeds tensor dimension!!")
    elif mode<=0:
        raise ValueError("Mode must be nonnegative!!")
    else:
        mode=mode-1
    old_indices = [ii for ii in range(tens_ndim) if ii != mode]
    num_cols_mat = np.prod([tensor.shape[ii] for ii in range(tens_ndim) if ii != mode])
    num_rows_mat = tensor.shape[mode]
    new_indices = [mode] + old_indices
    new_shape = [tensor.shape[mode]] + [tensor.shape[ii] for ii in range(tens_ndim) if ii != mode]
    unfolded_tensor = np.transpose(tensor, new_indices).reshape(num_rows_mat, num_cols_mat,order="F")

    if mult_mode:
        return unfolded_tensor,new_shape,new_indices
    else:
        return unfolded_tensor
def fold(tensor, new_shape,new_indices, mode):
    tens_folded=tensor.reshape(new_shape,order="F")
    return np.transpose(tens_folded,new_indices)

def tenprod(X,U,n):
    """
    n-Mode multiplication between a tensor X and a matrix/vector U

    """
    X=arrcheck(X)
    U=arrcheck(U)
    if len(U.shape)==2:
        matrix=True
    elif len(U.shape)>2:
        raise TypeError("U is neither matrix nor vector!!")
    else:
        matrix=False
    if matrix:
        if X.shape[n-1]!=U.shape[-1]:
            raise ValueError("Dimension mismatch between tensor A and matrix U!!")
        X,new_shape,perm_indices=unfold(X,n,mult_mode=True)
        Y_unfolded=np.dot(U,X)
        new_shape[0]=U.shape[0]
        Y_folded=fold(Y_unfolded,new_shape,perm_indices,n)
    else:
        U=np.transpose(U) if U.shape[0]!=1 else U
        if X.shape[n-1]!=U.shape[1]:
            raise ValueError("Dimension mismatch, please check mode!!")
        new_shape=list(X.shape)
        new_shape.pop(n-1)
        tuple(new_shape)
        Y_folded=np.matmul(U,unfold(X,n)).reshape(new_shape,order="F")
    return Y_folded


def kron(A,B):
    if (len(np.shape(A)) != 2) or (len(np.shape(B)) != 2):
        raise TypeError("At least one of the inputs supplied is not a matrix (2x2)")
    a_rows=np.shape(A)[0]
    b_rows=np.shape(B)[0]
    a_cols=np.shape(A)[1]
    b_cols=np.shape(B)[1]

    out=np.zeros(((a_rows*b_rows),(a_cols*b_cols)))
    for y in range(a_rows):
        for x in range(a_cols):
            out[y*b_rows:(y+1)*b_rows,x*b_cols:(x+1)*b_cols]=A[y,x]*B
    return out

def khatrirao(A,B):
    if (len(np.shape(A)) != 2) or (len(np.shape(B)) != 2):
        raise TypeError("At least one of the inputs supplied is not a matrix (2x2)")
    a_rows=np.shape(A)[0]
    b_rows=np.shape(B)[0]
    a_cols=np.shape(A)[1]
    b_cols=np.shape(B)[1]
    if a_cols!=b_cols:
        raise TypeError("Dimension mismatch in columns!!")
    out=np.zeros((a_rows*b_cols,a_cols))
    for x in range(a_cols):
        print(kron(a[:,x].reshape((-1,1)),b[:,x].reshape((-1,1))).shape)
        print(out[:,x*b_cols:(x+1)*b_cols].shape)
        out[:,x*b_cols:(x+1)*b_cols]=kron(a[:,x].reshape((-1,1)),b[:,x].reshape((-1,1)))
    return out

def haddamard(A,B):
    if A.shape!=B.shape:
        raise TypeError("Dimension mismatch in matrices!!")
    return A*B

def ttsvd(A,eps=0.1):
    #tensor train decomposition using svd
    input_shape=A.shape
    dims=len(A.shape)
    delta=(eps/((dims-1)**(0.5)))*np.linalg.norm(A)
    r=[1]
    cores=[]
    for k in range(dims-1):
        now=datetime.now()
        timestamp=now.strftime("%Y%m%d_%H%M")
        print(f'k:{k}  {timestamp}')
        nk=input_shape[k]
        A=A.reshape((r[k]*nk,int(np.prod(A.shape)/(r[k]*nk))),order='F')
        A=da.from_array(A)
        uda,sda,vhda=da.linalg.svd_compressed(A,k=1e100)
        sigma=sda.compute()
        svallist=list(sigma*sigma)
        svallist.reverse()
        truncpost=[idx for idx , element in enumerate(np.cumsum(svallist)) if element<=delta**2] 
        u=uda.compute()
        del uda
        cores.append(u[:,:r[k+1]].reshape((r[k],nk,r[k+1]),order='F'))
        del u
        vh=vhda.compute()
        del vhda, sda
        A=np.zeros_like(vh[:r[k+1],:])
        for idx,si in enumerate(sigma[0:r[k+1]]):
            A[idx,:]= si*vh[idx,:]
        del vh
    r.append(1)
    cores.append(A.reshape((r[-2],input_shape[-1],r[-1]),order='F'))
    return r,cores

# @profile
def ttsvd2(data,dataNorm,eps=0.1,dtype=np.float32):
    inputShape=data.shape
    dimensions=len(data.shape)
    delta=(eps/((dimensions-1)**(0.5)))*dataNorm
    ranks=[1]
    cores=[]
    for k in range(dimensions-1):
        now=datetime.now()
        timestamp=now.strftime("%Y%m%d_%H%M")
        nk=inputShape[k]
        data=data.reshape(ranks[k]*nk,int(np.prod(data.shape)/(ranks[k]*nk)),order='F').astype(dtype)
        
        svdTime=time.time()
        u,s,v=np.linalg.svd(data,False,True)
        slist=list(s*s)
        slist.reverse()
        truncpost=[idx for idx , element in enumerate(np.cumsum(slist)) if element<=delta**2] 
        ranks.append(len(s)-len(truncpost))

        u=u[:,:ranks[-1]]
        
        cores.append(u.reshape(ranks[k],nk,ranks[k+1],order='F'))
        data=np.zeros_like(v[:ranks[-1],:])
        for idx,sigma in enumerate(s[:ranks[-1]]):
            data[idx,:]=sigma*v[idx,:]#.compute() #if you delete v=v[:ranks[-1],:].compute(), activate the .compute() portion!!!
        
    ranks.append(1)
    cores.append(data.reshape(ranks[-2],inputShape[-1],ranks[-1],order='F'))
    return ranks,cores
# @profile
def choo_choo(cores):
    # Function for converting tt-cores to full size tensors
    for coreIdx in range(len(cores)-1):
        if coreIdx==0:
            coreProd=np.tensordot(cores[coreIdx],cores[coreIdx+1],axes=(-1,0))
        else:
            coreProd=np.tensordot(coreProd,cores[coreIdx+1],axes=(-1,0))
    coreProd=coreProd.reshape(coreProd.shape[1:-1])
    return coreProd
    
def primes(n):
    #function for finding prime factors for a given number, used for calculating the reshapings
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

def deltaSVD(data,dataNorm,dimensions,eps=0.1):
    #perform delta-truncated svd similar to that of the ttsvd algorithm
    delta=(eps/((dimensions-1)**(0.5)))*dataNorm
    try:
        u,s,v=np.linalg.svd(data,False,True)
    except np.linalg.LinAlgError:
        print("Numpy svd did not converge, using qr+svd")
        q,r=np.linalg.qr(data)
        u,s,v=np.linalg.svd(r)
        u=q@u        
    slist=list(s*s)
    slist.reverse()
    truncpost=[idx for idx , element in enumerate(np.cumsum(slist)) if element<=delta**2] 
    truncationRank=len(s)-len(truncpost)
    return u[:,:truncationRank],s[:truncationRank],v[:truncationRank,:]

class ttObject:
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
            self.imageWidth=self.originalShape[0]
            self.imageHeight=self.originalShape[1]
            self.imageDepth=self.originalShape[2]
            self.nElements=None
            self.imageCount=np.prod(self.originalShape[3:])
            if transposeMap==None:
                self.transposeMap=[idx for idx in range(len(data.shape))]
            else:
                self.transposeMap=transposeMap
        elif self.inputType==list: #if you are starting with compressed data (tt form) you will give list of arrays as input at the class initializatio
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
            self.imageWidth=84
            self.imageHeight=84
            self.imageDepth=3
            if transposeMap==None:
                self.transposeMap=[idx for idx in range(len(data))]
            else:
                self.transposeMap=transposeMap

        else:
            raise TypeError("Unknown input type!")
        # These attributes will be used later for other functionalities, do not hesitate to suggest anything you find useful
        self.singleReshapedImage=(self.imageWidth,self.imageHeight,self.imageDepth)
        self.method=method
        
        self.crossVerbose=True
        self.crossRankAdapt=False
        self.maxCrossIter=crossIter
        self.recoveryIndices=None
        self.imageCount=np.prod(self.originalShape[3:])
        self.imageOrder=imageOrder
        self.compressionTime=None

    def recoveryMap(self):
        #function used to recover single images from AG's C3 compression tool, will most likely be deleted
        primaryIndex=[]
        secondaryIndex=[]
        tertiaryIndex=[]
        for i in range(self.imageWidth):
            primaryIndex+=[i]*self.imageDepth*self.imageHeight
        for i in range(self.imageHeight):
            secondaryIndex+=[i]*self.imageDepth
        secondaryIndex=secondaryIndex*self.imageWidth
        for i in range(self.imageDepth):
            tertiaryIndex+=[i]
        tertiaryIndex=tertiaryIndex*self.imageWidth*self.imageHeight
        recoveryMap=np.concatenate((np.concatenate((np.array(primaryIndex).reshape(-1,1),np.array(secondaryIndex).reshape(-1,1)),axis=1),np.array(tertiaryIndex).reshape(-1,1)),axis=1)
        return recoveryMap
    def ttDecomp2(self,norm,dtype=np.float32):
        startTime=time.time()
        self.ttRanks,self.ttCores=ttsvd2(self.originalData,norm,self.ttEpsilon,dtype=dtype)
        self.compressionTime=time.time()-startTime
        self.nCores=len(self.ttCores)
        self.nElements=0
        for cores in self.ttCores:
            self.nElements+=np.prod(cores.shape)
        return None
    def ttDecomp(self):
        #compression function, you will prepare most of the inputs/options to this function while initializing the object
        if self.method=="ttsvd":
            startTime=time.time()
            self.ttRanks,self.ttCores=ttsvd(self.originalData,self.ttEpsilon)
            self.compressionTime=time.time()-startTime
            self.nCores=len(self.ttCores)
            self.nElements=0
            # for cores in self.ttCores:
            #     self.nElements+=np.prod(cores.shape)
        elif self.method=="ttcross":
            self.recoveryIndices=self.recoveryMap()
            self.ttCores=c3py.TensorTrain.cross_from_numpy_tensor(self.originalData,self.ttRanks,self.crossVerbose,self.crossRankAdapt,maxiter=self.maxCrossIter)
            self.nCores=len(self.ttCores.cores)
            self.nElements=0
            self.ttCores=self.ttCores.cores
            # for cores in self.ttCores.cores:
            #     self.nElements+=np.prod(cores.shape)
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

    def imageIndices(self,imageNumber:int):
        # This is a function used for image recoveries usikng AG's C3 library. Not needed for now.
        imageNumber=[imageNumber]
        mappedIndex=self.indexMap(self.originalShape,self.reshapedShape,imageNumber,self.singleReshapedImage,self.transposeMap)
        indexLength=len(mappedIndex[3:])
        imageIndex=[mappedIndex[3:]]*self.imageHeight*self.imageDepth*self.imageWidth
        imageIndex=np.array(imageIndex).reshape(-1,indexLength)
        indices=np.concatenate((self.recoveryIndices,imageIndex),axis=1)
        return indices
    def primeReshaping(self,ordered=False,full:bool=False):
        # This is a function used for create reshapings with the prime factorizations
        originalImageShape=(self.imageWidth,self.imageHeight,self.imageDepth)
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
        imageArray=choo_choo(imageCores)
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
    
    def returnOriginalImage(self,imageIndex):
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