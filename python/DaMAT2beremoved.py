''' File for things that i am not sure to remove... '''

import numpy as np


def arrcheck(A):
    ''' This simple function checks if input object A is an numpy array, it's repetitively used in other functions. May be removed to a separate file   '''
    if isinstance(A,np.ndarray):
        return A
    else:
        return np.array(A,order='F')

def dot(X,Y):
    ''' This function performs a dot product between 2 numpy arrays. May be removed entirely    '''
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
    ''' Computes the Frobenius norm of X '''
    return(pow(dot(X,X),0.5))


def unfold(tensor, mode ,mult_mode=False): 
    """Unfolds a tensors following the Kolda and Bader definition

        Moves the `mode` axis to the beginning and reshapes in Fortran order

        found at https://stackoverflow.com/questions/49970141/using-numpy-reshape-to-perform-3rd-rank-tensor-unfold-operation
        on 09.21.2020 19.01 GMT+3

        Will be removed to another file if not used
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

def coreContraction(cores):
    # Function for converting tt-cores to full size tensors
    for coreIdx in range(len(cores)-1):
        if coreIdx==0:
            coreProd=np.tensordot(cores[coreIdx],cores[coreIdx+1],axes=(-1,0))
        else:
            coreProd=np.tensordot(coreProd,cores[coreIdx+1],axes=(-1,0))
    coreProd=coreProd.reshape(coreProd.shape[1:-1])
    return coreProd

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


    # def imageIndices(self,imageNumber:int):
    #     ''' This is a function used for image recoveries usikng AG's C3 library. 
    #     Not needed for now. Likely to be removed
    #     '''
    #     imageNumber=[imageNumber]
    #     mappedIndex=self.indexMap(self.originalShape,self.reshapedShape,imageNumber,self.singleReshapedImage,self.transposeMap)
    #     indexLength=len(mappedIndex[3:])
    #     imageIndex=[mappedIndex[3:]]*self.imageHeight*self.imageDepth*self.imageWidth
    #     imageIndex=np.array(imageIndex).reshape(-1,indexLength)
    #     indices=np.concatenate((self.recoveryIndices,imageIndex),axis=1)
    #     return indices