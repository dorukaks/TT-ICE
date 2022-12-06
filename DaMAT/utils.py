""" This file contains the utility functions for running TT-ICE pack"""

import numpy as np


def primes(n):
    # function for finding prime factors for a given number,
    # used for calculating the reshapings
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def coreContraction(cores):
    # TODO: input checking

    # Function for converting tt-cores to full size tensors
    for coreIdx in range(len(cores) - 1):
        if coreIdx == 0:
            coreProd = np.tensordot(cores[coreIdx], cores[coreIdx + 1], axes=(-1, 0))
        else:
            coreProd = np.tensordot(coreProd, cores[coreIdx + 1], axes=(-1, 0))
    coreProd = coreProd.reshape(coreProd.shape[1:-1])
    return coreProd


def deltaSVD(data, dataNorm, dimensions, eps=0.1):
    """
    Performs delta-truncated SVD similar to that of the `TTSVD`_ algorithm.

    Given an unfolding matrix of a tensor, its norm and the number of dimensions
    of the original tensor, this function computes the truncated SVD of `data`.
    The truncation error is determined using the dimension dependant _delta_ formula
    from `TTSVD`_ paper.

    Parameters
    ----------
    data:obj:`numpy.array`
        Matrix for which the truncated SVD will be performed.
    dataNorm:obj:`float`
        Norm of the matrix. This parameter is used to determine the truncation bound.
    dimensions:obj:`int`
        Number of dimensions of the original tensor. This parameter is used to determine
        the truncation bound.
    eps:obj:`float`, optional
        Relative error upper bound for TT-decomposition.

    Returns
    -------
    u:obj:`numpy.ndarray`
        Column-wise orthonormal matrix of left-singular vectors. _Truncated_
    s:obj:`numpy.array`
        Array of singular values. _Truncated_
    v:obj:`numpy.ndarray`
        Row-wise orthonormal matrix of right-singular vectors. _Truncated_

    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286
    """

    # TODO: input checking

    delta = (eps / ((dimensions - 1) ** (0.5))) * dataNorm
    try:
        u, s, v = np.linalg.svd(data, False, True)
    except np.linalg.LinAlgError:
        print("Numpy svd did not converge, using qr+svd")
        q, r = np.linalg.qr(data)
        u, s, v = np.linalg.svd(r)
        u = q @ u
    slist = list(s * s)
    slist.reverse()
    truncpost = [
        idx for idx, element in enumerate(np.cumsum(slist)) if element <= delta**2
    ]
    truncationRank = len(s) - len(truncpost)
    return u[:, :truncationRank], s[:truncationRank], v[:truncationRank, :]


# def ttsvdLarge(A, eps=0.1):
#     """Tensor-Train decomposition using SVD for large matrices using dask serialization
#     check Oseledets 2011 for further information on ttsvd algorithm"""
#     input_shape = A.shape
#     dims = len(A.shape)
#     # delta = (eps / ((dims - 1) ** (0.5))) * np.linalg.norm(A)
#     r = [1]
#     cores = []
#     for k in range(dims - 1):
#         now = datetime.now()
#         timestamp = now.strftime("%Y%m%d_%H%M")
#         print(f"k:{k}  {timestamp}")
#         nk = input_shape[k]
#         A = A.reshape((r[k] * nk, int(np.prod(A.shape) / (r[k] * nk))), order="F")
#         A = da.from_array(A)
#         uda, sda, vhda = da.linalg.svd_compressed(A, k=1e100)
#         sigma = sda.compute()
#         svallist = list(sigma * sigma)
#         svallist.reverse()
#         # truncpost = [
#         #     idx
#         #     for idx, element in enumerate(np.cumsum(svallist))
#         #     if element <= delta**2
#         # ]
#         u = uda.compute()
#         del uda
#         cores.append(u[:, : r[k + 1]].reshape((r[k], nk, r[k + 1]), order="F"))
#         del u
#         vh = vhda.compute()
#         del vhda, sda
#         A = np.zeros_like(vh[: r[k + 1], :])
#         for idx, si in enumerate(sigma[0 : r[k + 1]]):
#             A[idx, :] = si * vh[idx, :]
#         del vh
#     r.append(1)
#     cores.append(A.reshape((r[-2], input_shape[-1], r[-1]), order="F"))
#     return r, cores


def ttsvd(data, dataNorm, eps=0.1, dtype=np.float32):
    inputShape = data.shape
    dimensions = len(data.shape)
    delta = (eps / ((dimensions - 1) ** (0.5))) * dataNorm
    ranks = [1]
    cores = []
    for k in range(dimensions - 1):
        nk = inputShape[k]
        data = data.reshape(
            ranks[k] * nk, int(np.prod(data.shape) / (ranks[k] * nk)), order="F"
        ).astype(dtype)
        u, s, v = np.linalg.svd(data, False, True)
        slist = list(s * s)
        slist.reverse()
        truncpost = [
            idx for idx, element in enumerate(np.cumsum(slist)) if element <= delta**2
        ]
        ranks.append(len(s) - len(truncpost))

        u = u[:, : ranks[-1]]

        cores.append(u.reshape(ranks[k], nk, ranks[k + 1], order="F"))
        data = np.zeros_like(v[: ranks[-1], :])
        for idx, sigma in enumerate(s[: ranks[-1]]):
            data[idx, :] = sigma * v[idx, :]

    ranks.append(1)
    cores.append(data.reshape(ranks[-2], inputShape[-1], ranks[-1], order="F"))
    return ranks, cores
