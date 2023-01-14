# -*- coding: utf-8 -*-
"""
This is a python class for tensors in TT-format.
Through this object you can compute TT-decomposition of multidimensional arrays in
one shot using [TTSVD algorithm](https://epubs.siam.org/doi/epdf/10.1137/090752286)
or incrementally using [TT-ICE algorithm](https://arxiv.org/abs/2211.12487).

Furthermore, this object allows exporting and importing TT-cores using native
format (`.ttc`) and intermediary formats (`.txt`).
"""

import warnings
import time

import numpy as np
import cupy  as cp 

from logging import warning
from .utils import deltaSVD, ttsvdCuPy, deltaSVDCuPy
from pickle import dump, load


class ttObjectCP:
    """
    Python object for tensors in Tensor-Train format.

    This object computes the TT-decomposition of multidimensional arrays using `TTSVD`_,
    `TT-ICE`_, and `TT-ICE*`_.ttObjectttObject copy copy
    It furthermore contains an inferior method, ITTD, for benchmarking purposes.

    This object handles various operations in TT-format such as dot product and norm.
    Also provides supporting operations for uncompressed multidimensional arrays such as
    reshaping and transpose.
    Once the tensor train approximation for a multidimensional array is computed, you can
    compute projection of appropriate arrays onto the TT-cores, reconstruct projected
    or compressed tensors and compute projection/compression accuracy.

    You can also save/load tensors as `.ttc` or `.txt` files.

    Attributes
    ----------
    inputType: type
        Type of the input data. This determines how the object is initialized.
    originalData:
        Original multidimensional input array. Generally will not be stored
        after computing
        an initial set of TT-cores.
    method: :obj:`str`
        Method of computing the initial set of TT-cores. Currently only accepts
        `'ttsvd'` as input
    ttEpsilon: :obj:`float`
        Desired relative error upper bound.
    ttCores: :obj:`list` of :obj:`numpy.array`
        Cores of the TT-decomposition. Stored as a list of numpy arrays.
    ttRanks: :obj:`list` of :obj:`int`
        TT-ranks of the decomposition. Stored as a list of integers.
    nCores: :obj:`int`
        Number of TT-cores in the decomposition.
    nElements: :obj:`int`
        Number of entries present in the current `ttObject`.
    originalShape: :obj:`tuple` or :obj:`list`
        Original shape of the multidimensional array.
    reshapedShape: :obj:`tuple` or :obj:`list`
        Shape of the multidimensional array after reshaping. Note that
        this attribute is only meaningful before computing a TT-decomposition
    indexOrder: :obj:`list` of :obj:`int`
        Keeps track of original indices in case of transposing the input array.
    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286
        _TT-ICE:
        https://arxiv.org/abs/2211.12487
        _TT-ICE*:
        https://arxiv.org/abs/2211.12487

    """

    def __init__(
        self,
        data,
        epsilon: float = None,
        keepData: bool = False,
        samplesAlongLastDimension: bool = True,
        method: str = "ttsvd",
    ) -> None:
        """
        Initializes the ttObject.

        Parameters
        ----------
        data: :obj:`numpy.array` or :obj:`list`
            Main input to the ttObject. It can either be a multidimensional `numpy array`
            or `list of numpy arrays`.
            If list of numpy arrays are presented as input, the object will interpret it
            as the TT-cores of an existing decomposition.
        epsilon: :obj:`float`, optional
            The relative error upper bound desired for approximation. Optional for cases
            when `data` has type `list`.
        keepData: :obj:`bool`, optional
            Optional boolean variable to determine if the original array will be kept
            after compression. Set to `False` by default.
        samplesAlongLastDimension: :obj:`bool`, optional
            Boolean variable to ensure if the samples are stacked along the last
            dimension. Assumed to be `True` since it is one of the assumptions in TT-ICE.
        method: :obj:`str`, optional
            Determines the computation method for tensor train decomposition of
            the multidimensional array presented as `data`. Set to `'ttsvd'` by default.

            Currently the package only has support for ttsvd, additional support such as
            `ttcross` might be included in the future.

        """
        # self.ttRanks=ranks
        self.ttCores = None
        self.nCores = None
        self.nElements = None
        self.inputType = type(data)
        self.method = method
        self.keepOriginal = keepData
        self.originalData = data
        self.samplesAlongLastDimension = samplesAlongLastDimension
        if self.inputType is np.memmap or np.ndarray:
            self.inputType = cp.ndarray

        if self.inputType is cp.ndarray:
            self.ttEpsilon = epsilon
            self.originalShape = list(data.shape)
            self.reshapedShape = self.originalShape
            self.indexOrder = [idx for idx in range(len(self.originalShape))]
        elif self.inputType == list:
            self.nCores = len(data)
            self.ttCores = data
            self.reshapedShape = [core.shape[1] for core in self.ttCores]
        else:
            raise TypeError("Unknown input type!")

    @property
    def coreOccupancy(self) -> None:
        """
        :obj:`list` of :obj:`float`: A metric showing the *relative rank* of each
        TT-core. This metric is used for a heuristic enhancement tool in `TT-ICE*`
        algorithm
        """
        try:
            return [
                core.shape[-1] / np.prod(core.shape[:-1]) for core in self.ttCores[:-1]
            ]
        except ValueError:
            warnings.warn(
                "No TT cores exist, maybe forgot calling object.ttDecomp?", Warning
            )
            return None

    @property
    def compressionRatio(self) -> float:
        """
        :obj:`float`: A metric showing how much compression with respect to the
        original multidimensional array is achieved.
        """
        originalNumEl = 1
        compressedNumEl = 0
        for core in self.ttCores:
            originalNumEl *= core.shape[1]
            compressedNumEl += np.prod(core.shape)
        return originalNumEl / compressedNumEl

    def changeShape(self, newShape: tuple or list) -> None:
        """
        Function to change shape of input tensors and keeping track of the reshaping.
        Reshapes `originalData` and saves the final shape in `reshapedShape`

        Note
        ----
        A simple `numpy.reshape` would be sufficient for this purpose but in order to keep
        track of the shape changes the `reshapedShape` attribute also needs to be updated
        accordingly.

        Parameters
        ----------
        newShape:obj:`tuple` or `list`
            New shape of the tensor

        Raises
        ------
        warning
            If an attempt is done to modify the shape after computing a TT-decomposition.
            This is important since it will cause incompatibilities in other functions
            regarding the shape of the uncompressed tensor.

        """
        if self.ttCores is not None:
            warning(
                "Warning! You are reshaping the original data after computing a\
                TT-decomposition! We will proceed without reshaping self.originalData!!"
            )
            return None
        self.reshapedShape = newShape
        self.originalData = cp.reshape(self.originalData, self.reshapedShape)
        self.reshapedShape = list(self.originalData.shape)
        # Changed reshapedShape to a list for the trick in ttICEstar
        if self.samplesAlongLastDimension:
            self.singleDataShape = self.reshapedShape[:-1]
            # This line assumes we keep the last index as the samples index and don't
            # interfere with its shape

    def computeTranspose(self, newOrder: list) -> None:
        """
        Transposes the axes of `originalData`.

        Similar to `changeShape`, a simple
        `numpy.transpose` would be sufficient for this purpose but in order to
        keep track of the transposition order `indexOrder` attribute also needs
        to be updated accordingly.

        Parameters
        ----------
        newOrder:obj:`list`
            New order of the axes.

        Raises
        ------
        ValueError
            When the number of transposition axes are not equal to the number of
            dimensions of `originalData`.
        """
        assert self.inputType == cp.ndarray and self.ttCores is None
        if len(newOrder) != len(self.indexOrder):
            raise ValueError(
                "size of newOrder and self.indexOrder does not match. \
                    Maybe you forgot reshaping?"
            )
        self.indexOrder = [self.indexOrder[idx] for idx in newOrder]
        self.originalData = self.originalData.transpose(newOrder)
        self.reshapedShape = list(self.originalData.shape)

    def indexMap(self) -> None:
        """
        Function to map the original indices to the reshaped indices.
        Currently not implemented.
        """
        self.A = 2

    def primeReshaping(self) -> None:
        """
        Function to reshape the first d dimensions to prime factors.
        Currently not implemented
        """
        self.A = 2

    def saveData(
        self, fileName: str, directory="./", justCores=True, outputType="ttc"
    ) -> None:
        """
        Writes the computed TT-cores to an external file.

        Parameters
        ----------
        fileName:obj:`str`

        directory:obj:`str`
            Location to save files with respect to the present working directory.
        justCores:obj:`bool`
            Boolean variable to determine if `originalData` will be discarded
            or not while saving.
        outputType:obj:`str`
            Type of the output file. `ttc` for pickled `ttObject`, `txt` for
            individual text files for each TT-core.
        """
        if justCores:
            if outputType == "ttc":
                with open(directory + fileName + ".ttc", "wb") as saveFile:
                    temp = ttObjectCP(self.ttCores)
                    for attribute in vars(self):
                        if attribute != "originalData":
                            setattr(temp, attribute, eval(f"self.{attribute}"))
                    dump(temp, saveFile)
            elif outputType == "txt":
                for coreIdx, core in enumerate(self.ttCores):
                    cp.savetxt(
                        directory + f"{fileName}_{coreIdx}.txt",
                        core.reshape(-1, core.shape[-1]),
                        header=f"{core.shape[0]} {core.shape[1]} {core.shape[2]}",
                        delimiter=" ",
                    )
            elif outputType == "cpy":
                for coreIdx, core in enumerate(self.ttCores):
                    cp.save(
                        directory + f"{fileName}_{coreIdx}.cpy",
                        core,
                    )
            elif outputType == "npy":
                for coreIdx, core in enumerate(self.ttCores):
                    np.save(
                        directory + f"{fileName}_{coreIdx}.npy",
                        core,
                    )
            else:
                raise ValueError(f"Output type {outputType} is not supported!")
        else:
            if outputType == "txt":
                raise ValueError(
                    ".txt type outputs are only supported for justCores=True!!"
                )
            if self.method == "ttsvd":
                with open(directory + fileName + ".ttc", "wb") as saveFile:
                    dump(self, saveFile)
            else:
                raise ValueError("Unknown Method!")

    @staticmethod
    def loadData(fileName: str, numCores=None) -> "ttObject":
        """
        Loads data from a `.ttc` or `.txt` file


        Static method to load TT-cores into a ttObject object.
        Note
        ----
        If data is stored in {coreFile}_{coreIdx}.txt format,
        the input fileName should just be coreFile.txt

        Parameters
        ----------
        fileName:obj:`str`
            Name of the file that will be loaded.
        numCores:obj:`int`
            Number of cores that the resulting `ttObject` will have
            (Only required when input data format is `.txt`)
        """
        fileExt = fileName.split(".")[-1]
        if fileExt == "ttc":
            with open(fileName, "rb") as f:
                dataSetObject = load(f)
            return dataSetObject
        elif fileExt == "txt":
            if numCores is None:
                raise ValueError("Number of cores are not defined!!")
            fileBody = fileName.split(".")[0]
            coreList = []
            for coreIdx in range(numCores):
                with open(f"{fileBody}_{coreIdx}.{fileExt}"):
                    coreShape = f.readline()[2:-1]
                    coreShape = [int(item) for item in coreShape.split(" ")]
                coreList.append(
                    np.loadtxt(f"{fileBody}_{coreIdx}.{fileExt}").reshape[coreShape]
                )
            return coreList
        elif fileExt == "npy":
            if numCores is None:
                raise ValueError("Number of cores are not defined!!")
            fileBody = fileName.split(".")[0]
            coreList = []
            for coreIdx in range(numCores):
                coreList.append(np.load(f"{fileBody}_{coreIdx}.{fileExt}"))
        elif fileExt == "cpy":
            if numCores is None:
                raise ValueError("Number of cores are not defined!!")
            fileBody = fileName.split(".")[0]
            coreList = []
            for coreIdx in range(numCores):
                coreList.append(cp.load(f"{fileBody}_{coreIdx}.{fileExt}"))
        else:
            raise ValueError(f"{fileExt} files are not supported!")

    @staticmethod
    def ttDot(tt1, tt2) -> float():
        """
        Computes the dot product of two tensors in TT format.

        Parameters
        ----------
        tt1:obj:`ttObject`
            Tensor in TT-format
        tt2:obj:`ttObject`
            Tensor in TT-format

        Returns
        -------
        prod:obj:`float`
            Dot product of `tt1` and `tt2`.

        Raises
        ------
        AttributeError
            When either one of the input parameters are not `ttObject`s.

        """
        if not isinstance(tt1, ttObject) or not isinstance(tt2, ttObject):
            if isinstance(tt1, list):
                tt1 = ttObject(tt1)
            if isinstance(tt2, list):
                tt2 = ttObject(tt2)
            if not isinstance(tt1, ttObject) or not isinstance(tt2, ttObject):
                raise AttributeError("One of the passed objects is not in TT-format!")
        prod = np.kron(tt1.ttCores[0][:, 0, :], tt2.ttCores[0][:, 0, :])
        for i1 in range(1, tt1.ttCores[0].shape[1]):
            prod += np.kron(tt1.ttCores[0][:, i1, :], tt2.ttCores[0][:, i1, :])
        for coreIdx in range(1, len(tt1.ttCores)):
            p = []
            for ik in range(tt1.ttCores[coreIdx].shape[1]):
                p.append(
                    prod
                    @ (
                        np.kron(
                            tt1.ttCores[coreIdx][:, ik, :], tt2.ttCores[coreIdx][:, ik, :]
                        )
                    )
                )
            prod = np.sum(p, axis=0)
        return prod.item()

    @staticmethod
    def ttNorm(tt1) -> float:
        """
        Computes norm of a tensor in TT-format.

        Parameters
        ----------
        tt1:obj:`ttObject`
            Tensor in TT-format

        Returns
        -------
        norm:obj:`float`
            Norm of `tt1`.

        Raises
        ------
        AttributeError
            When the input parameter is not a `ttObject`.
        """
        if not isinstance(tt1, ttObject):
            if isinstance(tt1, list):
                tt1 = ttObject(tt1)
            else:
                raise AttributeError("Passed object is not in TT-format!")
        try:
            norm = ttObject.ttDot(tt1, tt1)
            norm = np.sqrt(norm)
        except MemoryError:
            norm = np.linalg.norm(tt1.ttCores[-1])
        return norm

    def projectTensor(self, newData: np.array, upTo=None) -> cp.array:
        """
        Projects tensors onto basis spanned by TT-cores.

        Given a tensor with appropriate dimensions, this function leverages
        the fact that TT-cores obtained through `TTSVD`_ and `TT-ICE`_ are
        column-orthonormal in the mode-2 unfolding.

        Note
        ----
        This function will not yield correct results if the TT-cores are
        not comprised of orthogonal vectors.

        Parameters
        ----------
        newData:obj:`np.aray`
            Tensor to be projected
        upTo:obj:`int`, optional
            Index that the projection will be terminated. If an integer is
            passed as this parameter, `newTensor` will be projected up to
            (not including) the core that has index `upTo`. Assumes 1-based
            indexing.

         .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
            _TT-ICE:
            https://arxiv.org/abs/2211.12487
            _TT-ICE*:
            https://arxiv.org/abs/2211.12487
        """
        if type(newData) is not cp.ndarray:
            newData=cp.asarray(newData)
        for coreIdx, core in enumerate(self.ttCores):
            if (coreIdx == len(self.ttCores) - 1) or coreIdx == upTo:
                break
            newData = (
                core.reshape(np.prod(core.shape[:2]), -1).transpose()
            ) @ newData.reshape(
                self.ttRanks[coreIdx] * self.ttCores[coreIdx].shape[1], -1
            )
        return newData
    
    ## TODO projectTensor cupy yaz
    def projectTensorCuPy(self, newData: np.array, upTo=None) -> np.array:
        for coreIdx, core in enumerate(self.ttCores):
            if (coreIdx == len(self.ttCores) - 1) or coreIdx == upTo:
                break
            # core=cp.asarray(core)
            newData = (
                # core.reshape(np.prod(core.shape[:2]), -1).transpose()
                cp.asarray(core).reshape(np.prod(core.shape[:2]), -1).transpose()
            ) @ newData.reshape(
                self.ttRanks[coreIdx] * self.ttCores[coreIdx].shape[1], -1
            )
        return newData

    def reconstruct(self, projectedData, upTo=None):
        """
        Reconstructs tensors using TT-cores.

        Assumes that `projectedData` is a slice from the last
        TT-core.
        While reconstructing any projected tensor from  `projectTensor`,
        this function leverages the fact that TT-cores obtained through
        `TTSVD`_ and `TT-ICE`_ are column-orthonormal in the mode-2
        unfolding.

        Note
        ----
        This function might not yield correct results for projected tensors
        if the TT-cores are not comprised of orthogonal vectors.

        Parameters
        ----------
        projectedData:obj:`np.aray`
            A tensor slice (or alternatively an array)
        upTo:obj:`int`, optional
            Index that the reconstruction will be terminated. If an integer is
            passed as this parameter, `projectedData` will be projected up to
            (not including) the core that has index `upTo`. Assumes 1-based
            indexing.
        .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
            _TT-ICE:
            https://arxiv.org/abs/2211.12487

        """
        if upTo is None:
            upTo = len(self.ttCores) - 1  # Write the core index in 1-indexed form!!
        for core in self.ttCores[:upTo][::-1]:
            projectedData = cp.tensordot(core, projectedData, axes=(-1, 0))
        return projectedData

    def reconstructCuPy(self, projectedData, upTo=None):
        if upTo is None:
            upTo = len(self.ttCores) - 1  # Write the core index in 1-indexed form!!
        for core in self.ttCores[:upTo][::-1]:
            projectedData = cp.tensordot(cp.asarray(core), projectedData, axes=(-1, 0))
        return projectedData

    def updateRanks(self) -> None:
        """
        Updates the ranks of the `ttObject` after incremental updates.
        """
        self.ttRanks = [1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def computeRelError(self, newTensor: np.array) -> np.array:
        """
        Computes relative error by projecting data onto TT-cores.

        This function computes the error induced by projecting `data` onto the TT-cores
        of `ttObject`. The last index of `newTensor` is assumed to be the number of
        individual observations.

        Note
        ----
        - In order to compute the projection onto the TT-cores, the dimensions of `data`
        should match that of `ttObject`.
        - If a single observation will be passed as `newTensor`, an additional
        index/dimension should be introduced either through reshaping or [:,None]

        Parameters
        ----------
            Tensor for which the projection error is computed

        Returns
        -------
        relError:obj:`np.array`
            Array of relative errors
        """
        elementwiseNorm = cp.linalg.norm(newTensor, axis=0)
        for _ in range(len(newTensor.shape) - 2):
            elementwiseNorm = cp.linalg.norm(elementwiseNorm, axis=0)
        if type(newTensor) is cp.ndarray:
            projectedData = self.projectTensor(newTensor)
            reconstructedData = self.reconstruct(projectedData).reshape(newTensor.shape)
        elif type(newTensor) is np.ndarray:
            projectedData = self.projectTensorCuPy(newTensor)
            reconstructedData = self.reconstructCuPy(projectedData).reshape(newTensor.shape)
        difference = newTensor - reconstructedData
        differenceNorm = cp.linalg.norm(difference, axis=0)
        for _ in range(len(difference.shape) - 2):
            differenceNorm = cp.linalg.norm(differenceNorm, axis=0)
        relError = differenceNorm / elementwiseNorm
        return relError

    def computeRecError(self, data: np.array, start=None, finish=None) -> None:
        """
        Function to compute relative error by reconstructing data from slices
        of TT-cores.
        Currently not implemented.
        """
        self.A = 2

    def ttDecomp(self, norm=None, dtype=cp.float32) -> "ttObject.ttCores":
        """
        Computes TT-decomposition of a multidimensional array using `TTSVD`_ algorithm.

        Currently only supports `ttsvd` as method. In the future additional formats may
        be covered.

        Parameters
        ----------
        norm:obj:`float`, optional
            Norm of the tensor to be compressed
        dtype:obj:`type`, optional
            Desired data type for the compression. Intended to allow lower precision
            if needed.

        Raises
        ------
        ValueError
            When `method` is not one of the admissible methods.


        The following attributes are modified as a result of this function:
        -------
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`

        .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
        """
        if norm is None:
            norm = cp.linalg.norm(self.originalData)
        if self.method == "ttsvd":
            startTime = time.time()
            self.ttRanks, self.ttCores = ttsvdCuPy(
                self.originalData, norm, self.ttEpsilon, dtype=dtype
            )
            self.compressionTime = time.time() - startTime
            self.nCores = len(self.ttCores)
            self.nElements = 0
            for cores in self.ttCores:
                self.nElements += np.prod(cores.shape)
            if not self.keepOriginal:
                self.originalData = None
            return None
        else:
            raise ValueError("Method unknown. Please select a valid method!")

    def ttICE(self, newTensor, epsilon=None, tenNorm=None) -> None:
        """
        `TT-ICE`_ algorithmn without any heuristic upgrades.

        Given a set of TT-cores, this function provides incremental updates
        to the TT-cores to approximate `newTensor` within a relative error
        defined in `epsilon`

        Note
        ----
        This algorithm/function relies on the fact that TT-cores are columnwise
        orthonormal in the mode-2 unfolding.

        Parameters
        ----------
        newTensor:obj:`np.array`
            New/streamed tensor that will be used to expand the orthonormal bases defined
            in TT-cores
        epsilon:obj:`float`, optional
            Relative error upper bound for approximating `newTensor` after incremental
            updates. If not defined, `ttObject.ttEpsilon` is used.
        tenNorm:obj:`float`, optional
            Norm of `newTensor`

        Notes
        -------
        **The following attributes are modified as a result of this function:**
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`
        .. _TT-ICE:
            https://arxiv.org/abs/2211.12487
        """
        if tenNorm is None:
            tenNorm = np.linalg.norm(newTensor)
        if epsilon is None:
            epsilon = self.ttEpsilon
        newTensorSize = len(newTensor.shape) - 1
        newTensor = newTensor.reshape(list(self.reshapedShape[:-1]) + [-1])[None, :]
        newTensor = newTensor.reshape(self.reshapedShape[0], -1)
        Ui = self.ttCores[0].reshape(self.reshapedShape[0], -1)
        Ri = newTensor - Ui @ (Ui.T @ newTensor)
        for coreIdx in range(0, len(self.ttCores) - 2):
            URi, _, _ = deltaSVD(Ri, tenNorm, newTensorSize, epsilon)
            self.ttCores[coreIdx] = np.hstack((Ui, URi)).reshape(
                self.ttCores[coreIdx].shape[0], self.reshapedShape[coreIdx], -1
            )
            self.ttCores[coreIdx + 1] = np.concatenate(
                (
                    self.ttCores[coreIdx + 1],
                    np.zeros(
                        (
                            URi.shape[-1],
                            self.reshapedShape[coreIdx + 1],
                            self.ttRanks[coreIdx + 2],
                        )
                    ),
                ),
                axis=0,
            )
            Ui = self.ttCores[coreIdx].reshape(
                self.ttCores[coreIdx].shape[0] * self.reshapedShape[coreIdx], -1
            )
            newTensor = (Ui.T @ newTensor).reshape(
                np.prod(self.ttCores[coreIdx + 1].shape[:-1]), -1
            )
            Ui = self.ttCores[coreIdx + 1].reshape(
                self.ttCores[coreIdx].shape[-1] * self.reshapedShape[coreIdx + 1], -1
            )
            Ri = newTensor - Ui @ (Ui.T @ newTensor)
        coreIdx = len(self.ttCores) - 2
        URi, _, _ = deltaSVD(Ri, tenNorm, newTensorSize, epsilon)
        self.ttCores[coreIdx] = np.hstack((Ui, URi))
        self.ttCores[coreIdx + 1] = np.concatenate(
            (
                self.ttCores[coreIdx + 1],
                np.zeros(
                    (
                        URi.shape[-1],
                        self.ttCores[coreIdx + 1].shape[1],
                        self.ttRanks[coreIdx + 2],
                    )
                ),
            ),
            axis=0,
        )
        newTensor = self.ttCores[coreIdx].T @ newTensor
        self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
            self.ttCores[coreIdx - 1].shape[-1], self.reshapedShape[coreIdx], -1
        )
        coreIdx += 1
        Ui = self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0], -1)
        self.ttCores[coreIdx] = np.hstack((Ui, newTensor)).reshape(
            self.ttCores[coreIdx].shape[0], -1, 1
        )
        self.updateRanks()
        return None

    def ttICEstar(
        self,
        newTensor: cp.array,
        epsilon: float = None,
        tenNorm: float = None,
        elementwiseNorm: np.array = None,
        elementwiseEpsilon: np.array = None,
        heuristicsToUse: list = ["skip", "subselect", "occupancy"],
        occupancyThreshold: float = 0.8,
        simpleEpsilonUpdate: bool = False,
    ) -> None:
        """
        `TT-ICE*`_ algorithmn with heuristic performance upgrades.

        Given a set of TT-cores, this function provides incremental updates
        to the TT-cores to approximate `newTensor` within a relative error
        defined in `epsilon`.

        Note
        ----
        This algorithm/function relies on the fact that TT-cores are columnwise
        orthonormal in the mode-2 unfolding.

        Parameters
        ----------
        newTensor:obj:`np.array`
            New/streamed tensor that will be used to expand the orthonormal bases defined
            in TT-cores
        epsilon:obj:`float`, optional
            Relative error upper bound for approximating `newTensor` after incremental
            updates. If not defined, `ttObject.ttEpsilon` is used.
        tenNorm:obj:`float`, optional
            Norm of `newTensor`.
        elementwiseNorm:obj:`np.array`, optional
            Individual norms of the observations in `newTensor`.
        elementwiseEpsilon:obj:`np.array`, optional
            Individual relative projection errors of the observations in `newTensor`.
        heuristicsToUse:obj:`list`, optional
            List of heuristics to use while updating TT-cores. Currently only accepts
            `'skip'`, `'subselect'`, and `'occupancy'`.
        occupancyThreshold:obj:`float`, optional
            Threshold determining whether to skip updating a single core or not. Not used
            if `'occupancy'` is not in `heuristicsToUse`
        simpleEpsilonUpdate:obj:`bool`, optional
            Uses the simple epsilon update equation. *Warning*: this relies on the
            assumption that all observations in `newTensor` have similar norms.

        Notes
        -------
        **The following attributes are modified as a result of this function:**
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`
        .. _TT-ICE*:
            https://arxiv.org/abs/2211.12487
        """
        if epsilon is None:
            epsilon = self.ttEpsilon
        if ("subselect" in heuristicsToUse) and (newTensor.shape[-1] == 1):
            warning(
                "The streamed tensor has only 1 observation in it. \
                    Subselect heuristic will not be useful!!"
            )
        # print("a")
        newTensor = newTensor.reshape(list(self.reshapedShape[:-1]) + [-1])[None, :]
        if type(newTensor) is not cp.ndarray:
            newTensor=cp.asarray(newTensor)
        updEpsilon = epsilon
        newTensorSize = len(newTensor.shape) - 1
        # print("b")

        if "skip" in heuristicsToUse and elementwiseEpsilon is None:
            elementwiseEpsilon = self.computeRelError(newTensor)
        if "skip" in heuristicsToUse:
            if np.mean(elementwiseEpsilon).item() <= epsilon:
                newTensor = self.projectTensor(newTensor)
                self.ttCores[-1] = cp.hstack(
                    (self.ttCores[-1].reshape(self.ttRanks[-2], -1), newTensor)
                ).reshape(self.ttRanks[-2], -1, 1)
                return None
        if tenNorm is None and elementwiseNorm is None:
            tenNorm = np.linalg.norm(newTensor).item()
        elif tenNorm is None:
            tenNorm = np.linalg.norm(elementwiseNorm).item()
        # print("c")

        select = [True] * newTensor.shape[-1]
        discard = [False] * newTensor.shape[-1]
        if "subselect" in heuristicsToUse:
            select = elementwiseEpsilon > epsilon
            discard = elementwiseEpsilon <= epsilon
            if simpleEpsilonUpdate:
                updEpsilon = (
                    epsilon * newTensor.shape[-1]
                    - np.mean(elementwiseEpsilon[discard]) * discard.sum()
                ) / (select.sum())
            else:
                if elementwiseNorm is None:
                    elementwiseNorm = np.linalg.norm(newTensor, axis=0)
                    for _ in range(len(self.ttCores) - 1):
                        elementwiseNorm = np.linalg.norm(elementwiseNorm, axis=0)
                allowedError = (self.ttEpsilon * np.linalg.norm(elementwiseNorm)) ** 2
                discardedError = np.sum(
                    (elementwiseEpsilon[discard] * elementwiseNorm[discard]) ** 2
                )
                updEpsilon = np.sqrt(
                    (allowedError - discardedError)
                    / (np.linalg.norm(elementwiseNorm[select]) ** 2)
                )
        self.reshapedShape[-1] = np.array(
            select
        ).sum()  # a little trick for ease of coding

        indexString = "["
        for _ in range(len(self.reshapedShape)):
            # this heuristic assumes that the last dimension is for observations
            indexString += ":,"
        selectString = indexString + "select]"
        selected = eval("newTensor" + selectString)

        selected = selected.reshape(list(self.reshapedShape[:-1]) + [-1])[None, :]
        for coreIdx in range(0, len(self.ttCores) - 1):
            # print(coreIdx)
            selected = selected.reshape(np.prod(self.ttCores[coreIdx].shape[:-1]), -1)
            if ("occupancy" in heuristicsToUse) and (
                self.coreOccupancy[coreIdx] >= occupancyThreshold
            ):
                pass
            else:
                Ui = self.ttCores[coreIdx].reshape(
                    self.ttCores[coreIdx].shape[0] * self.reshapedShape[coreIdx], -1
                )
                Ri = selected - Ui @ (Ui.T @ selected)
                if (elementwiseNorm is None) or ("subselect" not in heuristicsToUse):
                    URi, _, _ = deltaSVDCuPy(
                        Ri, np.linalg.norm(selected), newTensorSize, updEpsilon
                    )
                else:
                    URi, _, _ = deltaSVDCuPy(
                        Ri,
                        np.linalg.norm(elementwiseNorm[select]),
                        newTensorSize,
                        updEpsilon,
                    )
                self.ttCores[coreIdx] = cp.hstack((Ui, URi))
                self.ttCores[coreIdx + 1] = cp.concatenate(
                    (
                        self.ttCores[coreIdx + 1],
                        cp.zeros(
                            (
                                URi.shape[-1],
                                self.ttCores[coreIdx + 1].shape[1],
                                self.ttRanks[coreIdx + 2],
                            )
                        ),
                    ),
                    axis=0,
                )

            self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
                np.prod(self.ttCores[coreIdx].shape[:-1]), -1
            )
            # #project onto existing core and reshape for next core
            selected = (self.ttCores[coreIdx].T @ selected).reshape(
                self.ttCores[coreIdx + 1].shape[0] * self.reshapedShape[coreIdx + 1], -1
            )
            # fold back the previous core
            self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
                -1, self.reshapedShape[coreIdx], self.ttCores[coreIdx].shape[-1]
            )
            # self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            # #fold back the previous core
        self.updateRanks()
        coreIdx += 1
        # coreIdx=len(self.ttCores), i.e working on the last core
        self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
            self.ttCores[coreIdx].shape[0], -1
        )
        self.ttCores[coreIdx] = cp.hstack(
            (self.ttCores[coreIdx], self.projectTensor(newTensor))
        ).reshape(self.ttCores[coreIdx].shape[0], -1, 1)
        return None

    def ttICEstarCuPy(
        self,
        newTensor: np.array,
        epsilon: float = None,
        tenNorm: float = None,
        elementwiseNorm: np.array = None,
        elementwiseEpsilon: np.array = None,
        heuristicsToUse: list = ["skip", "subselect", "occupancy"],
        occupancyThreshold: float = 0.8,
        simpleEpsilonUpdate: bool = False,
    ) -> None:
        if epsilon is None:
            epsilon = self.ttEpsilon
        if ("subselect" in heuristicsToUse) and (newTensor.shape[-1] == 1):
            warning(
                "The streamed tensor has only 1 observation in it. \
                    Subselect heuristic will not be useful!!"
            )
        # print("a")
        newTensor = cp.asarray(newTensor)
        newTensor = newTensor.reshape(list(self.reshapedShape[:-1]) + [-1])[None, :]
        updEpsilon = epsilon
        newTensorSize = len(newTensor.shape) - 1
        # print("b")

        if "skip" in heuristicsToUse and elementwiseEpsilon is None:
            elementwiseEpsilon = self.computeRelError(newTensor)
        if "skip" in heuristicsToUse:
            if np.mean(elementwiseEpsilon) <= epsilon:
                newTensor = self.projectTensor(newTensor)
                self.ttCores[-1] = np.hstack(
                    (self.ttCores[-1].reshape(self.ttRanks[-2], -1), newTensor)
                ).reshape(self.ttRanks[-2], -1, 1)
                return None
        if tenNorm is None and elementwiseNorm is None:
            tenNorm = cp.linalg.norm(newTensor)
        elif tenNorm is None:
            tenNorm = cp.linalg.norm(elementwiseNorm)
        # print("c")

        select = [True] * newTensor.shape[-1]
        discard = [False] * newTensor.shape[-1]
        if "subselect" in heuristicsToUse:
            select = elementwiseEpsilon > epsilon
            discard = elementwiseEpsilon <= epsilon
            if simpleEpsilonUpdate:
                updEpsilon = (
                    epsilon * newTensor.shape[-1]
                    - np.mean(elementwiseEpsilon[discard]) * discard.sum()
                ) / (select.sum())
            else:
                if elementwiseNorm is None:
                    elementwiseNorm = np.linalg.norm(newTensor, axis=0)
                    for _ in range(len(self.ttCores) - 1):
                        elementwiseNorm = np.linalg.norm(elementwiseNorm, axis=0)
                allowedError = (self.ttEpsilon * np.linalg.norm(elementwiseNorm)) ** 2
                discardedError = np.sum(
                    (elementwiseEpsilon[discard] * elementwiseNorm[discard]) ** 2
                )
                updEpsilon = np.sqrt(
                    (allowedError - discardedError)
                    / (np.linalg.norm(elementwiseNorm[select]) ** 2)
                )
        self.reshapedShape[-1] = np.array(
            select
        ).sum()  # a little trick for ease of coding

        indexString = "["
        for _ in range(len(self.reshapedShape)):
            # this heuristic assumes that the last dimension is for observations
            indexString += ":,"
        selectString = indexString + "select]"
        selected = eval("newTensor" + selectString)

        selected = selected.reshape(list(self.reshapedShape[:-1]) + [-1])[None, :]
        for coreIdx in range(0, len(self.ttCores) - 1):
            # print(coreIdx)
            selected = selected.reshape(np.prod(self.ttCores[coreIdx].shape[:-1]), -1)
            if ("occupancy" in heuristicsToUse) and (
                self.coreOccupancy[coreIdx] >= occupancyThreshold
            ):
                pass
            else:
                Ui = cp.asarray(self.ttCores[coreIdx].reshape(
                    self.ttCores[coreIdx].shape[0] * self.reshapedShape[coreIdx], -1
                ))
                Ri = selected - Ui @ (Ui.T @ selected)
                if (elementwiseNorm is None) or ("subselect" not in heuristicsToUse):
                    URi, _, _ = deltaSVDCuPy(
                        Ri, cp.linalg.norm(selected), newTensorSize, updEpsilon
                    )
                else:
                    URi, _, _ = deltaSVDCuPy(
                        Ri,
                        cp.linalg.norm(elementwiseNorm[select]),
                        newTensorSize,
                        updEpsilon,
                    )
                self.ttCores[coreIdx] = cp.hstack((Ui, URi))
                self.ttCores[coreIdx + 1] = cp.concatenate(
                    (
                        cp.asarray(self.ttCores[coreIdx + 1]),
                        # self.ttCores[coreIdx + 1],
                        cp.zeros(
                            (
                                URi.shape[-1],
                                self.ttCores[coreIdx + 1].shape[1],
                                self.ttRanks[coreIdx + 2],
                            )
                        ),
                    ),
                    axis=0,
                )

            self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
                np.prod(self.ttCores[coreIdx].shape[:-1]), -1
                # list(self.ttCores[coreIdx].shape[:-1])+[-1]
            )
            # #project onto existing core and reshape for next core
            selected = (cp.asarray(self.ttCores[coreIdx]).T @ selected).reshape(
                self.ttCores[coreIdx + 1].shape[0] * self.reshapedShape[coreIdx + 1], -1
            )
            # fold back the previous core
            self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
                -1, self.reshapedShape[coreIdx], self.ttCores[coreIdx].shape[-1]
            )
            # self.ttCores[coreIdx]=self.ttCores[coreIdx].reshape(self.ttCores[coreIdx].shape[0],self.reshapedShape[coreIdx],-1)
            # #fold back the previous core
        self.updateRanks()
        coreIdx += 1
        # coreIdx=len(self.ttCores), i.e working on the last core
        self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
            self.ttCores[coreIdx].shape[0], -1
        )
        self.ttCores[coreIdx] = np.hstack(
            (self.ttCores[coreIdx], self.projectTensorCuPy(newTensor))
        ).reshape(self.ttCores[coreIdx].shape[0], -1, 1)
        return None

    def ttRound(self, norm, epsilon=0) -> None:
        """
        Reorthogonalizes and recompresses TT-cores as per `Oseledets 2011`_.

        This is the TT-rounding algorithm described in `TTSVD`_ paper. This
        function first reorthogonalizes the cores from last to first using
        QR decomposition and then recompresses the reorthogonalized TT-cores
        from first to last using `TTSVD`_ algorithm.

        Parameters
        ----------
        norm:obj:`float`
            Norm of the TT-approximation. If there is no information on the
            uncompressed tensor, `ttNorm` function can be used.
        epsilon:obj:`float`, optional
            Relative error upper bound for the recompression step. Note that
            this relative error upper bound is not with respect to the original
            tensor but to the `compressed` tensor. Set to 0 by default.

        Notes
        -----
        **The following attributes are modified as a result of this function:**
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`

        .. _Oseledets 2011:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
            TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
        """
        d = [core.shape[1] for core in self.ttCores]
        for coreIdx in np.arange(len(self.ttCores))[::-1]:
            currCore = self.ttCores[coreIdx]
            # Compute mode 1 unfolding of the tt-core
            currCore = currCore.reshape(currCore.shape[0], -1)
            # Using the transpose results in a row orthogonal Q matrix !!!
            q, r = np.linalg.qr(currCore.T)
            q, r = q.T, r.T
            self.ttCores[coreIdx] = q
            # Contract previous tt-core and R matrix
            self.ttCores[coreIdx - 1] = np.tensordot(
                self.ttCores[coreIdx - 1], r, axes=(-1, 0)
            )
            if coreIdx == 1:
                break
                # pass #TODO: write the case for the last core here.
        # Compression of the orthogonalized representation #
        ranks = [1]
        for coreIdx in range(len(self.ttCores) - 1):
            core = self.ttCores[coreIdx]
            core = core.reshape(np.prod(core.shape[:-1]), -1)
            self.ttCores[coreIdx], sigma, v = deltaSVD(
                core, norm, len(self.ttCores), epsilon
            )
            ranks.append(self.ttCores[coreIdx].shape[-1])
            # fold matrices into tt-cores
            self.ttCores[coreIdx] = self.ttCores[coreIdx].reshape(
                ranks[coreIdx], d[coreIdx], -1
            )
            self.ttCores[coreIdx + 1] = (
                (np.diag(sigma) @ v) @ self.ttCores[coreIdx + 1]
            ).reshape(ranks[-1] * d[coreIdx + 1], -1)
        self.ttCores[-1] = self.ttCores[-1].reshape(-1, d[-1], 1)
        self.ttRanks = [1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    @staticmethod
    def ittd(tt1, tt2, rounding=True, epsilon=0) -> "ttObject":
        """
        Alternative incremental tensor decomposition algorithm used to benchmark.

        Note
        ----
        This method is empirically shown to be inferior to `TT-ICE`_ and
        `TT-ICE*`_.

        Parameters
        ----------
        tt1:obj:`ttObject`
            Tensor in TT-format.
        tt2:obj:`ttObject`
            Tensor in TT-format.
        rounding:obj:`bool`, optional
            Determines if `ttRounding` will be done to the incremented TT-core.
            Note that for unbounded streams, rounding is essential. Otherwise
            TT-ranks will grow unboundedly.
        epsilon:obj:`float`, optional
            Relative error upper bound for the recompression step. Note that
            this relative error upper bound is not with respect to the original
            tensor but to the `compressed` tensor. Set to 0 by default. Not used
            if `rounding` is set to `False`.

        Notes
        -----
        **The following attributes are modified as a result of this function:**
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`

        .. _TT-ICE:
            https://arxiv.org/abs/2211.12487
            _TT-ICE*:
            https://arxiv.org/abs/2211.12487
        """
        if not isinstance(tt1, ttObject) or not isinstance(tt2, ttObject):
            if isinstance(tt1, list):
                tt1 = ttObject(tt1)
            if isinstance(tt2, list):
                tt2 = ttObject(tt2)
            if not isinstance(tt1, ttObject) or not isinstance(tt2, ttObject):
                raise AttributeError("One of the passed objects is not in TT-format!")
        tt1.ttCores[-1] = tt1.ttCores[-1].squeeze(-1)
        tt2.ttCores[-1] = tt2.ttCores[-1].squeeze(-1)
        rank1, numObs1 = tt1.ttCores[-1].shape
        rank2, numObs2 = tt2.ttCores[-1].shape
        tt1.ttCores[-1] = np.concatenate(
            (tt1.ttCores[-1], np.zeros((rank1, numObs2))), axis=1
        )
        tt2.ttCores[-1] = np.concatenate(
            (np.zeros((rank2, numObs1)), tt2.ttCores[-1]), axis=1
        )
        # Addition in TT-format #
        for coreIdx in range(len(tt1.ttCores)):
            if coreIdx == 0:
                tt1.ttCores[coreIdx] = np.concatenate(
                    (tt1.ttCores[coreIdx].squeeze(0), tt2.ttCores[coreIdx].squeeze(0)),
                    axis=1,
                )[None, :, :]
            elif coreIdx == len(tt1.ttCores) - 1:
                tt1.ttCores[coreIdx] = np.concatenate(
                    (tt1.ttCores[coreIdx], tt2.ttCores[coreIdx]), axis=0
                )[:, :, None]
            else:
                s11, s12, s13 = tt1.ttCores[coreIdx].shape
                s21, s22, s23 = tt2.ttCores[coreIdx].shape
                tt1.ttCores[coreIdx] = np.concatenate(
                    (tt1.ttCores[coreIdx], np.zeros((s11, s12, s23))), axis=2
                )
                tt2.ttCores[coreIdx] = np.concatenate(
                    (np.zeros((s21, s22, s13)), tt2.ttCores[coreIdx]), axis=2
                )
                tt1.ttCores[coreIdx] = np.concatenate(
                    (tt1.ttCores[coreIdx], tt2.ttCores[coreIdx]), axis=0
                )
        tt1.ttRanks = [1]
        for core in tt1.ttCores:
            tt1.ttRanks.append(core.shape[-1])
        if rounding:
            tenNorm = ttObject.ttNorm(tt1)
            tt1.rounding(tenNorm, epsilon=epsilon)
        return tt1
