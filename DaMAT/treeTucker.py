import warnings

# import time
import vtk

import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
from utils import deltaSVD  # , ttsvd

# from pickle import dump, load


class NotFoundError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Node:
    def __init__(self, val, children=None) -> None:
        self.children = children or []
        self.val = val
        self._propagated = False

    def __str__(self) -> str:
        return self.children


class Tree:
    def __init__(self) -> None:
        self.root = None
        self._depth = 0
        self._size = 0
        self._leaveCount = 0

    def findNode(self, node, key):
        if (node is None) or (node.val == key):
            return node
        for child in node.children:
            return_node = self.findNode(child, key)
            if return_node:
                return return_node
        return None

    def isEmpty(self):
        return self._size == 0

    def initializeTree(self, vals):
        # Initalizes the tree
        if self.root is None:
            if type(vals) is list:
                self.root = vals
            else:
                raise TypeError(f"Type: {type(vals)} is not known!")
        else:
            warnings.warn("Root node already implemented! Doing nothing.")

    def insertNode(self, val, parent=None):
        newNode = Node(val)
        if parent is None:
            self.root = newNode
            self._depth = 1
            self._size = 1
        else:
            parentNode = self.findNode(self.root, parent)
            if not (parentNode):
                raise NotFoundError(f"No parent was found for parent name: {parent}")
            parentNode.children.append(newNode)
            parentNode._propagated = True
            self._size += 1

    def toList(self):
        # Returns a list from the tree
        return None


def createDimensionTree(inp, numSplits, minSplitSize):
    if type(inp) is np.ndarray:
        dims = np.array(inp.shape)
    elif type(inp) is tuple or list:
        dims = np.array(inp)  # NOQA
    else:
        raise TypeError(f"Type: {type(inp)} is unsupported!!")
    dimensionTree = Tree()
    dimensionTree.insertNode(inp.tolist())
    print(np.array(dimensionTree.root.val))
    leaves = []
    leaves.append(dimensionTree.root.val.copy())
    while leaves:
        leaf = leaves.pop(0)
        node = dimensionTree.findNode(dimensionTree.root, leaf)
        if (not node._propagated) and (len(node.val) > minSplitSize + 1):
            # for split in [data[x:x+10] for x in xrange(0, len(data), 10)]:
            for split in np.array_split(np.array(node.val), numSplits):
                print(split)
                # tree.insertNode(split,node.val)
                # leaves.append(split)
                dimensionTree.insertNode(split.tolist(), node.val)
                leaves.append(split.tolist())
    return dimensionTree


nSplits = 5
splitSize = 3
deneme = np.random.randint(0, 50, 56)
# print(deneme)
# anan=createDimensionTree(deneme,nSplits,splitSize)


reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(
    "/home/doruk/gameDataProject/TT-ICE/test_bd_ns308_np32_000001_000001.vtp"
)

reader.Update()

polydata = reader.GetOutput()
points = polydata.GetPoints()
array = points.GetData()
numpy_nodes = vtk_to_numpy(array)

data = numpy_nodes.reshape([2, 2, 2, 3, 7, 7, 11, 13, 3, 1])
dataNorm = np.linalg.norm(data)
dataLen = len(data.shape)
# dataLen=7

U1, s1, V1 = deltaSVD(data.reshape(168, -1), dataNorm, dataLen, eps=0.1, tuckerDelta=True)
ssqrt1 = np.sqrt(s1)
r1 = len(s1)
U1 = U1 @ np.diag(ssqrt1)  # 2,2,2,3,7
V1 = np.diag(ssqrt1) @ V1  # 7,11,13,3,1

U2, s2, V2 = deltaSVD(
    U1.reshape(8, -1), np.linalg.norm(U1), dataLen, eps=0.1, tuckerDelta=True
)
# U2,s2,V2=deltaSVD(U1.reshape(8,-1),dataNorm,dataLen,eps=0.1,tuckerDelta=True)
ssqrt2 = np.sqrt(s2)
U2 = U2 @ np.diag(ssqrt2)  # 2,2,2
V2 = np.diag(ssqrt2) @ V2  # 3,7,r1

U3, s3, V3 = deltaSVD(
    V1.reshape(-1, 13 * 3 * 1), np.linalg.norm(V1), dataLen, eps=0.1, tuckerDelta=True
)
# U3,s3,V3=deltaSVD(V1.reshape(-1,13*3*1),dataNorm,dataLen,eps=0.1,tuckerDelta=True)
ssqrt3 = np.sqrt(s3)
U3 = U3 @ np.diag(ssqrt3)  # r1,7,11
V3 = np.diag(ssqrt3) @ V3  # 13,3,1

2 + 2
