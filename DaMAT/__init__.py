"""
Welcome to DaMAT package documentation!


This python package currently offers support for multidimensional tensors in Tensor-Train format.
We use the TT-SVD algorithm proposed by Ivan Oseledets and TT-ICE algorithm proposed by Doruk Aksoy.

In future releases, the coverage may be extended to other tensor decomposition formats such as CP and/or Tucker.
"""

from .ttObject import ttObject
from .utils import *