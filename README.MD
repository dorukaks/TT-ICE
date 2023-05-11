# Welcome to TT-ICE code repository

This GitHub repository contains the python implementation of the Tensor-Train Incremental Core Expansion (TT-ICE) and its heuristically enhanced version TT-ICE*.

The TT-ICE algorithm is implemented as a part of the DaMAT library. More information on DaMAT library can be found in the [documentation page](https://dorukaks.github.io/TT-ICE/DaMAT.html) 

## DaMAT Library

DaMAT is a python library that provides support for tensor operations for Tensor-Train format. You can find both one-shot algorithms (such as TT-SVD) as well as incremental algorithms (such as TT-ICE, TT-FOA and ITTD). Although multiple options were provided for incremental computation of Tensor-Train format, the main aim of this library is to provide a python implementation for TT-ICE.

## Examples

In addition to the main library/codebase for TT-ICE, this repository also contains scripts that were used for the numerical experiments of the manuscript [An Incremental Tensor Train Decomposition Algorithm](https://arxiv.org/abs/2211.12487) under the examples folder.

As a part of the reproducibility badge requirements from SIAM SISC, we have submitted the necessary data to reproduce the experiments in Section 4 of the manuscript. You can temporarily access the files through [Google Drive](https://drive.google.com/drive/folders/1M1HZXL-QnKZOI1gjRSGpHyox5X7wQwG-?usp=sharing) until that data is released.

Once you clone this repository, you need to setup the DaMAT package using the following prompt.

```
python3 setup.py install --user
```

Please carefully review the associated [readme file](./examples/readme.md) for the experiments.