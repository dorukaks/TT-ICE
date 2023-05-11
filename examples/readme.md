# Examples
This folder contains python scripts that reproduces the results in the [manuscript](https://arxiv.org/abs/2211.12487). There are 3 scripts for each data type and they are named with the convention `<dataType>Tests<algorithmName>.py`. The `algorithmName` can be one of the following:

1. `TTICE`: TT-ICE
2. `TTICEstar`: TT-ICE*
3. `ITTD`: ITTD

There are 2 options for `dataType`:

1. `atari`: Image based dataset comprised of Atari gameplay sequences
2. `catgel`: Snapshots from PDE simulations of self-oscillating gel sheets

As mentioned in the manuscript, comparisons including TT-FOA approach were done by creating separate synthetic tensors and are also included in this folder.

## Atari gameplay sequences

This set of experiments use done with batches of `.png` files. There are some Arguments and flags that can/should be used to get various different versions.

This class of scripts is run as:
```
python3 examples/<Script Name>.py <Game Name>
```
Game name can be one of the following:

1. BeamRider
2. Breakout
3. Enduro
4. MsPacman
5. Pong
6. Qbert
7. Seaquest
8. SpaceInvaders

We provide the parameters as we have conducted the experiments in the scripts. However, other options that can be modified for exploration.

Those parameters are:

- `imgLocation`: Directory that files are saved.
- `epsilon` : Error upper bound term.
- `initialize` : Number of batches used to initialize the first set of TT-cores.
- `increment` : Number of batches used to expand the TT-cores at each increment step.
- `batchMultiplier` : This parameter does not have an effect on the results. In order to save memory while computing reconstruction/projection error, we split each batch into sub-batches. This parameter only controls that parameter size. Increase this number if you are experiencing problems while computing error.
- `heuristicsToUse` : This parameter exists only in TT-ICE* scripts and is a list of string that contains heuristic options.
- `roundingInterval` : This parameter exists only in ITTD scripts and controls the frequency in which TT-rounding is executed.
- `saveTrainedCores` : Boolean parameter that switches saving the compressed TT-cores to `.npy` files under `./trainedCores/` diretory
- `printMetrics2File` : Boolean parameter that switches printing the compression metrics to a `.txt` file under `./textOutputs/` diretory

The columns of the text file are specified in each file individually but can be summarized as:
- `epsilon` : Error upper bound used at that increment step 
- `numFrames` : Number of frames at that increment batch
- `usedFrames` : Number of frames used at that step to expand TT-cores 
- `errBeforeUpdate` : Relative error of approximating the streamed batch with existing TT-cores before update
- `errAfterUpdate` : Relative error of the streamed batch with existing TT-cores after update
- `stepTime` : Elapsed (wall) time during update step
- `cumulRecError` : Reconstruction error of the compressed data
- `cumulTestError` : Projection error of unseen data with existing TT-cores
- `compressionRatio` : Compression ratio of the existing TT-cores
- `ttRanks` : TT-ranks of the approximation.


## PDE snapshots

This set of experiments use done with `.cgf` files. The `.cgf` files can be read with NumPy's `load` function. There are some Arguments and flags that can/should be used to get various different versions.

This class of scripts is run as:
```
python3 examples/<Script Name>.py
```
Similar to previous type, we provide the parameters as we have conducted the experiments in the scripts. However, other options that can be modified for exploration.

Those parameters are:

- `simulationDir`: Directory that the simulation files are saved.
- `epsilon` : Error upper bound term.
- `initialize` : Number of batches used to initialize the first set of TT-cores.
- `increment` : Number of batches used to expand the TT-cores at each increment step.
- `heuristicsToUse` : This parameter exists only in TT-ICE* scripts and is a list of string that contains heuristic options.
- `roundingInterval` : This parameter exists only in ITTD scripts and controls the frequency in which TT-rounding is executed.
- `saveTrainedCores` : Boolean parameter that switches saving the compressed TT-cores to `.npy` files under `./trainedCores/` diretory
- `printMetrics2File` : Boolean parameter that switches printing the compression metrics to a `.txt` file under `./textOutputs/` diretory

The columns of the text file are specified in each file individually but can be summarized as:
- `startIdx` : Index of the first simulation in the batch
- `endIdx` : Index of the last simulation in the batch
- `stepTime` : Elapsed (wall) time during update step
- `stepError` : Relative error of the streamed batch with existing TT-cores after update
- `cumulRecError` : Reconstruction error of the compressed data
- `compressionRatio` : Compression ratio of the existing TT-cores
- `ttRanks` : TT-ranks of the approximation.

## TT-FOA experiments

For the reasons explained in the [manuscript](https://arxiv.org/abs/2211.12487) we couldn't use the same datasets to compare TT-FOA with TT-ICE. To provide a comparison between TT-FOA and TT-ICE, we created experiments with synthetic tensors.

There are some parameters that should carefully be adjusted for these experiments.

- `nDims` : Number of dimensions of the syntetic tensor. Note that last dimension is the temporal dimension here.
- `rep` : Number of repetitions for the experiment.
- `dims` : Dimensions of the synthetic tensor.
- `ranks` : True TT-ranks of the synthetic tensor.

**Note:** For this set of experiments, the script `createRandomTensor.py` before running either `randTensorTTFOATest.py` or `randTensorTTICETest.py`. This set of experiments only print outputs to terminal.




