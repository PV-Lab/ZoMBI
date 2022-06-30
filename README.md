# Zooming Memory-Based Initialization (`ZoMBI`)

This software package implements the Zooming Memory-Based Initialization (`ZoMBI`) algorithm as an augmentation to standard Bayesian optimization. The `ZoMBI` algorithm augments standard Bayesian optimization by (1) zooming in the bounds of the search space for each dimension based on previously high-performing datapoints stored in memory to quickly find solutions to "needle-in-a-haystack" problems and (2) purging the memory of all other historical data points to accelerate algorithm compute times from $O(n^3)$ to $O(1)$.

The package has two primary components:

- **Active learning acquisition functions**: Implementation of the LCB Adaptive and EI Abrupt custom acquisition functions that use active learning to tune the sampling of new data points within the search space (in `acquisitions.py`).
- **The `ZoMBI` optimization algorithm**: Takes in a multi-dimensional, complex "needle-in-a-haystack" dataset and optimizes it using one of the selected acquisition functions (in `zombi.py`).

# How to Cite
Please cite our paper if you want to use `ZoMBI`:

# File Summary
| Files | Description |
| ------------- | ------------------------------ |
| [examples.ipynb](./examples.ipynb)  | Run ZoMBI on the example datasets 5D Ackley function and 5D Negative Poisson's Ratio. |
| [zombi.py](./zombi.py)  | Code to run the ZoMBI optimization procedure. |
| [acquisitions.py](./acquisitions.py)  | Code for the acquisition functions LCB, LCB Adaptive, EI, and EI Abrupt. |
| [utils.py](./utils.py)  | Utility code for ZoMBI optimization and plotting. |
| [data](./data)  | Folder containing the [code to train the RF Poisson's ratio model](./data) and the [pickled pre-trained RF Poisson's ratio model](./data/poisson_RF_trained.pkl). |

# Installation
Install the `diversipy` package: `python -m pip install git+https://github.com/DavidWalz/diversipy`. 

# Usage
To use the `ZoMBI` algorithm, call `from zombi import *` and instantiate the `ZoMBI($\cdot$)` class variables:

| Class Variable | Input |
| ------------- | ------------------------------ |
| `dataset_X` | An (n,d) array of data points to sample from, where n is the number of data points and d is the number of dimensions. |
| `dataset_fX` | An (n,) array of labels, f(X), for the corresponding X data. |
| `fX_model` | A model or function of the form `func(X)` to predict f(X) from X data. |
| `BO` | The user-selected acquisition function. Options: `LCB`, `LCB_ada`, `EI`, `EI_abrupt`. |
| `nregular` | The number of regular BO experiments to run before `ZoMBI`. |
| `activations` | The number of `ZoMBI` activations. |
| `memory` | The number of memory points to retain per `ZoMBI` activation. |
| `forward` | The number of forwar experiments to run per `ZoMBI` activation. |
| `ensemble` | The number of independent ensemble runs. |

The optimization procedure begins by calling 



# Datasets

# Authors
The code was written by Alexander E. Siemenn and Zekun Run, under the supervision of Tonio Buonassisi and Qianxiao Li.
