# Zooming Memory-Based Initialization (`ZoMBI`)

This software package implements the Zooming Memory-Based Initialization (`ZoMBI`) algorithm as an augmentation to standard Bayesian optimization. The `ZoMBI` algorithm augments standard Bayesian optimization by (1) zooming in the bounds of the search space for each dimension based on previously high-performing datapoints stored in memory to quickly find solutions to "needle-in-a-haystack" problems and (2) purging the memory of all other historical data points to accelerate algorithm compute times from $O(n^3)$ to $O(1)$.

The package has two primary components:

- **Active learning acquisition functions**: Implementation of the LCB Adaptive and EI Abrupt custom acquisition functions that use active learning to tune acquisition of new data points within the search space (in `acquisitions.py`).
- **The `ZoMBI` optimization algorithm**: Takes in a multi-dimensional, complex "needle-in-a-haystack" dataset and optimizes it using one of the selected acquisition functions (in `zombi.py`).

## Installation
Install the `diversipy` package `python -m pip install git+https://github.com/DavidWalz/diversipy`

## Usage



| Files | Description |
| ------------- | ------------------------------ |
| [examples.ipynb](./examples.ipynb)  | Run ZoMBI on the example datasets 5D Ackley function and 5D Negative Poisson's Ratio. |
| [zombi.py](./zombi.py)  | Code to run the ZoMBI optimization procedure. |
| [acquisitions.py](./acquisitions.py)  | Code for the acquisition functions LCB, LCB Adaptive, EI, and EI Abrupt. |
| [utils.py](./utils.py)  | Utility code for ZoMBI optimization and plotting. |
| [data](./data)  | Folder containing the [code to train the RF Poisson's ratio model](./data) and the [pickled pre-trained RF Poisson's ratio model](./data/poisson_RF_trained.pkl). |


# Authors
The code was written by Alexander E. Siemenn and Zekun Run, under the supervision of Tonio Buonassisi and Qianxiao Li.
