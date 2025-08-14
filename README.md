# random_matrices
Generators for several types of random matrices in Python.

**Important note**: This repository has just recently been made public 
and is still undergoing debugging and development.

## Available ensembles of random matrices
- GOE: Gaussian Orthogonal Ensemble
- SG_0_+: positive-definite matrices with the identity matrix as mean value
- SG_eps_+: positive-definite matrices with identity matrix as mean value and with a positive lower bound
- SE_0_+: positive-definite matrices with given mean value
- SE_+0: positive semidefinite matrices with given mean value
- SE_rect: rectangular matrices with given mean value

## Installation
First, clone the repository in the folder of your choice using:
```
git clone git@github.com:rcapillon/random_matrices.git
```
Then, activate the virtual environment for your project and install this package and its requirements with:
```
cd random_matrices/
pip install .
```

## Usage
Choose and import the desired generators from generators.py, e.g.:
```
from random_matrices.generators import SE_0_plus
```
Users are invited to look up the necessary arguments to generate samples of a random matrix.