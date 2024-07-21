[![DOI](https://zenodo.org/badge/289387436.svg)](https://zenodo.org/badge/latestdoi/289387436)
[![Downloads](https://static.pepy.tech/badge/miceforest)](https://pepy.tech/project/miceforest)
[![Pypi](https://img.shields.io/pypi/v/miceforest.svg)](https://pypi.python.org/pypi/miceforest)
[![Conda
Version](https://img.shields.io/conda/vn/conda-forge/miceforest.svg)](https://anaconda.org/conda-forge/miceforest)
[![PyVersions](https://img.shields.io/pypi/pyversions/miceforest.svg?logo=python&logoColor=white)](https://pypi.org/project/miceforest/)  
[![tests +
mypy](https://github.com/AnotherSamWilson/miceforest/actions/workflows/run_tests.yml/badge.svg)](https://github.com/AnotherSamWilson/miceforest/actions/workflows/run_tests.yml)
[![Documentation
Status](https://readthedocs.org/projects/miceforest/badge/?version=latest)](https://miceforest.readthedocs.io/en/latest/?badge=latest)
[![CodeCov](https://codecov.io/gh/AnotherSamWilson/miceforest/branch/master/graphs/badge.svg?branch=master&service=github)](https://codecov.io/gh/AnotherSamWilson/miceforest)
<!-- [![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT) -->
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)   -->
<!-- [![DEV_Version_Badge](https://img.shields.io/badge/Dev-5.6.3-blue.svg)](https://pypi.org/project/miceforest/) -->


# miceforest: Fast, Memory Efficient Imputation with LightGBM

<a href='https://github.com/AnotherSamWilson/miceforest'><img src='https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/icon.png' align="right" height="300" /></a>

Fast, memory efficient Multiple Imputation by Chained Equations (MICE)
with lightgbm. The R version of this package may be found
[here](https://github.com/FarrellDay/miceRanger).

`miceforest` was designed to be:

  - **Fast**
      - Uses lightgbm as a backend
      - Has efficient mean matching solutions.
      - Can utilize GPU training
  - **Flexible**
      - Can impute pandas dataframes and numpy arrays
      - Handles categorical data automatically
      - Fits into a sklearn pipeline
      - User can customize every aspect of the imputation process
  - **Production Ready**
      - Can impute new, unseen datasets quickly
      - Kernels are efficiently compressed during saving and loading
      - Data can be imputed in place to save memory
      - Can build models on non-missing data


This document contains a thorough walkthrough of the package,
benchmarks, and an introduction to multiple imputation. More information
on MICE can be found in Stef van Buuren’s excellent online book, which
you can find
[here](https://stefvanbuuren.name/fimd/ch-introduction.html).

#### Table of Contents:

  - [Package
    Meta](https://github.com/AnotherSamWilson/miceforest#Package-Meta)
  - [The
    Basics](https://github.com/AnotherSamWilson/miceforest#The-Basics)
      - [Basic
        Examples](https://github.com/AnotherSamWilson/miceforest#Basic-Examples)
      - [Customizing LightGBM
        Parameters](https://github.com/AnotherSamWilson/miceforest#Customizing-LightGBM-Parameters)
      - [Available Mean Match
        Schemes](https://github.com/AnotherSamWilson/miceforest#Controlling-Tree-Growth)
      - [Imputing New Data with Existing
        Models](https://github.com/AnotherSamWilson/miceforest#Imputing-New-Data-with-Existing-Models)
      - [Saving and Loading
        Kernels](https://github.com/AnotherSamWilson/miceforest#Saving-and-Loading-Kernels)
      - [Implementing sklearn
        Pipelines](https://github.com/AnotherSamWilson/miceforest#Implementing-sklearn-Pipelines)
  - [Advanced
    Features](https://github.com/AnotherSamWilson/miceforest#Advanced-Features)
      - [Customizing the Imputation
        Process](https://github.com/AnotherSamWilson/miceforest#Customizing-the-Imputation-Process)
      - [Building Models on Nonmissing
        Data](https://github.com/AnotherSamWilson/miceforest#Building-Models-on-Nonmissing-Data)
      - [Tuning
        Parameters](https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters)
      - [On
        Reproducibility](https://github.com/AnotherSamWilson/miceforest#On-Reproducibility)
      - [How to Make the Process
        Faster](https://github.com/AnotherSamWilson/miceforest#How-to-Make-the-Process-Faster)
      - [Imputing Data In
        Place](https://github.com/AnotherSamWilson/miceforest#Imputing-Data-In-Place)
  - [Diagnostic
    Plotting](https://github.com/AnotherSamWilson/miceforest#Diagnostic-Plotting)
      - [Imputed
        Distributions](https://github.com/AnotherSamWilson/miceforest#Distribution-of-Imputed-Values)
      - [Correlation
        Convergence](https://github.com/AnotherSamWilson/miceforest#Convergence-of-Correlation)
      - [Variable
        Importance](https://github.com/AnotherSamWilson/miceforest#Variable-Importance)
      - [Mean
        Convergence](https://github.com/AnotherSamWilson/miceforest#Variable-Importance)
  - [Benchmarks](https://github.com/AnotherSamWilson/miceforest#Benchmarks)
  - [Using the Imputed
    Data](https://github.com/AnotherSamWilson/miceforest#Using-the-Imputed-Data)
  - [The MICE
    Algorithm](https://github.com/AnotherSamWilson/miceforest#The-MICE-Algorithm)
      - [Introduction](https://github.com/AnotherSamWilson/miceforest#The-MICE-Algorithm)
      - [Common Use
        Cases](https://github.com/AnotherSamWilson/miceforest#Common-Use-Cases)
      - [Predictive Mean
        Matching](https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching)
      - [Effects of Mean
        Matching](https://github.com/AnotherSamWilson/miceforest#Effects-of-Mean-Matching)

## Installation

This package can be installed using either pip or conda, through
conda-forge:

``` bash
# Using pip
$ pip install miceforest --no-cache-dir

# Using conda
$ conda install -c conda-forge miceforest
```

You can also download the latest development version from this
repository. If you want to install from github with conda, you must
first run `conda install pip git`.

``` bash
$ pip install git+https://github.com/AnotherSamWilson/miceforest.git
```

## Classes

miceforest has 3 main classes which the user will interact with:

  - [`ImputationKernel`](https://miceforest.readthedocs.io/en/latest/ik/miceforest.ImputationKernel.html#miceforest.ImputationKernel)
    - This class contains the raw data off of which the `mice` algorithm
    is performed. During this process, models will be trained, and the
    imputed (predicted) values will be stored. These values can be used
    to fill in the missing values of the raw data. The raw data can be
    copied, or referenced directly. Models can be saved, and used to
    impute new datasets.
  - [`ImputedData`](https://miceforest.readthedocs.io/en/latest/ik/miceforest.ImputedData.html#miceforest.ImputedData)
    - The result of `ImputationKernel.impute_new_data(new_data)`. This
    contains the raw data in `new_data` as well as the imputed values.  
  - [`MeanMatchScheme`](https://miceforest.readthedocs.io/en/latest/ik/miceforest.MeanMatchScheme.html#miceforest.MeanMatchScheme)
    - Determines how mean matching should be carried out. There are 3
    built-in mean match schemes available in miceforest, discussed
    below.


## Basic Usage

We will be looking at a few simple examples of imputation. We need to
load the packages, and define the data:

```python
import miceforest as mf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load data and introduce missing values
iris = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
iris.rename({"target": "species"}, inplace=True, axis=1)
iris['species'] = iris['species'].astype('category')
iris_amp = mf.ampute_data(iris,perc=0.25,random_state=1991)
```


```python

```


