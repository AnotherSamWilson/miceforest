.. miceforest documentation master file, created by
   sphinx-quickstart on Tue Dec 28 11:24:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./logo/icon_small.png
   :align: right
   :width: 250
   :alt: miceforest logo.

Welcome to miceforest's documentation!
======================================

``miceforest`` imputes missing data using LightGBM in an iterative method known as Multiple Imputation by Chained Equations (MICE). It was designed to be:

* **Fast**
   * Uses lightgbm as a backend
   * Has efficient mean matching solutions.
   * Can utilize GPU training
* **Flexible**
   * Can impute pandas dataframes and numpy arrays
   * Handles categorical data automatically
   * Fits into a sklearn pipeline
   * User can customize every aspect of the imputation process
* **Production Ready**
   * Can impute new, unseen datasets very quickly
   * Kernels are efficiently compressed during saving and loading
   * Data can be imputed in place to save memory
   * Can build models on non-missing data

There are very extensive `beginner <https://github.com/AnotherSamWilson/miceforest#The-Basics>`_ and `advanced <https://github.com/AnotherSamWilson/miceforest#Advanced-Features>`_ tutorials on the github readme. Below is a table of contents for the topics covered:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ImputationKernel
   utils
