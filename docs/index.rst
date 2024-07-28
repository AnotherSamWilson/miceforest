.. miceforest documentation master file, created by
   sphinx-quickstart on Sat Jul 27 20:34:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to miceforest's Documentation!
======================================

This documentation is meant to describe class methods and parameters only,
for a thorough walkthrough of usage, please see the 
`Github README <https://github.com/AnotherSamWilson/miceforest>`_.


Fast, memory efficient Multiple Imputation by Chained Equations (MICE)
with lightgbm. The R version of this package may be found
`here <https://github.com/FarrellDay/miceRanger>`_.

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



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ImputationKernel <ImputationKernel>
   ImputedData <ImputedData>
