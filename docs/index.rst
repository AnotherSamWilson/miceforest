.. miceforest documentation master file, created by
   sphinx-quickstart on Sat Jul 27 20:34:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to miceforest's Documentation!
======================================

This documentation is meant to describe class methods and parameters only,
for a thorough walkthrough of usage, please see the 
`Github README <https://github.com/AnotherSamWilson/miceforest>`_.

In general, the user will only be interacting with these two classes:


.. toctree::
   :maxdepth: 1
   :caption: Classes:

   ImputationKernel <ImputationKernel>
   ImputedData <ImputedData>


How miceforest Works
--------------------

Multiple Imputation by Chained Equations ‘fills in’ (imputes) missing
data in a dataset through an iterative series of predictive models. In
each iteration, each specified variable in the dataset is imputed using
the other variables in the dataset. These iterations should be run until
it appears that convergence has been met.

.. image:: https://i.imgur.com/2L403kU.png
   :target: https://github.com/AnotherSamWilson/miceforest?tab=readme-ov-file#the-mice-algorithm

This process is continued until all specified variables have been
imputed. Additional iterations can be run if it appears that the average
imputed values have not converged, although no more than 5 iterations
are usually necessary.

This package provides fast, memory efficient Multiple Imputation by Chained 
Equations (MICE) with lightgbm. The R version of this package may be found
`here <https://github.com/FarrellDay/miceRanger>`_.

`miceforest` was designed to be:

  - **Fast**
      - Uses lightgbm as a backend
      - Has efficient mean matching solutions.
      - Can utilize GPU training
  - **Flexible**
      - Can impute pandas dataframes
      - Handles categorical data automatically
      - Fits into a sklearn pipeline
      - User can customize every aspect of the imputation process
  - **Production Ready**
      - Can impute new, unseen datasets quickly
      - Kernels are efficiently compressed during saving and loading
      - Data can be imputed in place to save memory
      - Can build models on non-missing data