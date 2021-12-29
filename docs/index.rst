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

``miceforest`` was designed to be:

  * **Fast** Uses lightgbm as a backend, and has efficient mean matching solutions.
  * **Memory Efficient** Capable of performing multiple imputation without copying the dataset. If the dataset can fit in memory, it can (probably) be imputed.
  * **Flexible** Can handle pandas DataFrames and numpy arrays. The imputation process can be completely customized. Can handle categorical data automatically.
  * **Used In Production** Kernels can be saved and impute new, unseen datasets. Imputing new data is often orders of magnitude faster than including the new data in a new `mice` procedure. Imputation models can be built off of a kernel dataset, even if there are no missing values. New data can also be imputed in place.

There are very extensive `beginner <https://github.com/AnotherSamWilson/miceforest#The-Basics>`_ and `advanced <https://github.com/AnotherSamWilson/miceforest#Advanced-Features>`_ tutorials on the github readme. Below is a table of contents for the topics covered:


Table of Contents:

 - `Package Meta <https://github.com/AnotherSamWilson/miceforest#Package-Meta>`_
 - `The Basics <https://github.com/AnotherSamWilson/miceforest#The-Basics>`_
    - `Basic Examples <https://github.com/AnotherSamWilson/miceforest#Basic-Examples>`_
    - `Controlling Tree Growth <https://github.com/AnotherSamWilson/miceforest#Controlling-Tree-Growth>`_
    - `Preserving Data Assumptions <https://github.com/AnotherSamWilson/miceforest#Preserving-Data-Assumptions>`_
    - `Imputing With Gradient Boosted Trees <https://github.com/AnotherSamWilson/miceforest#Imputing-With-Gradient-Boosted-Trees>`_
 - `Advanced Features <https://github.com/AnotherSamWilson/miceforest#Advanced-Features>`_
    - `Customizing the Imputation Process <https://github.com/AnotherSamWilson/miceforest#Customizing-the-Imputation-Process>`_
    - `Imputing New Data with Existing Models <https://github.com/AnotherSamWilson/miceforest#Imputing-New-Data-with-Existing-Models>`_
    - `Building Models on Nonmissing Data <https://github.com/AnotherSamWilson/miceforest#Building-Models-on-Nonmissing-Data>`_
    - `Tuning Parameters <https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters>`_
    - `How to Make the Process Faster <https://github.com/AnotherSamWilson/miceforest#How-to-Make-the-Process-Faster>`_
    - `Imputing Data In Place <https://github.com/AnotherSamWilson/miceforest#Imputing-Data-In-Place>`_
 - `Diagnostic Plotting <https://github.com/AnotherSamWilson/miceforest#Diagnostic-Plotting>`_
    - `Imputed Distributions <https://github.com/AnotherSamWilson/miceforest#Distribution-of-Imputed-Values>`_
    - `Correlation Convergence <https://github.com/AnotherSamWilson/miceforest#Convergence-of-Correlation>`_
    - `Variable Importance <https://github.com/AnotherSamWilson/miceforest#Variable-Importance>`_
    - `Mean Convergence <https://github.com/AnotherSamWilson/miceforest#Variable-Importance>`_
 - `Using the Imputed Data <https://github.com/AnotherSamWilson/miceforest#Using-the-Imputed-Data>`_
 - `The MICE Algorithm <https://github.com/AnotherSamWilson/miceforest#The-MICE-Algorithm>`_
    - `Introduction <https://github.com/AnotherSamWilson/miceforest#The-MICE-Algorithm>`_
    - `Common Use Cases <https://github.com/AnotherSamWilson/miceforest#Common-Use-Cases>`_
    - `Predictive Mean Matching <https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching>`_
    - `Effects of Mean Matching <https://github.com/AnotherSamWilson/miceforest#Effects-of-Mean-Matching>`_




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   miceforest

Indices and tables
==================

* :ref:`genindex`
