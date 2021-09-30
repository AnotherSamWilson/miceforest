
[![Build
Status](https://travis-ci.org/AnotherSamWilson/miceforest.svg?branch=master)](https://travis-ci.org/AnotherSamWilson/miceforest)
[![Documentation
Status](https://readthedocs.org/projects/miceforest/badge/?version=latest)](https://miceforest.readthedocs.io/en/latest/?badge=latest)
[![MIT
license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![CodeCov](https://codecov.io/gh/AnotherSamWilson/miceforest/branch/master/graphs/badge.svg?branch=master&service=github)](https://codecov.io/gh/AnotherSamWilson/miceforest)
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
[![DEV\_Version\_Badge](https://img.shields.io/badge/Dev-4.0.2-blue.svg)](https://pypi.org/project/miceforest/)
[![Pypi](https://img.shields.io/pypi/v/miceforest.svg)](https://pypi.python.org/pypi/miceforest)
[![Conda
Version](https://img.shields.io/conda/vn/conda-forge/miceforest.svg)](https://anaconda.org/conda-forge/miceforest)
[![PyVersions](https://img.shields.io/pypi/pyversions/miceforest.svg?logo=python&logoColor=white)](https://pypi.org/project/miceforest/)
[![Downloads](https://pepy.tech/badge/miceforest/month)](https://pepy.tech/project/miceforest)

## miceforest: Fast Imputation with lightgbm in Python

<a href='https://github.com/AnotherSamWilson/miceforest'><img src='https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/icon.png' align="right" height="300" /></a>

Fast, memory efficient Multiple Imputation by Chained Equations (MICE)
with random forests. It can impute categorical and numeric data without
much setup, and has an array of diagnostic plots available. The R
version of this package may be found
[here](https://github.com/FarrellDay/miceRanger).

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
      - [Single
        Imputation](https://github.com/AnotherSamWilson/miceforest#Imputing-A-Single-Dataset)
      - [Multiple
        Imputation](https://github.com/AnotherSamWilson/miceforest#Simple-Example-Of-Multiple-Imputation)
      - [Controlling Tree
        Growth](https://github.com/AnotherSamWilson/miceforest#Controlling-Tree-Growth)
      - [Preserving Data
        Assumptions](https://github.com/AnotherSamWilson/miceforest#Preserving-Data-Assumptions)
      - [Imputing With Gradient Boosted
        Trees](https://github.com/AnotherSamWilson/miceforest#Imputing-With-Gradient-Boosted-Trees)
  - [Advanced
    Features](https://github.com/AnotherSamWilson/miceforest#Advanced-Features)
      - [Customizing the Imputation
        Process](https://github.com/AnotherSamWilson/miceforest#Customizing-the-Imputation-Process)
      - [Tuning
        Parameters](https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters)
      - [Imputing New Data with Existing
        Models](https://github.com/AnotherSamWilson/miceforest#Imputing-New-Data-with-Existing-Models)
      - [How to Make the Process
        Faster](https://github.com/AnotherSamWilson/miceforest#How-to-Make-the-Process-Faster)
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

## Package Meta

### News

New major update = 4.0.0.

  - Huge performance improvements, especially if categorical variables
    were being imputed.
  - Ability to tune parameters of models, and use best parameters for
    mice.
  - Improvements to code layout - got rid of ImputationSchema.
  - Raw data is now stored as a numpy array to save space and improve
    indexing.
  - Numpy arrays can be imputed, if you want to avoid pandas.
  - Options of multiple built-in mean matching functions.

### Installation

This package can be installed using either pip or conda, through
conda-forge:

``` bash
# Using pip
$ pip install miceforest

# Using conda
$ conda install -c conda-forge miceforest
```

You can also download the latest development version from this
repository. If you want to install from github with conda, you must
first run `conda install pip git`.

``` bash
$ pip install git+https://github.com/AnotherSamWilson/miceforest.git
```

### Classes

miceforest has 4 main classes which the user will interact with:

  - `KernelDataSet` - a kernel data set is a dataset on which the MICE
    algorithm is performed. Models are saved inside the instance, which
    can also be called on to impute new data. Several plotting methods
    are included to run diagnostics on the imputed data. A
    `KernelDataSet` represents a single imputed dataset.  
  - `MultipleImputedKernel` - a collection of `KernelDataSet`s. Has
    additional methods for accessing and comparing multiple kernel
    datasets together. Subsetting a `MultipleImputedKernel` will result
    in a `KernelDataSet`. Methods of this class tend to affect all of
    the kernel datasets contained within, unless a certain dataset is
    specified.  
  - `ImputedDataSet` - a single dataset that has been imputed. These are
    returned after `impute_new_data()` is called. Has some plotting
    functionality, but contains no models.  
  - `MultipleImputedDataSet` - A collection of datasets that have been
    imputed. Has additional methods for comparing the imputations
    between datasets.

## The Basics

We will be looking at a few simple examples of imputation. We need to
load the packages, and define the data:

``` python
import miceforest as mf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load data and introduce missing values
iris = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
iris['target'] = iris['target'].astype('category')
iris_amp = mf.ampute_data(iris,perc=0.25,random_state=1991)
```

### Imputing a Single Dataset

If you only want to create a single imputed dataset, you can use
`KernelDataSet`:

``` python
# Create kernel. 
kds = mf.KernelDataSet(
  iris_amp,
  save_all_iterations=True,
  random_state=1991
)

# Run the MICE algorithm for 3 iterations
kds.mice(2)

# Return the completed kernel data
completed_data = kds.complete_data()
```

There are also an array of plotting functions available, these are
discussed below in the section [Diagnostic
Plotting](https://github.com/AnotherSamWilson/miceforest#Diagnostic-Plotting).
The plotting behavior between single imputed datasets and multi-imputed
datasets is slightly different.

### Simple Example of Multiple Imputation

In statistics, multiple imputation is a process by which the
uncertainty/other effects caused by missing values can be examined by
creating multiple different imputed datasets. We can create a class
which contains multiple `KernelDataSet`s, along with easy ways to
compare them:

``` python
# Create kernel. 
kernel = mf.MultipleImputedKernel(
  iris_amp,
  datasets=4,
  save_all_iterations=True,
  random_state=1
)

# Run the MICE algorithm for 2 iterations on each of the datasets
kernel.mice(2)

# We can subset out MultipleImputedKernel to get access to the different KernelDataSets:
print(kernel[0])
```

    ##               Class: KernelDataSet
    ##          Iterations: 2
    ##   Imputed Variables: 5
    ## save_all_iterations: True

Printing the `MultipleImputedKernel` object will tell you some high
level information:

``` python
print(kernel)
```

    ##               Class: MultipleImputedKernel
    ##        Models Saved: Last Iteration
    ##            Datasets: 4
    ##          Iterations: 2
    ##   Imputed Variables: 5
    ## save_all_iterations: True

### Controlling Tree Growth

Parameters can be passed directly to lightgbm in several different ways.
Parameters you wish to apply globally to every model can simply be
passed as kwargs to `mice`:

``` python
# Run the MICE algorithm for 1 more iteration on the kernel with new parameters
kernel.mice(1,n_estimators=50)
```

You can also pass pass variable-specific arguments to
`variable_parameters` in mice. For instance, let’s say you noticed the
imputation of the `[target]` column was taking a little longer, because
it is multiclass. You could decrease the n\_estimators specifically for
that column with:

``` python
# Run the MICE algorithm for 2 more iterations on the kernel 
kernel.mice(1,variable_parameters={'target': {'n_estimators': 25}},n_estimators=50)
```

In this scenario, any parameters specified in `variable_parameters`
takes presidence over the kwargs.

### Preserving Data Assumptions

If your data contains count data, or any other data which can be
parameterized by lightgbm, you can simply specify that variable to be
modeled with the corresponding objective function. For example, let’s
pretend `sepal width (cm)` is a count field which can be parameterized
by a Poisson distribution:

``` python
# Create kernel. 
cust_kernel = mf.KernelDataSet(
  iris_amp,
  save_all_iterations=True,
  random_state=1
)

cust_kernel.mice(1, variable_parameters={'sepal width (cm)': {'objective': 'poisson'}})
```

Other nice parameters like `monotone_constraints` can also be passed.

### Imputing with Gradient Boosted Trees

Since any arbitrary parameters can be passed to `lightgbm.train()`, it
is possible to change the algrorithm entirely. This can be done simply
like so:

``` python
# Create kernel. 
kds_gbdt = mf.KernelDataSet(
  iris_amp,
  save_all_iterations=True,
  random_state=1991
)

# We need to add a small minimum hessian, or lightgbm will complain:
kds_gbdt.mice(3, boosting='gbdt', min_sum_hessian_in_leaf=0.01)

# Return the completed kernel data
completed_data = kds_gbdt.complete_data()
```

Note: It is HIGHLY recommended to run parameter tuning if using gradient
boosted trees. The parameter tuning process returns the optimal number
of iterations, and usually results in much more useful parameter sets.
See the section [Tuning
Parameters](https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters)
for more details.

## Advanced Features

There are many ways to alter the imputation procedure to fit your
specific dataset.

### Customizing the Imputation Process

It is possible to heavily customize our imputation procedure by
variable. By passing a named list to `variable_schema`, you can specify
the predictors for each variable to impute. You can also specify
`mean_match_candidates` and `mean_match_subset` by variable by passing a
dict of valid values, with variable names as keys. You can even replace
the entire default mean matching function if you wish:

``` python
var_sch = {
    'sepal width (cm)': ['target','petal width (cm)'],
    'petal width (cm)': ['target','sepal length (cm)']
}
var_mmc = {
    'sepal width (cm)': 5,
    'petal width (cm)': 0
}
var_mms = {
  'sepal width (cm)': 50
}

# The mean matching function requires these parameters, even
# if it does not use them.
def mmf(
  mmc,
  mms,
  model,
  candidate_features,
  bachelor_features,
  candidate_values,
  random_state
):

    bachelor_preds = model.predict(bachelor_features)
    imp_values = random_state.choice(candidate_values, size=bachelor_preds.shape[0])

    return imp_values

cust_kernel = mf.MultipleImputedKernel(
    iris_amp,
    datasets=3,
    variable_schema=var_sch,
    mean_match_candidates=var_mmc,
    mean_match_subset=var_mms,
    mean_match_function=mmf
)
cust_kernel.mice(1)
```

### Tuning Parameters

`miceforest` allows you to tune the parameters on a kernel dataset.
These parameters can then be used to build the models in future
iterations of mice. In its most simple invocation, you can just call the
function with the desired optimization steps:

``` python
# Using the first KernelDataSet in kernel to tune parameters
# with the default settings.
optimal_parameters, losses = kernel[0].tune_parameters(
  optimization_steps=5
)

# Run mice with our newly tuned parameters.
kernel.mice(1, variable_parameters=optimal_parameters)

# The optimal parameters are kept in KernelDataSet.optimal_parameters:
print(kernel[0].optimal_parameters)
```

    ## {0: {'boosting': 'gbdt', 'max_depth': 8, 'num_leaves': 13, 'min_data_in_leaf': 5, 'min_sum_hessian_in_leaf': 0.0001, 'min_gain_to_split': 0.0, 'bagging_fraction': 0.37682973025732225, 'feature_fraction': 1.0, 'feature_fraction_bynode': 0.547703450726179, 'bagging_freq': 1, 'verbosity': -1, 'objective': 'regression', 'seed': 636898, 'learning_rate': 0.05, 'num_iterations': 80}, 1: {'boosting': 'gbdt', 'max_depth': 8, 'num_leaves': 44, 'min_data_in_leaf': 7, 'min_sum_hessian_in_leaf': 0.0001, 'min_gain_to_split': 0.0, 'bagging_fraction': 0.9446955888574227, 'feature_fraction': 1.0, 'feature_fraction_bynode': 0.9736751332478455, 'bagging_freq': 1, 'verbosity': -1, 'objective': 'regression', 'seed': 995364, 'learning_rate': 0.05, 'num_iterations': 98}, 2: {'boosting': 'gbdt', 'max_depth': 8, 'num_leaves': 46, 'min_data_in_leaf': 16, 'min_sum_hessian_in_leaf': 0.0001, 'min_gain_to_split': 0.0, 'bagging_fraction': 0.9418843111072678, 'feature_fraction': 1.0, 'feature_fraction_bynode': 0.8549603507783645, 'bagging_freq': 1, 'verbosity': -1, 'objective': 'regression', 'seed': 417328, 'learning_rate': 0.05, 'num_iterations': 101}, 3: {'boosting': 'gbdt', 'max_depth': 8, 'num_leaves': 42, 'min_data_in_leaf': 15, 'min_sum_hessian_in_leaf': 0.0001, 'min_gain_to_split': 0.0, 'bagging_fraction': 0.5508856111545465, 'feature_fraction': 1.0, 'feature_fraction_bynode': 0.7566610407113622, 'bagging_freq': 1, 'verbosity': -1, 'objective': 'regression', 'seed': 687630, 'learning_rate': 0.05, 'num_iterations': 168}, 4: {'boosting': 'gbdt', 'max_depth': 8, 'num_leaves': 25, 'min_data_in_leaf': 32, 'min_sum_hessian_in_leaf': 0.0001, 'min_gain_to_split': 0.0, 'bagging_fraction': 0.9782829438066223, 'feature_fraction': 1.0, 'feature_fraction_bynode': 0.5178493607898189, 'bagging_freq': 1, 'verbosity': -1, 'objective': 'multiclass', 'num_class': 3, 'seed': 517288, 'learning_rate': 0.05, 'num_iterations': 137}}

This will perform 10 fold cross validation on random samples of
parameters. By default, all variables models are tuned. If you are
curious about the default parameter space that is searched within, check
out the `miceforest.default_lightgbm_parameters` module.

The parameter tuning is pretty flexible. If you wish to set some model
parameters static, or to change the bounds that are searched in, you can
simply pass this information to either the `variable_parameters`
parameter, `**kwbounds`, or both:

``` python
# Using a complicated setup:
optimal_parameters, losses = kernel[0].tune_parameters(
  variables = ['sepal width (cm)','target','petal width (cm)'],
  variable_parameters = {
    'sepal width (cm)': {'bagging_fraction': 0.5},
    'target': {'bagging_freq': (5,10)}
  },
  optimization_steps=5,
  extra_trees = [True, False]
)

kernel.mice(1, variable_parameters=optimal_parameters)
```

In this example, we did a few things - we specified that only `sepal
width (cm)`, `target`, and `petal width (cm)` should be tuned. We also
specified some specific parameters in `variable_parameters.` Notice that
`bagging_fraction` was passed as a scalar, `0.5`. This means that, for
the variable `sepal width (cm)`, the parameter `bagging_fraction` will
be set as that number and not be tuned. We did the opposite for
`bagging_freq`. We specified bounds that the process should search in.
We also passed the argument `extra_trees` as a list. Since it was passed
to \*\*kwbounds, this parameter will apply to all variables that are
being tuned. Passing values as a list tells the process that it should
randomly sample values from the list, instead of treating them as set of
counts to search within.

The tuning process follows these rules for different parameter values it
finds:

  - Scalar: That value is used, and not tuned.  
  - Tuple: Should be length 2. Treated as the lower and upper bound to
    search in.  
  - List: Treated as a distinct list of values to try randomly.

### Imputing New Data with Existing Models

Multiple Imputation can take a long time. If you wish to impute a
dataset using the MICE algorithm, but don’t have time to train new
models, it is possible to impute new datasets using a
`MultipleImputedKernel` object. The `impute_new_data()` function uses
the random forests collected by `MultipleImputedKernel` to perform
multiple imputation without updating the random forest at each
iteration:

``` python
# Our 'new data' is just the first 15 rows of iris_amp
new_data = iris_amp.iloc[range(15)]
new_data_imputed = kernel.impute_new_data(new_data=new_data)
print(new_data_imputed)
```

    ##               Class: MultipleImputedDataSet
    ##            Datasets: 4
    ##          Iterations: 6
    ##   Imputed Variables: 5
    ## save_all_iterations: True

All of the imputation parameters (variable\_schema,
mean\_match\_candidates, etc) will be carried over from the original
`MultipleImputedKernel` object. When mean matching, the candidate values
are pulled from the original kernel dataset. To impute new data, the
`save_models` parameter in `MultipleImputedKernel` must be \> 0. If
`save_models == 1`, the model from the latest iteration is saved for
each variable. If `save_models > 1`, the model from each iteration is
saved. This allows for new data to be imputed in a more similar fashion
to the original mice procedure.

### How to Make the Process Faster

Multiple Imputation is one of the most robust ways to handle missing
data - but it can take a long time. There are several strategies you can
use to decrease the time a process takes to run:

  - Decrease `mean_match_subset`. By default all non-missing datapoints
    are considered as candidates. This can cause the nearest-neighbors
    search to take a long time for large data. A subset of these points
    can be searched instead by using `mean_match_subset`.  
  - Decrease `mean_match_candidates`. If your data is large, the default
    mean\_match\_candidates for each variable is 0.001 \* (\#
    non-missing datapoints). This can grow to be an unweidly large
    number of neighbors that need to be found.  
  - Use different lightgbm parameters. lightgbm is usually not the
    problem, however if a certain variable has a large number of
    classes, then the max number of trees actually grown is (\# classes)
    \* (n\_estimators). You can specifically decrease the bagging
    fraction or n\_estimators for large multi-class variables.  
  - Use a faster mean matching function. The default mean matching
    function uses the scipy.Spatial.KDtree algorithm. There are faster
    alternatives out there, if you think mean matching is the holdup.

## Diagnostic Plotting

As of now, miceforest has four diagnostic plots available.

### Distribution of Imputed-Values

We probably want to know how the imputed values are distributed. We can
plot the original distribution beside the imputed distributions in each
dataset by using the `plot_imputed_distributions` method of an
`MultipleImputedKernel` object:

``` python
kernel.plot_imputed_distributions(wspace=0.3,hspace=0.3)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/distributions.png" width="600px" />

The red line is the original data, and each black line are the imputed
values of each dataset.

### Convergence of Correlation

We are probably interested in knowing how our values between datasets
converged over the iterations. The `plot_correlations` method shows you
a boxplot of the correlations between imputed values in every
combination of datasets, at each iteration. This allows you to see how
correlated the imputations are between datasets, as well as the
convergence over iterations:

``` python
kernel.plot_correlations()
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/plot_corr.png" width="600px" />

### Variable Importance

We also may be interested in which variables were used to impute each
variable. We can plot this information by using the
`plot_feature_importance` method.

``` python
kernel.plot_feature_importance(annot=True,cmap="YlGnBu",vmin=0, vmax=1)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/var_imp.png" width="600px" />

The numbers shown are returned from the
`lightgbm.Booster.feature_importance()` function. Each square represents
the importance of the column variable in imputing the row variable.

### Mean Convergence

If our data is not missing completely at random, we may see that it
takes a few iterations for our models to get the distribution of
imputations right. We can plot the average value of our imputations to
see if this is occurring:

``` python
kernel.plot_mean_convergence(wspace=0.3, hspace=0.4)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/mean_convergence.png" width="600px" />

Our data was missing completely at random, so we don’t see any
convergence occurring here.

## Using the Imputed Data

To return the imputed data simply use the `complete_data` method:

``` python
dataset_1 = kernel.complete_data(0)
```

This will return a single specified dataset. Multiple datasets are
typically created so that some measure of confidence around each
prediction can be created.

Since we know what the original data looked like, we can cheat and see
how well the imputations compare to the original data:

``` python
acclist = []
for iteration in range(kernel.iteration_count()+1):
    target_na_count = kernel.na_counts[4]
    compdat = kernel.complete_data(dataset=0,iteration=iteration)
    
    # Record the accuract of the imputations of target.
    acclist.append(
      round(1-sum(compdat['target'] != iris['target'])/target_na_count,2)
    )

# acclist shows the accuracy of the imputations
# over the iterations.
print(acclist)
```

    ## [0.22, 0.65, 0.7, 0.81, 0.78, 0.86, 0.81]

In this instance, we went from a \~32% accuracy (which is expected with
random sampling) to an accuracy of \~65% after the first iteration. This
isn’t the best example, since our kernel so far has been abused to show
the flexability of the imputation procedure.

## The MICE Algorithm

Multiple Imputation by Chained Equations ‘fills in’ (imputes) missing
data in a dataset through an iterative series of predictive models. In
each iteration, each specified variable in the dataset is imputed using
the other variables in the dataset. These iterations should be run until
it appears that convergence has been met.

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/MICEalgorithm.png" style="display: block; margin: auto;" />

This process is continued until all specified variables have been
imputed. Additional iterations can be run if it appears that the average
imputed values have not converged, although no more than 5 iterations
are usually necessary.

### Common Use Cases

##### **Data Leakage:**

MICE is particularly useful if missing values are associated with the
target variable in a way that introduces leakage. For instance, let’s
say you wanted to model customer retention at the time of sign up. A
certain variable is collected at sign up or 1 month after sign up. The
absence of that variable is a data leak, since it tells you that the
customer did not retain for 1 month.

##### **Funnel Analysis:**

Information is often collected at different stages of a ‘funnel’. MICE
can be used to make educated guesses about the characteristics of
entities at different points in a funnel.

##### **Confidence Intervals:**

MICE can be used to impute missing values, however it is important to
keep in mind that these imputed values are a prediction. Creating
multiple datasets with different imputed values allows you to do two
types of inference:

  - Imputed Value Distribution: A profile can be built for each imputed
    value, allowing you to make statements about the likely distribution
    of that value.  
  - Model Prediction Distribution: With multiple datasets, you can build
    multiple models and create a distribution of predictions for each
    sample. Those samples with imputed values which were not able to be
    imputed with much confidence would have a larger variance in their
    predictions.

### Predictive Mean Matching

`miceforest` can make use of a procedure called predictive mean matching
(PMM) to select which values are imputed. PMM involves selecting a
datapoint from the original, nonmissing data which has a predicted value
close to the predicted value of the missing sample. The closest N
(`mean_match_candidates` parameter) values are chosen as candidates,
from which a value is chosen at random. This can be specified on a
column-by-column basis. Going into more detail from our example above,
we see how this works in practice:

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/PMM.png" style="display: block; margin: auto;" />

This method is very useful if you have a variable which needs imputing
which has any of the following characteristics:

  - Multimodal  
  - Integer  
  - Skewed

### Effects of Mean Matching

As an example, let’s construct a dataset with some of the above
characteristics:

``` python
randst = np.random.RandomState(1991)
# random uniform variable
nrws = 1000
uniform_vec = randst.uniform(size=nrws)

def make_bimodal(mean1,mean2,size):
    bimodal_1 = randst.normal(size=nrws, loc=mean1)
    bimodal_2 = randst.normal(size=nrws, loc=mean2)
    bimdvec = []
    for i in range(size):
        bimdvec.append(randst.choice([bimodal_1[i], bimodal_2[i]]))
    return np.array(bimdvec)

# Make 2 Bimodal Variables
close_bimodal_vec = make_bimodal(2,-2,nrws)
far_bimodal_vec = make_bimodal(3,-3,nrws)


# Highly skewed variable correlated with Uniform_Variable
skewed_vec = np.exp(uniform_vec*randst.uniform(size=nrws)*3) + randst.uniform(size=nrws)*3

# Integer variable correlated with Close_Bimodal_Variable and Uniform_Variable
integer_vec = np.round(uniform_vec + close_bimodal_vec/3 + randst.uniform(size=nrws)*2)

# Make a DataFrame
dat = pd.DataFrame(
    {
    'uniform_var':uniform_vec,
    'close_bimodal_var':close_bimodal_vec,
    'far_bimodal_var':far_bimodal_vec,
    'skewed_var':skewed_vec,
    'integer_var':integer_vec
    }
)

# Ampute the data.
ampdat = mf.ampute_data(dat,perc=0.25,random_state=randst)

# Plot the original data
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.PairGrid(dat)
g.map(plt.scatter,s=5)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/dataset.png" width="600px" style="display: block; margin: auto;" />
We can see how our variables are distributed and correlated in the graph
above. Now let’s run our imputation process twice, once using mean
matching, and once using the model prediction.

``` r
kernelmeanmatch <- mf.MultipleImputedKernel(ampdat,mean_match_candidates=5)
kernelmodeloutput <- mf.MultipleImputedKernel(ampdat,mean_match_candidates=0)

kernelmeanmatch.mice(5)
kernelmodeloutput.mice(5)
```

Let’s look at the effect on the different variables.

##### With Mean Matching

``` python
kernelmeanmatch.plot_imputed_distributions(wspace=0.2,hspace=0.4)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/meanmatcheffects.png" width="600px" style="display: block; margin: auto;" />

##### Without Mean Matching

``` python
kernelmodeloutput.plot_imputed_distributions(wspace=0.2,hspace=0.4)
```

<img src="https://raw.githubusercontent.com/AnotherSamWilson/miceforest/master/examples/nomeanmatching.png" width="600px" style="display: block; margin: auto;" />

You can see the effects that mean matching has, depending on the
distribution of the data. Simply returning the value from the model
prediction, while it may provide a better ‘fit’, will not provide
imputations with a similair distribution to the original. This may be
beneficial, depending on your goal.
