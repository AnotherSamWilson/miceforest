
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf

# Define data
iris = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
iris.rename({'target': 'species'}, inplace=True, axis=1)
iris["species"] = iris['species'].map({0: 'aa', 1: 'bb', 2: 'cc'}).astype('category')
amp_iris = mf.ampute_data(iris,perc=0.25)

def test_kernel_defaults():
    kernel = mf.KernelDataSet(amp_iris, random_state=1991)
    kernel.mice(2)
    acc = (iris.loc[kernel.na_where['species'],'species'] == kernel.imputation_values['species'][2]).mean()
    assert acc > 0.6

    newdat = amp_iris.loc[range(25)]
    newimp = kernel.impute_new_data(newdat)
    compdat = newimp.complete_data()
    acc = (iris.loc[range(25),'species'] == compdat['species']).mean()
    assert acc > 0.6

    kernel = mf.MultipleImputedKernel(amp_iris,datasets=2,random_state=1991)
    kernel.mice(2)
    acc = (iris.loc[kernel.na_where['species'], 'species'] == kernel[0].imputation_values['species'][2]).mean()
    assert acc > 0.6
