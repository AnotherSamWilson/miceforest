import pytest
from sklearn.datasets import load_iris
import miceforest as mf
import pandas as pd
import numpy as np

# Make random state and load data
random_state = np.random.RandomState(1991)
iris = pd.concat(load_iris(as_frame=True, return_X_y=True), axis=1)
iris['target'] = iris['target'].astype('category')
iris_amp = mf.ampute_data(iris, perc=0.25, random_state=random_state)

var_sch = {
    'sepal width (cm)': ['target', 'petal width (cm)'],
    'petal width (cm)': ['target', 'sepal length (cm)']
}
var_mmc = {
    'sepal width (cm)': 5,
    'petal width (cm)': 0
}


def test_cskernel():
    kernel = mf.MultipleImputedKernel(
        iris_amp,
        datasets=3,
        variable_schema=var_sch,
        mean_match_candidates=var_mmc,
        save_all_iterations=True,
        random_state=random_state
    )
    assert kernel.get_iterations() == 0
    assert len(kernel.dataset_list) == 3
    assert kernel.categorical_features == ['target']


def test_csmice():
    kernel = mf.MultipleImputedKernel(
        iris_amp,
        datasets=2,
        variable_schema=var_sch,
        mean_match_candidates=var_mmc,
        save_all_iterations=True,
        random_state=random_state
    )
    kernel.mice(2)
    assert kernel.get_iterations() == 2

    compdat = kernel.complete_data()
    assert all(compdat.isna().sum()[
               ['petal width (cm)', 'sepal width (cm)']] == [0, 0])

    featimp = kernel.get_feature_importance()
    assert isinstance(featimp, pd.DataFrame)


def test_csimpute_new():
    kernel = mf.MultipleImputedKernel(
        iris_amp,
        datasets=1,
        variable_schema=var_sch,
        mean_match_candidates=var_mmc,
        save_all_iterations=True,
        random_state=random_state
    )
    kernel.mice(1)
    newdat = iris_amp.iloc[range(25)]
    newdatimp = kernel.impute_new_data(newdat)
    assert isinstance(newdatimp, mf.ImputedDataSet)
    newdatcomp = newdatimp.complete_data()
    assert all(newdatcomp.isna().sum()[
               ['petal width (cm)', 'sepal width (cm)']] == [0, 0])


def test_csget_correlations():
    kernel = mf.MultipleImputedKernel(
        iris_amp,
        datasets=3,
        variable_schema=var_sch,
        mean_match_candidates=var_mmc,
        save_all_iterations=True,
        random_state=random_state
    )
    correlation_dict = kernel.get_correlations()
    assert list(correlation_dict) == ['petal width (cm)', 'sepal width (cm)']


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_custom_schemas.py
    """
    pytest.main([__file__])
