import pytest
import miceforest as mf
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Set random state and load data from sklearn
random_state = np.random.RandomState(1991)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston[3] = boston[3].astype('category')
boston[8] = boston[8].astype('category')
boston.columns = [str(i) for i in boston.columns]
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)


def test_kernel():
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=3,
        save_all_iterations=True,
        random_state=random_state
    )
    assert kernel.get_iterations() == 0
    assert len(kernel.dataset_list) == 3
    assert kernel.categorical_features == ['3', '8']


def test_mice():
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=4,
        save_all_iterations=False,
        random_state=random_state
    )
    kernel.mice(3)
    assert kernel.get_iterations() == 3

    compdat = kernel.complete_data()
    assert all(compdat.isna().sum() == 0)

    featimp = kernel.get_feature_importance()
    assert isinstance(featimp, pd.DataFrame)

    # Throw plotting in here because creating kernel is expensive
    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance()
    kernel.plot_correlations()


def test_impute_new():
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=1,
        save_all_iterations=True,
        random_state=random_state
    )
    kernel.mice(1)
    newdat = boston_amp.iloc[range(25)]
    newdatimp = kernel.impute_new_data(newdat)
    assert isinstance(newdatimp, mf.ImputedDataSet)
    newdatcomp = newdatimp.complete_data()
    assert all(newdatcomp.isna().sum() == 0)


def test_get_correlations():
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=3,
        save_all_iterations=True,
        random_state=random_state
    )
    correlation_dict = kernel.get_correlations()
    assert list(correlation_dict) == sorted(list(set(boston.columns) -
                                                 set(kernel.categorical_features)))


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_basic.py
    """
    pytest.main([__file__])
