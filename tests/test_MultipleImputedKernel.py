import pytest
import miceforest as mf
from miceforest.utils import _get_default_mmc
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Set random state and load data from sklearn
random_state = np.random.RandomState(1991)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston[3] = boston[3].astype("category")
boston[8] = boston[8].astype("category")
boston.columns = [str(i) for i in boston.columns]
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)


def test_kernel():
    kernel = mf.MultipleImputedKernel(
        boston_amp, save_all_iterations=True, random_state=random_state
    )
    assert kernel.iteration_count() == 0
    assert kernel.categorical_variables == ["3", "8"]
    assert kernel.mean_match_candidates == {
        i: _get_default_mmc() for i in boston_amp.columns
    }


def test_mice():
    schem = {"3": ["2", "5"], "2": ["3", "5"], "5": ["6", "7", "8"]}
    mmc = {"3": 3, "2": 4, "5": 5}
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=5,
        variable_schema=schem,
        mean_match_candidates=mmc,
        save_all_iterations=True,
        random_state=random_state,
    )
    kernel.mice(3)
    assert kernel.iteration_count() == 3

    compdat = kernel.complete_data(dataset=0)
    assert all(compdat[["3", "2", "5"]].isna().sum() == 0)

    compdat1 = kernel.complete_data(dataset=1, iteration=1)
    assert all(compdat1[["3", "2", "5"]].isna().sum() == 0)

    featimp = kernel.get_feature_importance()
    assert isinstance(featimp, pd.DataFrame)

    kernel.remove(1)
    assert kernel.dataset_count() == 4

    # Throw plotting in here because creating kernel is expensive
    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance()
    kernel.plot_mean_convergence()
    kernel.plot_correlations()


def test_impute_new():
    schem = {"3": ["2", "5"], "2": ["3", "5"], "5": ["6", "7", "8"]}
    mmc = {"3": 3, "2": 4, "5": 5}
    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=1,
        variable_schema=schem,
        mean_match_candidates=mmc,
        save_all_iterations=True,
        random_state=random_state,
    )
    kernel.mice(4)
    newdat = boston_amp.iloc[range(25)]
    newdatimp = kernel.impute_new_data(newdat)
    assert isinstance(newdatimp, mf.ImputedDataSet)
    newdatcomp = newdatimp.complete_data()
    assert all(newdatcomp[["3", "2", "5"]].isna().sum() == 0)

    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance()
    kernel.plot_mean_convergence()
