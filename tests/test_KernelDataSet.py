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


def test_kernel_initialization():
    kernel = mf.KernelDataSet(
        boston_amp, save_all_iterations=True, random_state=random_state
    )
    assert kernel.iteration_count() == 0
    assert kernel.categorical_variables == ["3", "8"]
    assert kernel.mean_match_candidates == {
        i: _get_default_mmc() for i in boston_amp.columns
    }


def test_mice():
    kernel = mf.KernelDataSet(
        boston_amp, save_all_iterations=True, random_state=random_state
    )
    kernel.mice(3, n_estimators=10, verbose=True)
    assert kernel.iteration_count() == 3

    compdat = kernel.complete_data()
    assert all(compdat.isna().sum() == 0)

    compdat1 = kernel.complete_data(iteration=1)
    assert all(compdat1.isna().sum() == 0)

    featimp = kernel.get_feature_importance()
    assert isinstance(featimp, pd.DataFrame)

    # Throw plotting in here because creating kernel is expensive
    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance()
    kernel.plot_mean_convergence()


def test_cust_schem_1():
    mmc = {"1": 4, "2": 0.01, "3": 0}
    mms = {"2": 100, "3": 0.5}
    vs = {"1": ["2","3","4","5"], "2": ["6","7"], "3": ["1","2","8"]}
    def mmf(
            mmc,
            candidate_preds,
            bachelor_preds,
            candidate_values,
            cat_dtype,
            random_state):
        return random_state.choice(candidate_values, size=bachelor_preds.shape[0])
    kernel = mf.KernelDataSet(
        boston_amp,
        variable_schema=vs,
        mean_match_candidates=mmc,
        mean_match_subset=mms,
        mean_match_function=mmf,
        save_all_iterations=True,
        random_state=random_state
    )

    assert kernel.mean_match_candidates == {'1': 4, '2': 3, '3': 0}, "mean_match_candidates initialization failed"
    assert kernel.mean_match_subset == {'1': 380, '2': 100, '3': 190}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == ["3", "8"], "categorical recognition failed."

    kernel.mice(3, n_estimators=10)
    assert kernel.iteration_count() == 3, "iteration counting is incorrect."

    compdat = kernel.complete_data()
    assert all(compdat[["1","2","3"]].isnull().sum() == 0)


def test_impute_new():
    schem = {"3": ["2", "5","8"], "2": ["3", "5"], "5": ["6", "7", "8"], "8": ["1","3","5"]}
    mmc = {"3": 3, "2": 4, "5": 5}
    kernel = mf.KernelDataSet(
        boston_amp,
        variable_schema=schem,
        mean_match_candidates=mmc,
        save_all_iterations=True,
        random_state=random_state,
    )
    kernel.mice(3, n_estimators=10)
    new_data = boston_amp.iloc[range(25)]
    newdatimp = kernel.impute_new_data(new_data)
    assert isinstance(newdatimp, mf.ImputedDataSet)
    newdatcomp = newdatimp.complete_data()
    assert all(newdatcomp[["3", "2", "5", "8"]].isna().sum() == 0)

    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance()
    kernel.plot_mean_convergence()
