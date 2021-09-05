import pytest
import miceforest as mf
from miceforest.ImputationSchema import _ImputationSchema
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Set random state and load data from sklearn
random_state = np.random.RandomState(1991)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston[3] = boston[3].astype("category")
boston[8] = boston[8].astype("category")
boston.columns = [str(i) for i in boston.columns]

# Several types of datasets are tested:
boston_amp = mf.ampute_data(boston, variables=[str(i) for i in [0,1,9,3,4]], perc=0.25, random_state=random_state)
boston_amp = mf.ampute_data(boston_amp, variables=[str(i) for i in [5,6,7,8]], perc=0.50, random_state=random_state)
boston_amp = mf.ampute_data(boston_amp, variables=[str(i) for i in [2,10,11,12]], perc=0.90, random_state=random_state)

# Ampute only some variables
somevars = ["1", "2", "5", "10"]
boston_amp_somevars = mf.ampute_data(
    boston, variables=somevars, perc=0.25, random_state=random_state
)

# Ampute only 1 variable
onevar = ["1"]
boston_amp_onevar = mf.ampute_data(
    boston, variables=onevar, perc=0.25, random_state=random_state
)


def test_kernel_behavior_default():
    impschem = _ImputationSchema(
        kernel_data=boston_amp,
        impute_data=boston_amp
    )
    assert set(impschem.response_vars) == set(boston_amp.columns)
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    impschem = _ImputationSchema(
        kernel_data=boston_amp_somevars,
        impute_data=boston_amp_somevars
    )
    assert set(impschem.response_vars) == set(somevars)
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    impschem = _ImputationSchema(
        kernel_data=boston_amp_onevar,
        impute_data=boston_amp_onevar,
    )
    assert set(impschem.response_vars) == set(onevar)
    assert set(impschem.predictor_vars) == set(boston_amp.columns) - set(onevar)


def test_kernel_behavior_customs():
    impschem = _ImputationSchema(
        kernel_data=boston_amp,
        impute_data=boston_amp,
        variable_schema=["1", "2", "5"],
        mean_match_candidates=4,
        mean_match_subset=0.5,
        mean_match_function=None,
        imputation_order="descending"
    )
    assert set(impschem.response_vars) == {"1", "2", "5"}
    assert set(impschem.predictor_vars) == set(boston_amp.columns)
    assert impschem.imputation_order == ["2","5","1"]

    # Make sure "2" does not appear in response vars
    # Make sure "1" does not appear in predictor vars, because "2" was removed from response vars.
    impschem = _ImputationSchema(
        kernel_data=boston_amp_onevar,
        impute_data=boston_amp_onevar,
        variable_schema=["1", "2"],
        mean_match_candidates=4,
        mean_match_subset=0.5,
        mean_match_function=None,
        imputation_order="descending"
    )
    assert set(impschem.response_vars) == set(onevar)
    assert set(impschem.predictor_vars) == set(boston.columns) - {"1"}

def test_newdata_custom():
    schem = {"1": ["2", "5"], "2": ["3", "5"], "5": ["6", "7", "8"]}
    mmc = {"1": 3, "2": 4, "5": 5}
    impschem = _ImputationSchema(
        kernel_data=boston_amp,
        impute_data=boston_amp,
        variable_schema=schem,
        mean_match_candidates=mmc,
        imputation_order="descending"
    )
    assert set(impschem.response_vars) == {"1", "2", "5"}
    assert set(impschem.predictor_vars) == {"2", "5", "3", "6", "7", "8"}
    assert impschem.mean_match_candidates == {"1": 3, "2": 4, "5": 5}
    assert impschem.imputation_order == ['2','5','1']

    # 6 has no missing data, make sure it doesn't show up in response_vars
    impschem = _ImputationSchema(
        kernel_data=boston_amp,
        impute_data=boston_amp_somevars,
        variable_schema=["1", "2", "5", "6"],
        mean_match_candidates=None,
        verbose=True
    )
    assert set(impschem.response_vars) == {"1", "2", "5"}
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    # Make sure kernel didn't leak into attributes.
    impschem = _ImputationSchema(
        kernel_data=boston_amp,
        impute_data=boston_amp_onevar,
        variable_schema=["1", "2"],
        mean_match_candidates=4,
        mean_match_subset=0.5,
        mean_match_function=None,
        imputation_order="descending"
    )
    assert set(impschem.response_vars) == set(onevar)
    assert set(impschem.predictor_vars) == set(boston.columns) - {"1"}
    assert impschem.imputation_order == ['1']
    assert impschem.mean_match_candidates == {'1': 4}
    assert all(isinstance(item, int) for item in list(impschem.mean_match_subset.values()))