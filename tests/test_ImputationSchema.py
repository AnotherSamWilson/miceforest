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
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)

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


def test_vanilla():
    impschem = _ImputationSchema(
        validation_data=boston_amp, variable_schema=None, mean_match_candidates=None
    )
    assert set(impschem.response_vars) == set(boston_amp.columns)
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    impschem = _ImputationSchema(
        validation_data=boston_amp_somevars,
        variable_schema=None,
        mean_match_candidates=None,
    )
    assert set(impschem.response_vars) == set(somevars)

    impschem = _ImputationSchema(
        validation_data=boston_amp_onevar,
        variable_schema=None,
        mean_match_candidates=None,
    )
    assert set(impschem.response_vars) == set(onevar)


def test_var_schem_list():
    impschem = _ImputationSchema(
        validation_data=boston_amp,
        variable_schema=["1", "2", "5"],
        mean_match_candidates=4,
    )
    assert set(impschem.response_vars) == set(["1", "2", "5"])
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    # 6 has no missing data, make sure it doesn't show up in response_vars
    impschem = _ImputationSchema(
        validation_data=boston_amp_somevars,
        variable_schema=["1", "2", "5", "6"],
        mean_match_candidates=None,
    )
    assert set(impschem.response_vars) == set(["1", "2", "5"])
    assert set(impschem.predictor_vars) == set(boston_amp.columns)

    impschem = _ImputationSchema(
        validation_data=boston_amp_onevar,
        variable_schema=["1"],
        mean_match_candidates=10,
    )
    assert set(impschem.response_vars) == set(onevar)
    bostcols = list(boston.columns)
    bostcols.remove(onevar[0])
    assert set(impschem.predictor_vars) == set(bostcols)


def test_var_schem_dict():
    schem = {"1": ["2", "5"], "2": ["3", "5"], "5": ["6", "7", "8"]}
    mmc = {"1": 3, "2": 4, "5": 5}
    impschem = _ImputationSchema(
        validation_data=boston_amp, variable_schema=schem, mean_match_candidates=mmc
    )
    assert set(impschem.response_vars) == {"1", "2", "5"}
    assert set(impschem.predictor_vars) == {"2", "5", "3", "6", "7", "8"}
    assert impschem.mean_match_candidates

    # 6 has no missing data, make sure it doesn't show up in response_vars
    schem = {"1": ["2", "5"], "2": ["3", "5"], "6": ["10", "7", "8"]}
    mmc = {"1": 3, "3": 4, "5": 5}
    impschem = _ImputationSchema(
        validation_data=boston_amp_somevars,
        variable_schema=schem,
        mean_match_candidates=mmc,
    )
    assert set(impschem.response_vars) == {"1", "2"}
    assert set(impschem.predictor_vars) == {"2", "5", "3"}
    assert set(impschem.mean_match_candidates.keys()) == {"1", "3", "5"}
