import pytest
import numpy as np
from miceforest.default_mean_match import default_mean_match
from pandas import CategoricalDtype


def test_simple_regression_mmc_0():
    ret = default_mean_match(
        mmc=0,
        candidate_preds=np.array([1,2,3,4,5]),
        bachelor_preds=np.array([1.1,3.2]),
        candidate_values=np.array([94,95,96,97,98]),
        random_state=np.random.RandomState(1),
        cat_dtype=None
    )
    assert all(np.array([1.1,3.2]) == ret)


def test_simple_regression_mmc_1():
    ret = default_mean_match(
        mmc=1,
        candidate_preds=np.array([1,2,3,4,5]),
        bachelor_preds=np.array([1.1,3.2]),
        candidate_values=np.array([94,95,96,97,98]),
        random_state=np.random.RandomState(1),
        cat_dtype=None
    )
    assert all(np.array([94,96]) == ret)


def test_simple_binary_mmc_0():
    ret = default_mean_match(
        mmc=0,
        candidate_preds=np.array([0.2,0.1,0.4,0.5,0.9]),
        bachelor_preds=np.array([0.3,0.5]),
        candidate_values=np.array([0,1]),
        random_state=np.random.RandomState(1),
        cat_dtype=CategoricalDtype(categories=['a','b'])
    )
    assert all(np.array(['a','a']) == ret)


def test_simple_binary_mmc_1():
    ret = default_mean_match(
        mmc=1,
        candidate_preds=np.array([0.2,0.1,0.4,0.5,0.9]),
        bachelor_preds=np.array([0.3,0.5]),
        candidate_values=np.array([0,1]),
        random_state=np.random.RandomState(1),
        cat_dtype=CategoricalDtype(categories=['a','b'])
    )
    assert all(np.array(['a', 'b']) == ret)


def test_simple_multiclass_mmc_0():

    ret = default_mean_match(
        mmc=0,
        candidate_preds=np.array([[0.1,0.8,0.1],[0.3,0.3,0.4],[0.5,0.5,0]]),
        bachelor_preds=np.array([[0.7,0.2,0.1],[0.9,0.05,0.05],[0.1,0.1,0.8]]),
        candidate_values=np.array(['b','c','a']),
        random_state=np.random.RandomState(1),
        cat_dtype=CategoricalDtype(categories=['a','b','c'])
    )

    assert all(ret == np.array(['a','a','c']))


def test_simple_multiclass_mmc_1():

    ret = []
    seed = np.random.randint(0,100000,1)[0]

    for i in range(1000):
        ret.append(
            default_mean_match(
                mmc=1,
                candidate_preds=np.array([[0.1,0.8,0.1],[0.3,0.3,0.4],[0.5,0.5,0]]),
                bachelor_preds=np.array([[0.7,0.2,0.1],[0.9,0.05,0.05],[0.1,0.1,0.8]]),
                candidate_values=np.array(['b','c','a']),
                random_state=np.random.RandomState(seed * i),
                cat_dtype=CategoricalDtype(categories=['a','b','c'])
            )
        )
    _, counts = np.unique(np.array(ret)[:,0], return_counts=True)

    # The probability of this happening is astronomically low if the function is working correctly
    assert abs(counts/1000 - np.array([0.7,0.2,0.1])).sum() <= 0.1