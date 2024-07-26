from miceforest.utils import stratified_subset
import numpy as np
import pandas as pd


def test_subset():

    strat_std_closer = []
    strat_mean_closer = []
    for i in range(1000):
        y = pd.Series(np.random.normal(size=1000))
        size = 100
        ss_ind = stratified_subset(y, size, groups=10, random_state=i)
        y_strat_sub = y[ss_ind]
        y_rand_sub = np.random.choice(y, size, replace=False)

        # See which random sample has a closer stdev
        y_strat_std_diff = abs(y.std() - y_strat_sub.std())
        y_rand_std_diff = abs(y.std() - y_rand_sub.std())
        strat_std_closer.append(y_strat_std_diff < y_rand_std_diff)

        # See which random sample has a closer mean
        y_strat_mean_diff = abs(y.mean() - y_strat_sub.mean())
        y_rand_mean_diff = abs(y.mean() - y_rand_sub.mean())
        strat_mean_closer.append(y_strat_mean_diff < y_rand_mean_diff)

    # Assert that the mean and stdev of the
    # stratified random draws are closer to the
    # original distribution over 50% of the time.
    assert np.array(strat_std_closer).mean() > 0.5
    assert np.array(strat_mean_closer).mean() > 0.5


def test_subset_continuous_reproduce():
    # Tests for reproducibility in numeric stratified subsetting
    for i in range(100):
        y = pd.Series(np.random.normal(size=1000))
        size = 100

        ss1 = stratified_subset(y, size, groups=10, random_state=i)
        ss2 = stratified_subset(y, size, groups=10, random_state=i)

        assert np.all(ss1 == ss2)


def test_subset_categorical_reproduce():
    # Tests for reproducibility in categorical stratified subsetting
    for i in range(100):
        y = pd.Series(np.random.randint(low=1, high=10, size=1000)).astype("category")
        size = 100

        ss1 = stratified_subset(y, size, groups=10, random_state=i)
        ss2 = stratified_subset(y, size, groups=10, random_state=i)

        assert np.all(ss1 == ss2)
