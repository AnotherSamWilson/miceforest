from miceforest.utils import (
    _draw_random_int32,
    _expand_value_to_dict,
    _list_union,
    ampute_data,
    ensure_rng,
    get_best_int_downcast,
    hash_numpy_int_array,
    logodds,
    logistic_function,
    stratified_categorical_folds,
    stratified_continuous_folds,
    stratified_subset,
)
import pytest
import numpy as np
import pandas as pd


def test_get_best_int_downcast_selects_smallest_dtype():
    assert get_best_int_downcast(12) == "uint8"
    assert get_best_int_downcast(np.iinfo("uint8").max) == "uint8"
    assert get_best_int_downcast(np.iinfo("uint16").max) == "uint16"
    assert get_best_int_downcast(np.iinfo("uint32").max) == "uint32"
    with pytest.raises(ValueError):
        get_best_int_downcast(np.iinfo("uint64").max + 1)


def test_ampute_data_reproducible_and_scoped():
    data = pd.DataFrame(
        {
            "a": np.arange(6, dtype="float64"),
            "b": np.arange(6, 12, dtype="float64"),
            "c": np.arange(12, 18, dtype="float64"),
        }
    )
    amputed_one = ampute_data(data, variables=["a", "b"], perc=0.5, random_state=7)
    amputed_two = ampute_data(data, variables=["a", "b"], perc=0.5, random_state=7)

    assert amputed_one.equals(amputed_two)
    assert amputed_one["a"].isna().sum() == 3
    assert amputed_one["b"].isna().sum() == 3
    assert amputed_one["c"].isna().sum() == 0
    assert amputed_one.isna().sum().sum() == 6


def test_hash_numpy_int_array_mutates_and_validates_dtype():
    arr = np.array([1, 2, 3], dtype="int32")
    baseline = arr.copy()
    hash_numpy_int_array(arr)
    assert not np.array_equal(arr, baseline)
    assert arr.dtype == baseline.dtype

    arr_uint = np.array([1, 2, 3], dtype="uint64")
    baseline_uint = arr_uint.copy()
    hash_numpy_int_array(arr_uint)
    assert not np.array_equal(arr_uint, baseline_uint)

    arr_float = np.array([1.0, 2.0], dtype="float64")
    with pytest.raises(ValueError):
        hash_numpy_int_array(arr_float)


def test_draw_random_int32_respects_bounds_and_dtype():
    rng = np.random.RandomState(4)
    samples = _draw_random_int32(rng, size=10)
    assert samples.dtype == np.int32
    assert np.all(samples >= 0)
    assert np.all(samples <= np.iinfo("int32").max)


def test_expand_value_to_dict_with_scalar_and_partial_dict():
    keys = ["x", "y"]
    scalar_result = _expand_value_to_dict(0, 5, keys)
    assert scalar_result == {"x": 5, "y": 5}

    dict_result = _expand_value_to_dict(0, {"x": 1}, keys)
    assert dict_result == {"x": 1, "y": 0}


def test_list_union_returns_intersection_preserving_order():
    assert _list_union(["a", "b", "c"], ["b", "d", "c"]) == ["b", "c"]


def test_logodds_and_logistic_are_inverses_for_valid_probability():
    probability = 0.2
    logits = logodds(probability)
    recovered = logistic_function(logits)
    assert pytest.approx(recovered) == probability

    with pytest.raises(ValueError):
        logodds(1.0)


def test_stratified_continuous_folds_cover_all_indices():
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    folds = list(stratified_continuous_folds(y, nfold=4))

    assert len(folds) == 4
    all_validation = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(all_validation), np.arange(len(y)))
    for train, val in folds:
        assert len(np.intersect1d(train, val)) == 0
        assert len(val) == len(y) // 4


def test_stratified_categorical_folds_adjusts_fold_count(capsys):
    y = pd.Series([0, 0, 0, 1, 1, 2], dtype="int64")
    folds = list(stratified_categorical_folds(y, nfold=3))
    captured = capsys.readouterr()

    assert "Decreasing nfold" in captured.out
    assert len(folds) == 1
    train, val = folds[0]
    assert len(val) == len(y)
    assert len(train) == 0


def test_stratified_categorical_folds_balanced_counts():
    y = pd.Series([0, 0, 1, 1, 2, 2], dtype="int64")
    folds = list(stratified_categorical_folds(y, nfold=2))

    assert len(folds) == 2
    all_val = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(all_val), np.arange(len(y)))



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
