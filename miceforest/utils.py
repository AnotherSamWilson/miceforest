from typing import Dict, List, Optional, Union

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame, Series


def get_best_int_downcast(x: int):
    assert isinstance(x, int)
    int_dtypes = ["uint8", "uint16", "uint32", "uint64"]
    np_iinfo_max = {dtype: np.iinfo(dtype).max for dtype in int_dtypes}
    for dtype, max in np_iinfo_max.items():
        if x <= max:
            break
        if dtype == "uint64":
            raise ValueError("Number too large to downcast")
    return dtype


def ampute_data(
    data: DataFrame,
    variables: Optional[List[str]] = None,
    perc: float = 0.1,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
):
    """
    Ampute Data

    Returns a copy of data with specified variables amputed.

    Parameters
    ----------
     data : Pandas DataFrame
        The data to ampute

     variables : None or list
        If None, are variables are amputed.

     perc : double
        The percentage of the data to ampute.

    random_state: None, int, or np.random.RandomState
        The random state to use.

    Returns
    -------
    pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()
    num_rows = amputed_data.shape[0]
    amp_rows = int(perc * num_rows)
    random_state = ensure_rng(random_state)
    variables = list(data.columns) if variables is None else variables

    for col in variables:
        ind = random_state.choice(amputed_data.index, size=amp_rows, replace=False)
        amputed_data.loc[ind, col] = np.nan

    return amputed_data


def stratified_subset(
    y: Series,
    size: int,
    groups: int,
    random_state: Optional[Union[int, np.random.RandomState]],
):
    """
    Subsample y using stratification. y is divided into quantiles,
    and then elements are randomly chosen from each quantile to
    come up with the subsample.

    Parameters
    ----------
    y: np.ndarray
        The variable to use for stratification
    size: int
        How large the subset should be
    groups: int
        How many groups to break y into. The more groups, the more
        balanced (but less random) y will be
    cat: bool
        Is y already categorical? If so, we can skip the group creation
    seed: int
        The random seed to use.

    Returns
    -------
    The indices of y that have been chosen.

    """

    random_state = ensure_rng(random_state=random_state)

    cat = False
    if y.dtype.name == "category":
        cat = True
        y = y.cat.codes
    y = y.to_numpy()

    if cat:
        digits = y
    else:
        q = [x / groups for x in range(1, groups)]
        bins = np.quantile(y, q)
        digits = np.digitize(y, bins, right=True)

    digits_v, digits_c = np.unique(digits, return_counts=True)
    digits_i = np.arange(digits_v.shape[0])
    digits_p = digits_c / digits_c.sum()
    digits_s = (digits_p * size).round(0).astype("int32")
    diff = size - digits_s.sum()
    if diff != 0:
        digits_fix = random_state.choice(
            digits_i, size=abs(diff), p=digits_p, replace=False
        )
        if diff < 0:
            for d in digits_fix:
                digits_s[d] -= 1
        else:
            for d in digits_fix:
                digits_s[d] += 1

    sub = np.zeros(shape=size).astype("int32")
    added = 0
    for d_i in digits_i:
        d_v = digits_v[d_i]
        n = digits_s[d_i]
        ind = np.where(digits == d_v)[0]
        choice = random_state.choice(ind, size=n, replace=False)
        sub[added : (added + n)] = choice
        added += n

    sub.sort()

    return sub


def stratified_continuous_folds(y: Series, nfold: int):
    """
    Create primitive stratified folds for continuous data.
    Should be digestible by lightgbm.cv function.
    """
    y = y.to_numpy()
    elements = y.shape[0]
    assert elements >= nfold, "more splits then elements."
    sorted = np.argsort(y)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    for v in val:
        yield (np.setdiff1d(np.arange(elements), v), v)


def stratified_categorical_folds(y: Series, nfold: int):
    """
    Create primitive stratified folds for categorical data.
    Should be digestible by lightgbm.cv function.
    """
    assert isinstance(y, Series), "y must be a pandas Series"
    assert y.dtype.name[0:3].lower() == "int", "y should be the category codes"
    y = y.to_numpy()
    elements = len(y)
    uniq, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    assert elements >= nfold, "more splits then elements."
    if any(counts < nfold):
        print("Decreasing nfold to lowest categorical level count...")
        nfold = min(counts)
    sorted = np.argsort(inv)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    for v in val:
        yield (np.setdiff1d(range(elements), v), v)


# https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
# This hash performs well enough in testing.
def hash_int32(x: np.ndarray):
    """
    A hash function which generates random uniform (enough)
    int32 integers. Used in mean matching and initialization.
    """
    assert isinstance(x, np.ndarray)
    assert x.dtype in ["uint32", "int32"], "x must be int32"
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


def hash_uint64(x: np.ndarray):
    assert isinstance(x, np.ndarray)
    assert x.dtype == "uint64", "x must be uint64"
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x


def hash_numpy_int_array(x: np.ndarray, ind: Union[np.ndarray, slice] = slice(None)):
    """
    Deterministically set the values of the elements in x
    at the locations ind to some uniformly distributed number
    within the range of the datatype of x.

    This function acts on x in place
    """
    assert isinstance(x, np.ndarray)
    if x.dtype in ["uint32", "int32"]:
        x[ind] = hash_int32(x[ind])
    elif x.dtype == "uint64":
        x[ind] = hash_uint64(x[ind])
    else:
        raise ValueError("random_seed_array must be uint32, int32, or uint64 datatype")


def _draw_random_int32(random_state, size):
    nums = random_state.randint(
        low=0, high=np.iinfo("int32").max, size=size, dtype="int32"
    )
    return nums


def ensure_rng(random_state) -> RandomState:
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = RandomState()
    elif isinstance(random_state, int):
        random_state = RandomState(random_state)
    else:
        assert isinstance(random_state, RandomState)
    return random_state


# def _ensure_iterable(x):
#     """
#     If the object is iterable, return the object.
#     Else, return the object in a length 1 list.
#     """
#     return x if hasattr(x, "__iter__") else [x]


# def _assert_dataset_equivalent(ds1: _t_dat, ds2: _t_dat):
#     if isinstance(ds1, DataFrame):
#         assert isinstance(ds2, DataFrame)
#         assert ds1.equals(ds2)
#     else:
#         assert isinstance(ds2, np.ndarray)
#         np.testing.assert_array_equal(ds1, ds2)


# def _ensure_np_array(x):
#     if isinstance(x, np.ndarray):
#         return x
#     if isinstance(x, DataFrame) | isinstance(x, Series):
#         return x.values
#     else:
#         raise ValueError("Can't cast to numpy array")


def _expand_value_to_dict(default, value, keys) -> dict:
    if isinstance(value, dict):
        ret = {key: value.get(key, default) for key in keys}
    else:
        assert default.__class__ == value.__class__
        ret = {key: value for key in keys}

    return ret


def _list_union(x: List, y: List):
    return [z for z in x if z in y]


def logodds(probability):
    try:
        odds_ratio = probability / (1 - probability)
        log_odds = np.log(odds_ratio)
    except ZeroDivisionError:
        raise ValueError(
            "lightgbm output a probability of 1.0 or 0.0. "
            "This is usually because of rare classes. "
            "Try adjusting min_data_in_leaf."
        )

    return log_odds


def logistic_function(log_odds):
    return 1 / (1 + np.exp(-log_odds))
