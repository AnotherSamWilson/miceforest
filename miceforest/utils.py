from .compat import pd_DataFrame, pd_Series, pd_read_parquet
import numpy as np
from numpy.random import RandomState
import blosc
import dill
from typing import Union, List, Dict, Optional


_t_var_list = Union[List[str], List[int]]
_t_var_dict = Union[Dict[str, List[str]], Dict[int, List[int]]]
_t_var_sub = Union[Dict[Union[int, int], Union[int, float]]]
_t_dat = Union[pd_DataFrame, np.ndarray]
_t_random_state = Union[int, RandomState, None]


def ampute_data(
    data: _t_dat,
    variables: Optional[_t_var_list] = None,
    perc: float = 0.1,
    random_state: _t_random_state = None,
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
    data_shape = amputed_data.shape
    amp_rows = int(perc * data_shape[0])
    random_state = ensure_rng(random_state)

    if len(data_shape) > 1:
        if variables is None:
            variables = [i for i in range(amputed_data.shape[1])]
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                assert isinstance(
                    data, pd_DataFrame
                ), "np array was passed but variables are strings"
                variables = [data.columns.tolist().index(i) for i in variables]

        if isinstance(amputed_data, pd_DataFrame):
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data.iloc[na_ind, v] = np.NaN

        if isinstance(amputed_data, np.ndarray):
            amputed_data = amputed_data.astype("float64")
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data[na_ind, v] = np.NaN

    else:

        na_ind = random_state.choice(
            np.arange(data_shape[0]), replace=False, size=amp_rows
        )
        amputed_data[na_ind] = np.NaN

    return amputed_data


def load_kernel(filepath: str, n_threads: Optional[int] = None):
    """
    Loads a kernel that was saved using save_kernel().

    Parameters
    ----------
    filepath: str
        The filepath of the saved kernel

    n_threads: int
        The threads to use for decompression. By default, all threads are used.

    Returns
    -------
    ImputationKernel
    """
    n_threads = blosc.detect_number_of_cores() if n_threads is None else n_threads
    blosc.set_nthreads(n_threads)
    with open(filepath, "rb") as f:
        kernel = dill.loads(blosc.decompress(dill.load(f)))

    if kernel.original_data_class == "pd_DataFrame":
        kernel.working_data = pd_read_parquet(kernel.working_data)
        for col in kernel.working_data.columns:
            kernel.working_data[col] = kernel.working_data[col].astype(
                kernel.working_dtypes[col]
            )

    return kernel


def stratified_subset(y, size, groups, cat, seed):
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
    rs = RandomState(seed)

    if isinstance(y, pd_Series):
        if y.dtype.name == "category":
            y = y.cat.codes
        y = y.values

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
        digits_fix = rs.choice(digits_i, size=abs(diff), p=digits_p, replace=False)
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
        choice = rs.choice(ind, size=n, replace=False)
        sub[added : (added + n)] = choice
        added += n

    sub.sort()

    return sub


def stratified_continuous_folds(y, nfold):
    """
    Create primitive stratified folds for continuous data.
    Should be digestible by lightgbm.cv function.
    """
    if isinstance(y, pd_Series):
        y = y.values
    elements = len(y)
    assert elements >= nfold, "more splits then elements."
    sorted = np.argsort(y)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    for v in val:
        yield (np.setdiff1d(range(elements), v), v)


def stratified_categorical_folds(y, nfold):
    """
    Create primitive stratified folds for categorical data.
    Should be digestible by lightgbm.cv function.
    """
    if isinstance(y, pd_Series):
        y = y.values
    y = y.reshape(
        y.shape[0],
    ).copy()
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
# We don't really need to worry that much about diffusion
# since we take % n at the end, and n (mmc) is usually
# very small. This hash performs well enough in testing.
def hash_int32(x):
    """
    A hash function which generates random uniform (enough)
    int32 integers. Used in mean matching and initialization.
    """
    assert isinstance(x, np.ndarray)
    assert x.dtype == "int32", "x must be int32"
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


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


def _ensure_iterable(x):
    """
    If the object is iterable, return the object.
    Else, return the object in a length 1 list.
    """
    return x if hasattr(x, "__iter__") else [x]


def _assert_dataset_equivalent(ds1: _t_dat, ds2: _t_dat):
    if isinstance(ds1, pd_DataFrame):
        assert isinstance(ds2, pd_DataFrame)
        assert ds1.equals(ds2)
    else:
        assert isinstance(ds2, np.ndarray)
        np.testing.assert_array_equal(ds1, ds2)


def _ensure_np_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd_DataFrame) | isinstance(x, pd_Series):
        return x.values
    else:
        raise ValueError("Can't cast to numpy array")


def _interpret_ds(val, avail_can):
    if isinstance(val, int):
        assert val <= avail_can, "data subset is more than available candidates"
    elif isinstance(val, float):
        assert (val <= 1.0) and (val > 0.0), "if float, 0.0 < data_subset <= 1.0"
        val = int(val * avail_can)
    else:
        raise ValueError("malformed data_subset passed")
    return val


def _dict_set_diff(iter1, iter2) -> Dict[int, List[int]]:
    """
    Returns a dict, where the elements in iter1 are
    the keys, and the values are the set differences
    between the key and the values of iter2.
    """
    ret = {int(y): [int(x) for x in iter2 if int(x) != int(y)] for y in iter1}
    return ret


def _slice(dat, row_slice=slice(None), col_slice=slice(None)):
    """
    Returns a view of the subset data if possible.
    """

    if isinstance(dat, pd_DataFrame):
        return dat.iloc[row_slice, col_slice]
    elif isinstance(dat, np.ndarray):
        return dat[row_slice, col_slice]
    else:
        raise ValueError("Unknown data class passed.")


def _assign_col_values_without_copy(dat, row_ind, col_ind, val):
    """
    Insert values into different data frame objects.
    """

    row_ind = _ensure_iterable(row_ind)

    if isinstance(dat, pd_DataFrame):
        # Remove iterable attribute if
        # we are only assigning 1 value
        if len(val) == 1:
            val = val[0]

        dat.iloc[row_ind, col_ind] = val

    elif isinstance(dat, np.ndarray):
        val.shape = -1
        dat[row_ind, col_ind] = val

    else:
        raise ValueError("Unknown data class passed.")


def _subset_data(dat, row_ind=None, col_ind=None, return_1d=False):
    """
    Can subset data along 2 axis.
    Explicitly returns a copy.
    """

    row_ind = range(dat.shape[0]) if row_ind is None else row_ind
    col_ind = range(dat.shape[1]) if col_ind is None else col_ind

    if isinstance(dat, pd_DataFrame):
        data_copy = dat.iloc[row_ind, col_ind]
        return data_copy.to_numpy().flatten() if return_1d else data_copy
    elif isinstance(dat, np.ndarray):
        row_ind = _ensure_iterable(row_ind)
        col_ind = _ensure_iterable(col_ind)
        data_copy = dat[np.ix_(row_ind, col_ind)]
        return data_copy.flatten() if return_1d else data_copy
    else:
        raise ValueError("Unknown data class passed.")


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
