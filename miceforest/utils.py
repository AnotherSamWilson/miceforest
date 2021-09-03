import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from pandas import DataFrame


MeanMatchType = Union[int, float, Dict[str, float], Dict[str, int]]
VarSchemType = Optional[Union[List[str], Dict[str, List[str]]]]


def ampute_data(
    data: "DataFrame",
    variables: Optional[List[str]] = None,
    perc: float = 0.1,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> "DataFrame":
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

    Returns
    -------
    pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()
    nrow = amputed_data.shape[0]
    amp_rows = int(perc * nrow)
    random_state = ensure_rng(random_state)

    if variables is None:
        variables = list(amputed_data.columns)

    for v in variables:
        na_ind = random_state.choice(range(nrow), replace=False, size=amp_rows)
        amputed_data.loc[na_ind, v] = np.NaN

    return amputed_data


def ensure_rng(
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> np.random.RandomState:
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


def _var_comparison(variables: Optional[List[str]], comparison: List[str]) -> List[str]:
    """
    If variables is None, set it equal to the comparison list
    Else, make sure all of variables are in comparison list.
    """
    if variables is None:
        variables = comparison
    elif any([v not in comparison for v in variables]):
        raise ValueError("Action not permitted on supplied variables.")
    return variables


def _copy_and_remove(lst, elements):
    lt = lst.copy()
    for element in elements:
        lt.remove(element)
    return lt


def _get_default_mmc(candidates=None) -> int:
    if candidates is None:
        return 5
    else:
        percent = 0.001
        minimum = 5
        mean_match_candidates = max(minimum, int(percent * candidates))
        return mean_match_candidates


def _get_default_mms(candidates) -> int:
    return int(candidates)


def _list_union(a, b):
    return [element for element in a if element in b]


def _setequal(a, b):
    if not hasattr(a, "__iter__"):
        return a == b
    else:
        return set(a) == set(b)


# Check for n_estimators aliases that might confuse lightgbm
disallowed_aliases_n_estimators = [
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "num_boost_round",
]
