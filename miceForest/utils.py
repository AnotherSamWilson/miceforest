import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


def ampute_data(data,vars=None,perc=0.1,random_state=None):
    """
    Ampute Data

    Returns a copy of data with specified variables amputed.

    Parameters
    ----------
     data : Pandas DataFrame
        The data to ampute
     vars : None or list
        If None, are variables are amputed.
     perc : double
        The percentage of the data to ampute.

    Returns
    -------
    amputed_data : pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()

    random_state = ensure_rng(random_state)

    if vars is None:
        vars = list(amputed_data.columns)

    nrow = amputed_data.shape[0]
    amp_rows = int(perc*nrow)

    for v in vars:
        na_ind = random_state.choice(range(nrow),replace=False,size=amp_rows)
        amputed_data.loc[na_ind,v] = np.NaN

    return amputed_data


def ensure_rng(random_state=None):
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


# These exist so we can make a default classifier with the same parameters
# as those that may be passed to **kw_fit
def _default_rf_classifier(random_state,max_features="sqrt",**kw_fit):
    rfc = RandomForestClassifier(
        random_state=random_state,
        max_features=max_features,
        **kw_fit
    )
    return rfc


def _default_rf_regressor(random_state,max_features="sqrt",**kw_fit):
    rfc = RandomForestRegressor(
        random_state=random_state,
        max_features=max_features,
        **kw_fit
    )
    return rfc


def _distinct_from_list(lst):
    output = []
    for item in lst:
        if item not in output:
            output.append(item)
    return output