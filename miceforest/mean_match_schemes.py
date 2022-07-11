import numpy as np
from lightgbm import Booster
from .utils import (
    _ensure_np_array,
    logodds,
    _REGRESSIVE_OBJECTIVES,
    _CATEGORICAL_OBJECTIVES,
)

try:
    from scipy.spatial import KDTree
except ImportError:
    raise ImportError(
        "scipy.spatial.KDTree is required for " + "the default mean matching function."
    )


def _mean_match_reg(
    mmc,
    bachelor_preds,
    candidate_preds,
    candidate_values,
    random_state,
    hashed_seeds,
):
    """
    Determines the values of candidates which will be used to impute the bachelors
    """
    num_bachelors = bachelor_preds.shape[0]
    bachelor_preds.shape = (-1, 1)
    candidate_preds.shape = (-1, 1)

    # balanced_tree = False fixes a recursion issue for some reason.
    # https://github.com/scipy/scipy/issues/14799
    kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
    _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)

    # We can skip the random selection process if mmc == 1
    if mmc == 1:
        index_choice = knn_indices
    else:
        # Use the random_state if seed_array was not passed. Faster
        if hashed_seeds is None:
            ind = random_state.randint(mmc, size=(num_bachelors))
        # Use the random_seed_array if it was passed. Deterministic.
        else:
            ind = hashed_seeds % mmc

        index_choice = knn_indices[np.arange(num_bachelors), ind]

    imp_values = _ensure_np_array(candidate_values)[index_choice]

    return imp_values


def _mean_match_binary_fast(bachelor_preds, random_state, hashed_seeds):
    num_bachelors = bachelor_preds.shape[0]
    if hashed_seeds is None:
        imp_values = random_state.binomial(n=1, p=bachelor_preds)
    else:
        imp_values = []
        for i in range(num_bachelors):
            np.random.seed(seed=hashed_seeds[i])
            imp_values.append(np.random.binomial(n=1, p=bachelor_preds[i]))

        imp_values = np.array(imp_values)

    return imp_values


def _mean_match_multiclass_fast(bachelor_preds, random_state, hashed_seeds):
    """
    Choose random class weighted by class probabilities (fast)
    """
    num_bachelors = bachelor_preds.shape[0]

    # Turn bachelor_preds into discrete cdf:
    bachelor_preds = bachelor_preds.cumsum(axis=1)

    # Randomly choose uniform numbers 0-1
    if hashed_seeds is None:
        unif = random_state.uniform(0, 1, size=num_bachelors)
    else:
        unif = []
        for i in range(num_bachelors):
            np.random.seed(seed=hashed_seeds[i])
            unif.append(np.random.uniform(0, 1, size=1)[0])
        unif = np.array(unif)

    # Choose classes according to their cdf.
    # Distribution will match probabilities
    imp_values = [
        np.searchsorted(bachelor_preds[i, :], unif[i]) for i in range(num_bachelors)
    ]

    return imp_values


def _mean_match_multiclass_accurate(
    mmc,
    bachelor_preds,
    candidate_preds,
    candidate_values,
    random_state,
    hashed_seeds,
):
    """
    Performs nearest neighbors search on class probabilities
    """
    num_bachelors = bachelor_preds.shape[0]
    kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
    _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)

    # Come up with random numbers 0-mmc, with priority given to hashed_seeds
    if hashed_seeds is None:
        ind = random_state.randint(mmc, size=(num_bachelors))
    else:
        ind = hashed_seeds % mmc

    index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]
    imp_values = candidate_values[index_choice]

    return imp_values


def mean_match_function_default(
    mmc,
    model,
    bachelor_features,
    candidate_values,
    random_state,
    hashed_seeds,
    candidate_preds=None,
):
    """
    The default mean matching function that comes with miceforest. This can be replaced.
    This function is called upon by other classes to perform mean matching. This function
    is called with all parameters every time. If replacing this function with your own,
    you must include all of the parameters above.

    This function is build for speed, but may be less accurate for categorical variables.

        .. code-block:: text

            Mean match procedure for different data types:
                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, return class based on random draw weighted by
                        class probability for each sample.
                Numeric or binary:
                    If mmc = 0, the predicted value is used
                    If mmc > 0, obtain the mmc closest candidate
                        predictions and collect the associated
                        real candidate values. Choose 1 randomly.

    Parameters
    ----------
    mmc: int
        The number of mean matching candidates (derived from mean_match_candidates
        parameter)
    model: lgb.Booster
        The model that was trained.
    candidate_features: pd.DataFrame or np.ndarray
        The features used to train the model.
        If mmc == 0, this will be None.
    bachelor_features: pd.DataFrame or np.ndarray
        The features corresponding to the missing values of the response variable
        used to train the model.
    candidate_values:  pd.Series or np.ndarray
        The real (not predicted) values of the candidates from the original dataset.
        Will be 1D.
        If the feature is pandas categorical, this will be the category codes.
    random_state: np.random.RandomState
        The random state from the process calling this function is passed.
    hashed_seeds: None, np.ndarray (int32)
        Used to make imputations deterministic at the record level. If this array
        is passed, random_state is ignored in favor of these seeds. These seeds are
        derived as a hash of the random_seed_array passed to the imputation functions.
        The distribution of these seeds is uniform enough.

    Returns
    -------
    The imputation values
    Must be np.ndarray or shape (n,), where n is the length of dimension 1 of bachelor_features.
    If the feature is categorical, return its category code (integer corresponding to its category).
    """

    objective = model.params["objective"]
    assert objective in _REGRESSIVE_OBJECTIVES + _CATEGORICAL_OBJECTIVES, (
        "lightgbm objective not recognized - please check for aliases or "
        + "define a custom mean matching function to handle this objective."
    )

    # Need these no matter what.
    bachelor_preds = model.predict(bachelor_features)

    # mmc = 0 is deterministic
    if mmc == 0:

        if objective in _REGRESSIVE_OBJECTIVES:

            imp_values = bachelor_preds

        else:

            if objective == "binary":

                imp_values = np.floor(bachelor_preds + 0.5)

            elif objective in ["multiclass", "multiclassova"]:

                imp_values = np.argmax(bachelor_preds, axis=1)

    else:

        if objective in _REGRESSIVE_OBJECTIVES:

            imp_values = _mean_match_reg(
                mmc,
                bachelor_preds,
                candidate_preds,
                candidate_values,
                random_state,
                hashed_seeds,
            )

        elif objective == "binary":

            imp_values = _mean_match_binary_fast(
                bachelor_preds, random_state, hashed_seeds
            )

        elif objective in ["multiclass", "multiclassova"]:

            imp_values = _mean_match_multiclass_fast(
                bachelor_preds, random_state, hashed_seeds
            )

    return imp_values


def mean_match_function_kdtree_cat(
    mmc,
    model: Booster,
    bachelor_features,
    candidate_values,
    random_state,
    hashed_seeds,
    candidate_preds=None,
):
    """
    This mean matching function selects categorical features by performing nearest
    neighbors on the output class probabilities. This tends to be more accurate, but
    takes more time, especially for variables with large number of classes.

    This function is slower for categorical datatypes, but results in better imputations.

        .. code-block:: text

            Mean match procedure for different datatypes:
                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, get N nearest neighbors from class probabilities.
                        Select 1 at random.
                Numeric:
                    If mmc = 0, the predicted value is used
                    If mmc > 0, obtain the mmc closest candidate
                        predictions and collect the associated
                        real candidate values. Choose 1 randomly.

    Parameters
    ----------
    mmc: int
        The number of mean matching candidates (derived from mean_match_candidates parameter)
    model: lgb.Booster
        The model that was trained.
    candidate_features: pd.DataFrame or np.ndarray
        The features used to train the model.
        If mmc == 0, this will be None.
    bachelor_features: pd.DataFrame or np.ndarray
        The features corresponding to the missing values of the response variable used to train
        the model.
    candidate_values:  pd.Series or np.ndarray
        The real (not predicted) values of the candidates from the original dataset.
        Will be 1D
        If the feature is pandas categorical, this will be the category codes.
    random_state: np.random.RandomState
        The random state from the process calling this function is passed.
    hashed_seeds: None, np.ndarray (int32)
        Used to make imputations deterministic at the record level. If this array
        is passed, random_state is ignored in favor of these seeds. These seeds are
        derived as a hash of the random_seed_array passed to the imputation functions.
        The distribution of these seeds is uniform enough.

    Returns
    -------
    The imputation values
    Must be np.ndarray or shape (n,), where n is the length of dimension 1 of bachelor_features.
    If the feature is categorical, return its category code (integer corresponding to its category).

    """

    objective = model.params["objective"]
    assert objective in _REGRESSIVE_OBJECTIVES + _CATEGORICAL_OBJECTIVES, (
        "lightgbm objective not recognized - please check for aliases or "
        + "define a custom mean matching function to handle this objective."
    )

    # Need these no matter what.
    bachelor_preds = model.predict(bachelor_features)

    if mmc == 0:

        if objective in _REGRESSIVE_OBJECTIVES:

            imp_values = bachelor_preds

        elif objective == "binary":

            imp_values = np.floor(bachelor_preds + 0.5)

        elif objective in ["multiclass", "multiclassova"]:

            imp_values = np.argmax(bachelor_preds, axis=1)

    else:

        if objective in _REGRESSIVE_OBJECTIVES:

            imp_values = _mean_match_reg(
                mmc,
                bachelor_preds,
                candidate_preds,
                candidate_values,
                random_state,
                hashed_seeds,
            )

        elif objective == "binary":

            bachelor_preds = logodds(bachelor_preds)

            imp_values = _mean_match_reg(
                mmc,
                bachelor_preds,
                candidate_preds,
                candidate_values,
                random_state,
                hashed_seeds,
            )

        elif objective in ["multiclass", "multiclassova"]:

            # inner_predict returns a flat array, need to reshape for KDTree
            bachelor_preds = logodds(bachelor_preds)

            imp_values = _mean_match_multiclass_accurate(
                mmc,
                bachelor_preds,
                candidate_preds,
                candidate_values,
                random_state,
                hashed_seeds,
            )

    return imp_values


# Candidate predictions are only calculated for the objectives listed
# specifically in candidate_preds_objectives. This saves time and space
# if we compile the predictions.
mean_match_scheme_fast_cat = {
    "mean_match_function": mean_match_function_default,
    "candidate_preds_objectives": _REGRESSIVE_OBJECTIVES,
}

mean_match_scheme_default = {
    "mean_match_function": mean_match_function_kdtree_cat,
    "candidate_preds_objectives": _REGRESSIVE_OBJECTIVES + _CATEGORICAL_OBJECTIVES,
}
