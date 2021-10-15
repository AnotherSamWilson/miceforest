import numpy as np
from lightgbm import Booster
from .utils import _ensure_np_array

try:
    from scipy.spatial import KDTree
except ImportError:
    raise ImportError(
        "scipy.spatial.KDTree is required for " + "the default mean matching function."
    )


def default_mean_match(
    mmc,
    model: Booster,
    candidate_features,
    bachelor_features,
    candidate_values,
    random_state,
):
    """
    The default mean matching function that comes with miceforest. This can be replaced.
    This function is called upon by other classes to perform mean matching. This function
    is called with all parameters every time. If replacing this function with your own,
    you must include all of the parameters above.

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

    Returns
    -------
    The imputation values
    Must be np.ndarray or shape (n,), where n is the length of dimension 1 of
    bachelor_features.
    If the feature is categorical, return its category code (integer corresponding
    to its category).
    """

    objective = model.params["objective"]
    regressive_objectives = [
        "regression",
        "regression_l1",
        "poisson",
        "huber",
        "fair",
        "mape",
        "cross_entropy",
        "cross_entropy_lambda" "quantile",
        "tweedie",
        "gamma",
    ]
    assert objective in regressive_objectives + [
        "binary",
        "multiclass",
        "multiclassova",
    ], (
        "lightgbm objective not recognized - please check for aliases or "
        + "define a custom mean matching function to handle this objective."
    )

    # Need these no matter what.
    bachelor_preds = model.predict(bachelor_features)

    if mmc == 0:

        if objective in regressive_objectives:

            imp_values = bachelor_preds

        else:

            if objective == "binary":

                imp_values = np.floor(bachelor_preds + 0.5)

            elif objective in ["multiclass", "multiclassova"]:

                imp_values = np.argmax(bachelor_preds, axis=1)

    else:

        if objective in regressive_objectives:

            candidate_preds = model.predict(candidate_features)

            # lightgbm predict for regression is shape (n,).
            # Need it to be shape (n,1)
            bachelor_preds = bachelor_preds.reshape(-1, 1)
            candidate_preds = candidate_preds.reshape(-1, 1)

            # balanced_tree = False fixes a recursion issue for some reason.
            # https://github.com/scipy/scipy/issues/14799
            kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
            _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)

            # We can skip the random selection process if mmc == 1
            if mmc == 1:
                index_choice = knn_indices
            else:
                ind = random_state.randint(mmc, size=(knn_indices.shape[0]))
                index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]

            imp_values = _ensure_np_array(candidate_values)[index_choice]

        elif objective == "binary":

            imp_values = random_state.binomial(n=1, p=bachelor_preds)

        elif objective in ["multiclass", "multiclassova"]:

            # Choose random class weighted by class probabilities (fast)
            bachelor_preds = bachelor_preds.cumsum(axis=1)
            unif = random_state.uniform(0, 1, size=bachelor_preds.shape[0])
            imp_values = [
                np.searchsorted(bachelor_preds[i, :], unif[i])
                for i in range(bachelor_preds.shape[0])
            ]

    return imp_values


def mean_match_kdtree_classification(
    mmc,
    model: Booster,
    candidate_features,
    bachelor_features,
    candidate_values,
    random_state,
):
    """
    This mean matching function selects categorical features by performing nearest
    neighbors on the output class probabilities. This tends to be more accurate, but
    takes more time, especially for variables with large number of classes.

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

    Returns
    -------

    """

    objective = model.params["objective"]
    regressive_objectives = [
        "regression",
        "regression_l1",
        "poisson",
        "huber",
        "fair",
        "mape",
        "cross_entropy",
        "cross_entropy_lambda" "quantile",
        "tweedie",
        "gamma",
    ]
    assert objective in regressive_objectives + [
        "binary",
        "multiclass",
        "multiclassova",
    ], (
        "lightgbm objective not recognized - please check for aliases or "
        + "define a custom mean matching function to handle this objective."
    )

    # Need these no matter what.
    bachelor_preds = model.predict(bachelor_features)

    if mmc == 0:

        if objective in regressive_objectives:

            imp_values = bachelor_preds

        elif objective == "binary":

            imp_values = np.floor(bachelor_preds + 0.5)

        elif objective in ["multiclass", "multiclassova"]:

            imp_values = np.argmax(bachelor_preds, axis=1)

    else:

        if objective in regressive_objectives + ["binary"]:

            candidate_preds = model.predict(candidate_features)

            # lightgbm predict for regression is shape (n,).
            # Need it to be shape (n,1)
            bachelor_preds = bachelor_preds.reshape(-1, 1)
            candidate_preds = candidate_preds.reshape(-1, 1)
            kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
            _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)

            # We can skip the random selection process if mmc == 1
            if mmc == 1:
                index_choice = knn_indices
            else:
                ind = random_state.randint(mmc, size=(knn_indices.shape[0]))
                index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]

            imp_values = _ensure_np_array(candidate_values)[index_choice]

        elif objective in ["multiclass", "multiclassova"]:

            candidate_preds = model.predict(candidate_features)

            kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
            _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)
            ind = random_state.randint(mmc, size=(knn_indices.shape[0]))
            index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]
            imp_values = candidate_values[index_choice]

    return imp_values
