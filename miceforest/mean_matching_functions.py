import numpy as np
from lightgbm import Booster

try:
    from scipy.spatial import KDTree
except ImportError:
    raise ImportError(
        "scipy.spatial.KDTree is required for " + "the default mean matching function."
    )


def default_mean_match(
    mmc: int,
    mms: int,
    model: Booster,
    candidate_features: np.ndarray,
    bachelor_features: np.ndarray,
    candidate_values: np.ndarray,
    random_state: np.random.RandomState,
):
    """
    The default mean matching function that comes with miceforest. This can be replaced.
    This function is called upon by other classes to perform mean matching. This function
    is called with all parameters every time. If replacing this function with your own,
    you must include all of the parameters above.

    Parameters
    ----------
    mmc: The number of mean matching candidates (derived from mean_match_candidates parameter)
    mms: The number of samples to include in mean matching (derived from mean_match_subset parameter)
    model: The model that was trained.
    candidate_features: The features used to train the model.
    bachelor_features: The features corresponding to the missing values of the response variable
        used to train the model.
    candidate_values: The real (not predicted) values of the candidates from the original dataset.
    random_state: The random state from the process calling this function is passed.

    Returns
    -------
    The imputation values
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

        if objective in regressive_objectives:

            # If we need to subset
            candidate_count = len(candidate_values)
            if mms < candidate_count:
                candidate_subset = random_state.choice(
                    range(candidate_count), size=mms, replace=False
                )
                candidate_preds = model.predict(candidate_features[candidate_subset, :])
                candidate_values = candidate_values[candidate_subset]
            else:
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

            imp_values = candidate_values[index_choice]

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

        else:

            raise ValueError(
                "lightgbm objective not recognized - please check for aliases or "
                + "define a custom mean matching function to handle this objective."
            )

    return imp_values


def mean_match_kdtree_classification(
    mmc: int,
    mms: int,
    model: Booster,
    candidate_features: np.ndarray,
    bachelor_features: np.ndarray,
    candidate_values: np.ndarray,
    random_state: np.random.RandomState,
):
    """
    This mean matching function selects categorical features by performing nearest
    neighbors on the output class probabilities. This tends to be more accurate, but
    takes more time, especially for variables with large number of classes.

    Parameters
    ----------
    mmc: The number of mean matching candidates (derived from mean_match_candidates parameter)
    mms: The number of samples to include in mean matching (derived from mean_match_subset parameter)
    model: The model that was trained.
    candidate_features: The features used to train the model.
    bachelor_features: The features corresponding to the missing values of the response variable
        used to train the model.
    candidate_values: The real (not predicted) values of the candidates from the original dataset.
    random_state: The random state from the process calling this function is passed.

    Returns
    -------
    The imputation values
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

    # Need these no matter what.
    bachelor_preds = model.predict(bachelor_features)

    if mmc == 0:

        if objective in regressive_objectives:

            imp_values = bachelor_preds

        elif objective == "binary":

            # Fastest method I could find.
            # Beats transpose + np.argmax (like multiclass)
            # Beats np.round somehow.
            imp_values = np.floor(bachelor_preds + 0.5)

        elif objective in ["multiclass", "multiclassova"]:

            imp_values = np.argmax(bachelor_preds, axis=1)

    else:

        if objective in regressive_objectives + ["binary"]:

            # If we need to subset
            candidate_count = len(candidate_values)
            if mms < candidate_count:
                candidate_subset = random_state.choice(
                    range(candidate_count), size=mms, replace=False
                )
                candidate_preds = model.predict(candidate_features[candidate_subset, :])
                candidate_values = candidate_values[candidate_subset]
            else:
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

            imp_values = candidate_values[index_choice]

        elif objective in ["multiclass", "multiclassova"]:

            # Perform nearest neighbors on class probabilities (accurate)
            candidate_count = len(candidate_values)
            if mms < candidate_count:
                candidate_subset = random_state.choice(
                    range(candidate_count), size=mms, replace=False
                )
                candidate_preds = model.predict(candidate_features[candidate_subset, :])
                candidate_values = candidate_values[candidate_subset]
            else:
                candidate_preds = model.predict(candidate_features)

            kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
            _, knn_indices = kd_tree.query(bachelor_preds, k=mmc, workers=-1)
            ind = random_state.randint(mmc, size=(knn_indices.shape[0]))
            index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]
            imp_values = candidate_values[index_choice]

        else:

            raise ValueError(
                "lightgbm objective not recognized - please check for aliases or "
                + "define a custom mean matching function to handle this objective."
            )

    return imp_values
