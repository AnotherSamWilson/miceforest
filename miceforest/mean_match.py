
from pandas import Series
import inspect
from copy import deepcopy
from lightgbm import Booster
from typing import Callable, Union, Dict, Set, Optional
import numpy as np
from scipy.spatial import KDTree
from .utils import logodds


# Lightgbm can output 0.0 probabilities for extremely
# rare categories. This causes logodds to return inf.
_LIGHTGBM_PROB_THRESHOLD = 0.00000001


_REGRESSIVE_OBJECTIVES = [
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

_CATEGORICAL_OBJECTIVES = [
    "binary",
    "multiclass",
    "multiclassova",
]


def _to_2d(x):
    if x.ndim == 1:
        x.shape = (-1, 1)


def mean_match_reg(
    mean_match_candidates: int,
    bachelor_preds: np.ndarray,
    candidate_preds: np.ndarray,
    candidate_values: np.ndarray,
    random_state: np.random.RandomState,
    hashed_seeds: Optional[np.ndarray],
):
    """
    Determines the values of candidates which will be used to impute the bachelors
    """

    if mean_match_candidates == 0:
        imp_values = bachelor_preds

    else:
        _to_2d(bachelor_preds)
        _to_2d(candidate_preds)

        num_bachelors = bachelor_preds.shape[0]

        # balanced_tree = False fixes a recursion issue for some reason.
        # https://github.com/scipy/scipy/issues/14799
        kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
        _, knn_indices = kd_tree.query(
            bachelor_preds, k=mean_match_candidates, workers=-1
        )

        # We can skip the random selection process if mean_match_candidates == 1
        if mean_match_candidates == 1:
            index_choice = knn_indices

        else:
            # Use the random_state if seed_array was not passed. Faster
            if hashed_seeds is None:
                ind = random_state.randint(mean_match_candidates, size=(num_bachelors))
            # Use the random_seed_array if it was passed. Deterministic.
            else:
                ind = hashed_seeds % mean_match_candidates

            index_choice = knn_indices[np.arange(num_bachelors), ind]

        imp_values = np.array(candidate_values)[index_choice]

    return imp_values


def mean_match_binary_accurate(
    mean_match_candidates: int,
    bachelor_preds: np.ndarray,
    candidate_preds: np.ndarray,
    candidate_values: np.ndarray,
    random_state: np.random.RandomState,
    hashed_seeds: Optional[np.ndarray],
):
    """
    Determines the values of candidates which will be used to impute the bachelors.
    This function works just like the regression version - chooses candidates with
    close probabilities to the bachelor prediction.
    """

    return mean_match_reg(
        mean_match_candidates,
        bachelor_preds,
        candidate_preds,
        candidate_values,
        random_state,
        hashed_seeds,
    )


def mean_match_binary_fast(
    mean_match_candidates: int,
    bachelor_preds: np.ndarray,
    random_state: np.random.RandomState,
    hashed_seeds: Optional[np.ndarray],
):
    """
    Chooses 0/1 randomly weighted by probability obtained from prediction.
    If mean_match_candidates is 0, choose class with highest probability.
    """
    if mean_match_candidates == 0:
        imp_values = np.floor(bachelor_preds + 0.5)

    else:
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


def mean_match_multiclass_fast(
    mean_match_candidates: int,
    bachelor_preds: np.ndarray,
    random_state: np.random.RandomState,
    hashed_seeds: Optional[np.ndarray],
):
    """
    If mean_match_candidates is 0, choose class with highest probability.
    Otherwise, randomly choose class weighted by class probabilities.
    """
    if mean_match_candidates == 0:
        imp_values = np.argmax(bachelor_preds, axis=1)

    else:
        num_bachelors = bachelor_preds.shape[0]

        # Turn bachelor_preds into discrete cdf:
        bachelor_preds = bachelor_preds.cumsum(axis=1)

        # Randomly choose uniform numbers 0-1
        if hashed_seeds is None:
            # This is the fastest way to adjust for numeric
            # imprecision of float16 dtype. Actually ends up
            # barely taking any time at all.
            bp_dtype = bachelor_preds.dtype
            unif = np.minimum(
                random_state.uniform(0, 1, size=num_bachelors).astype(bp_dtype),
                bachelor_preds[:, -1],
            )
        else:
            unif = []
            for i in range(num_bachelors):
                np.random.seed(seed=hashed_seeds[i])
                unif.append(np.random.uniform(0, 1, size=1)[0])
            unif = np.array(unif)

        # Choose classes according to their cdf.
        # Distribution will match probabilities
        imp_values = np.array(
            [
                np.searchsorted(bachelor_preds[i, :], unif[i])
                for i in range(num_bachelors)
            ]
        )

    return imp_values


def mean_match_multiclass_accurate(
    mean_match_candidates: int,
    bachelor_preds: np.ndarray,
    candidate_preds: np.ndarray,
    candidate_values: np.ndarray,
    random_state: np.random.RandomState,
    hashed_seeds: Optional[np.ndarray],
):
    """
    Performs nearest neighbors search on class probabilities.
    """
    if mean_match_candidates == 0:
        return np.argmax(bachelor_preds, axis=1)

    else:
        return mean_match_reg(
        mean_match_candidates,
        bachelor_preds,
        candidate_preds,
        candidate_values,
        random_state,
        hashed_seeds,
    )


def adjust_shap_for_rf(model, sv):
    if model.params["boosting"] in ["random_forest", "rf"]:
        sv /= model.current_iteration()


def predict_normal(model: Booster, data):
    preds = model.predict(data)
    return preds


def predict_normal_shap(model: Booster, data):
    preds = model.predict(data, pred_contrib=True)[:, :-1] # type: ignore
    adjust_shap_for_rf(model, preds)
    return preds


def predict_binary_logodds(model: Booster, data):
    preds = logodds(
        model.predict(data).clip( # type: ignore
            _LIGHTGBM_PROB_THRESHOLD, 1.0 - _LIGHTGBM_PROB_THRESHOLD
        )
    )
    return preds


def predict_multiclass_logodds(model: Booster, data):
    preds = model.predict(data).clip( # type: ignore
        _LIGHTGBM_PROB_THRESHOLD, 1.0 - _LIGHTGBM_PROB_THRESHOLD
    )
    preds = logodds(preds)
    return preds


def predict_multiclass_shap(model: Booster, data):
    """
    Returns a 3d array of shape (samples, columns, classes)
    It is faster to copy into a new array than delete from
    the old one.
    """
    preds = model.predict(data, pred_contrib=True)
    samples, cols = data.shape
    classes = model._Booster__num_class # type: ignore
    p = np.empty(shape=(samples, cols * classes), dtype=preds.dtype) # type: ignore
    for c in range(classes):
        s1 = slice(c * cols, (c + 1) * cols)
        s2 = slice(c * (cols + 1), (c + 1) * (cols + 1) - 1)
        p[:, s1] = preds[:, s2] # type: ignore

    # If objective is random forest, the shap values are summed
    # without ever taking an average, so we divide by the iters
    adjust_shap_for_rf(model, p)

    return p
