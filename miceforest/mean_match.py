from pandas import Series, DataFrame
import inspect
from copy import deepcopy
from lightgbm import Booster
from typing import Callable, Union, Dict, Set, Optional
import numpy as np

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


def adjust_shap_for_rf(model, sv):
    if model.params["boosting"] in ["random_forest", "rf"]:
        sv /= model.current_iteration()


def predict_normal(model: Booster, data):
    preds = model.predict(data)
    return preds


def predict_normal_shap(model: Booster, data):
    preds = model.predict(data, pred_contrib=True)[:, :-1]  # type: ignore
    adjust_shap_for_rf(model, preds)
    return preds


def predict_binary_logodds(model: Booster, data):
    preds = logodds(
        model.predict(data).clip(  # type: ignore
            _LIGHTGBM_PROB_THRESHOLD, 1.0 - _LIGHTGBM_PROB_THRESHOLD
        )
    )
    return preds


def predict_multiclass_logodds(model: Booster, data):
    preds = model.predict(data).clip(  # type: ignore
        _LIGHTGBM_PROB_THRESHOLD, 1.0 - _LIGHTGBM_PROB_THRESHOLD
    )
    preds = logodds(preds)
    return preds


def predict_multiclass_shap(model: Booster, data: DataFrame):
    """
    Returns a 3d array of shape (samples, columns, classes)
    It is faster to copy into a new array than delete from
    the old one.
    """
    preds = model.predict(data, pred_contrib=True)
    samples, cols = data.shape
    classes = model._Booster__num_class  # type: ignore
    p = np.empty(shape=(samples, cols * classes), dtype=preds.dtype)  # type: ignore
    for c in range(classes):
        s1 = slice(c * cols, (c + 1) * cols)
        s2 = slice(c * (cols + 1), (c + 1) * (cols + 1) - 1)
        p[:, s1] = preds[:, s2]  # type: ignore

    # If objective is random forest, the shap values are summed
    # without ever taking an average, so we divide by the iters
    adjust_shap_for_rf(model, p)

    return p
