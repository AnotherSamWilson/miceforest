"""
Default prediction functions that come with miceforest.
"""
from .utils import logodds
from lightgbm import Booster
import numpy as np


# Lightgbm can output 0.0 probabilities for extremely
# rare categories. This causes logodds to return inf.
_LIGHTGBM_PROB_THRESHOLD = 0.00000001


def _adjust_shap_for_rf(model, sv):
    if model.params["boosting"] in ["random_forest", "rf"]:
        sv /= model.current_iteration()


def predict_normal(model: Booster, data):
    preds = model.predict(data)
    return preds


def predict_normal_shap(model: Booster, data):
    preds = model.predict(data, pred_contrib=True)[:, :-1]
    _adjust_shap_for_rf(model, preds)
    return preds


def predict_binary_logodds(model: Booster, data):
    preds = logodds(
        model.predict(data).clip(
            _LIGHTGBM_PROB_THRESHOLD,
            1.0 - _LIGHTGBM_PROB_THRESHOLD
        )
    )
    return preds


def predict_multiclass_logodds(model: Booster, data):
    preds = model.predict(data).clip(
        _LIGHTGBM_PROB_THRESHOLD,
        1.0 - _LIGHTGBM_PROB_THRESHOLD
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
    classes = model._Booster__num_class
    p = np.empty(shape=(samples, cols * classes), dtype=preds.dtype)
    for c in range(classes):
        s1 = slice(c * cols, (c + 1) * cols)
        s2 = slice(c * (cols + 1), (c + 1) * (cols + 1) - 1)
        p[:, s1] = preds[:, s2]

    # If objective is random forest, the shap values are summed
    # without ever taking an average, so we divide by the iters
    _adjust_shap_for_rf(model, p)

    return p
