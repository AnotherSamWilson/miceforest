"""
Built-in mean matching schemes.
These schemes vary in their speed and accuracy.
"""

from typing import Dict, Callable
from .MeanMatchScheme import (
    MeanMatchScheme,
    _REGRESSIVE_OBJECTIVES,
    _CATEGORICAL_OBJECTIVES,
    _DEFAULT_MMC,
)
from .builtin_pred_funcs import (
    predict_binary_logodds,
    predict_multiclass_logodds,
    predict_multiclass_shap,
    predict_normal,
    predict_normal_shap,
)
from .builtin_mean_match_functions import (
    _mean_match_reg,
    _mean_match_binary_accurate,
    _mean_match_binary_fast,
    _mean_match_multiclass_accurate,
    _mean_match_multiclass_fast,
)

_DEFAULT_OBJECTIVE_DTYPES = {
    **{o: "float16" for o in _CATEGORICAL_OBJECTIVES},
    **{o: "float32" for o in _REGRESSIVE_OBJECTIVES},
}


##################################
##### DEFAULT MEAN MATCHING SCHEME

_MEAN_MATCH_FUNCTIONS_DEFAULT: Dict[str, Callable] = {
    "binary": _mean_match_binary_accurate,
    "multiclass": _mean_match_multiclass_accurate,
    "multiclassova": _mean_match_multiclass_accurate,
    **{o: _mean_match_reg for o in _REGRESSIVE_OBJECTIVES},
}
_LGB_PRED_FUNCTIONS_DEFAULT: Dict[str, Callable] = {
    "binary": predict_binary_logodds,
    "multiclass": predict_multiclass_logodds,
    "multiclassova": predict_multiclass_logodds,
    **{o: predict_normal for o in _REGRESSIVE_OBJECTIVES},
}
mean_match_default = MeanMatchScheme(
    mean_match_candidates=_DEFAULT_MMC,
    mean_match_functions=_MEAN_MATCH_FUNCTIONS_DEFAULT,
    lgb_model_pred_functions=_LGB_PRED_FUNCTIONS_DEFAULT,
    objective_pred_dtypes=_DEFAULT_OBJECTIVE_DTYPES,
)
mean_match_default.__doc__ = """
Built-in instance of miceforest.MeanMatchScheme.

This scheme is of medium speed and accuracy.

The rules are:
    Categorical:
        If mmc = 0, the class with the highest probability is chosen.
        If mmc > 0, run K-Nearest-Neighbors search on candidate class
        probabilities, and choose 1 neighbor randomly for each bachelor.
        Use the candidate value of the associated selection to impute.
    Numeric:
        If mmc = 0, the predicted value is used
        If mmc > 0, run K-Nearest-Neighbors search on candidate
        predictions, and choose 1 neighbor randomly for each bachelor.
        Use the candidate value of the associated selection to impute.
"""


###################################
##### FAST CAT MEAN MATCHING SCHEME

_MEAN_MATCH_FUNCTIONS_FAST_CAT: Dict[str, Callable] = {
    "binary": _mean_match_binary_fast,
    "multiclass": _mean_match_multiclass_fast,
    "multiclassova": _mean_match_multiclass_fast,
    **{o: _mean_match_reg for o in _REGRESSIVE_OBJECTIVES},
}
_LGB_PRED_FUNCTIONS_FAST_CAT: Dict[str, Callable] = {
    "binary": predict_normal,
    "multiclass": predict_normal,
    "multiclassova": predict_normal,
    **{o: predict_normal for o in _REGRESSIVE_OBJECTIVES},
}
mean_match_fast_cat = MeanMatchScheme(
    mean_match_candidates=_DEFAULT_MMC,
    mean_match_functions=_MEAN_MATCH_FUNCTIONS_FAST_CAT,
    lgb_model_pred_functions=_LGB_PRED_FUNCTIONS_FAST_CAT,
    objective_pred_dtypes=_DEFAULT_OBJECTIVE_DTYPES,
)
mean_match_fast_cat.__doc__ = """
Built-in instance of miceforest.MeanMatchScheme.

This scheme is faster for categorical variables 
specifically, but may not be as accurate. 

The rules are:
    Categorical:
        If mmc = 0, the class with the highest probability is chosen.
        If mmc > 0, return class based on random draw weighted by
            class probability for each sample.
    Numeric:
        If mmc = 0, the predicted value is used
        If mmc > 0, obtain the mmc closest candidate predictions and 
        collect the associated real candidate values. Choose 1 randomly.
"""


###############################
##### SHAP MEAN MATCHING SCHEME

_MEAN_MATCH_FUNCTIONS_SHAP: Dict[str, Callable] = {
    "binary": _mean_match_binary_accurate,
    "multiclass": _mean_match_multiclass_accurate,
    "multiclassova": _mean_match_multiclass_accurate,
    **{o: _mean_match_reg for o in _REGRESSIVE_OBJECTIVES},
}
_LGB_PRED_FUNCTIONS_SHAP: Dict[str, Callable] = {
    "binary": predict_normal_shap,
    "multiclass": predict_multiclass_shap,
    "multiclassova": predict_multiclass_shap,
    **{o: predict_normal_shap for o in _REGRESSIVE_OBJECTIVES},
}
mean_match_shap = MeanMatchScheme(
    mean_match_candidates=_DEFAULT_MMC,
    mean_match_functions=_MEAN_MATCH_FUNCTIONS_SHAP,
    lgb_model_pred_functions=_LGB_PRED_FUNCTIONS_SHAP,
    objective_pred_dtypes=_DEFAULT_OBJECTIVE_DTYPES,
)
mean_match_shap.__doc__ = """
Built-in instance of miceforest.MeanMatchScheme.

This scheme has the lowest speed and 
highest accuracy on high dimension data.

The rules are:
    Categorical:
        If mmc = 0, the class with the highest probability is chosen.
        If mmc > 0, run K-Nearest-Neighbors search on candidate shap
        values, and choose 1 neighbor randomly for each bachelor.
        Use the candidate value of the associated selection to impute.
    Numeric:
        If mmc = 0, the predicted value is used
        If mmc > 0, run K-Nearest-Neighbors search on candidate
        shap values, and choose 1 neighbor randomly for each bachelor.
        Use the candidate value of the associated selection to impute.
"""
