import numpy as np

from .utils import _draw_random_int32

# A few parameters that generally benefit
# from searching in the log space.
_LOG_SPACE_SEARCH = [
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "lambda_l1",
    "lambda_l2",
    "cat_l2",
    "cat_smooth",
    "path_smooth",
    "min_gain_to_split",
]


# THESE VALUES WILL ALWAYS BE USED WHEN VALUES ARE NOT PASSED BY USER.
# seed is always set by the calling processes _random_state.
# These need to be main parameter names, not aliases
_DEFAULT_LGB_PARAMS = {
    "boosting": "random_forest",
    "data_sample_strategy": "bagging",
    "num_iterations": 48,
    "max_depth": 8,
    "num_leaves": 128,
    "min_data_in_leaf": 1,
    "min_sum_hessian_in_leaf": 0.01,
    "min_gain_to_split": 0.0,
    "bagging_fraction": 0.632,
    "feature_fraction_bynode": 0.632,
    "bagging_freq": 1,
    "verbosity": -1,
}


def _sample_parameters(parameters: dict, random_state, parameter_sampling_method: str):
    """
    Searches through a parameter set and selects a random
    number between the values in any provided tuple of length 2.
    """
    assert (
        parameter_sampling_method == "random"
    ), "Only random parameter sampling is supported right now."
    parameters = parameters.copy()
    for p, v in parameters.items():
        if isinstance(v, list):
            choice = random_state.choice(v)
        elif isinstance(v, tuple):
            assert (
                len(v) == 2
            ), "Tuples passed must be length 2, representing the bounds."
            assert v[0] < v[1], f"{p} lower bound > upper bound"
            if p in _LOG_SPACE_SEARCH:
                choice = np.exp(
                    random_state.uniform(
                        np.log(v[0]),
                        np.log(v[1]),
                        size=1,
                    )[0]
                )
            else:
                choice = random_state.uniform(
                    v[0],
                    v[1],
                    size=1,
                )[0]
            if isinstance(v[0], int):
                choice = int(choice)
        else:
            choice = parameters[p]
        parameters[p] = choice

    parameters["seed"] = _draw_random_int32(random_state, size=1)[0]

    return parameters
