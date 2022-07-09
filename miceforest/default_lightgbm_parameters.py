# THESE VALUES WILL ALWAYS BE USED WHEN VALUES ARE NOT PASSED BY USER.
# seed is always set by the calling processes _random_state.
# These need to be main parameter names, not aliases
default_parameters = {
    "boosting": "random_forest",
    "num_iterations": 48,
    "max_depth": 8,
    "num_leaves": 128,
    "min_data_in_leaf": 1,
    "min_sum_hessian_in_leaf": 0.00001,
    "min_gain_to_split": 0.0,
    "bagging_fraction": 0.632,
    "feature_fraction": 1.0,
    "feature_fraction_bynode": 0.632,
    "bagging_freq": 1,
    "verbosity": -1,
}

# WHEN TUNING, THESE PARAMETERS OVERWRITE THE DEFAULTS ABOVE
# These need to be main parameter names, not aliases
def make_default_tuning_space(min_samples, max_samples):
    space = {
        "boosting": "gbdt",
        "learning_rate": 0.02,
        "num_iterations": 250,
        "min_data_in_leaf": (min_samples, max_samples),
        "min_sum_hessian_in_leaf": 0.1,
        "num_leaves": (2, 25),
        "bagging_fraction": (0.1, 1.0),
        "feature_fraction_bynode": (0.1, 1.0),
        "cat_smooth": (0, 25),
    }
    return space
