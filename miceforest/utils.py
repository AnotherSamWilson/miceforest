import numpy as np
from typing import List, Optional, Union, Dict, Any
from .compat import pd_DataFrame, pd_Series, pd_read_parquet
import blosc
import dill

MeanMatchType = Union[int, float, Dict[str, float], Dict[str, int]]
VarSchemType = Optional[Union[List[str], Dict[str, List[str]]]]
VarParamType = Dict[Union[str, int], Dict[str, Any]]
CatFeatType = Union[str, List[str], List[int], Dict[int, dict]]


def ampute_data(
    data,
    variables=None,
    perc=0.1,
    random_state=None,
):
    """
    Ampute Data

    Returns a copy of data with specified variables amputed.

    Parameters
    ----------
     data : Pandas DataFrame
        The data to ampute
     variables : None or list
        If None, are variables are amputed.
     perc : double
        The percentage of the data to ampute.
    random_state: None, int, or np.random.RandomState

    Returns
    -------
    pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()
    data_shape = amputed_data.shape
    amp_rows = int(perc * data_shape[0])
    random_state = ensure_rng(random_state)

    if len(data_shape) > 1:
        if variables is None:
            variables = [i for i in range(amputed_data.shape[1])]
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                variables = [data.columns.tolist().index(i) for i in variables]

        if isinstance(amputed_data, pd_DataFrame):
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data.iloc[na_ind, v] = np.NaN

        if isinstance(amputed_data, np.ndarray):
            amputed_data = amputed_data.astype("float64")
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data[na_ind, v] = np.NaN

    else:

        na_ind = random_state.choice(
            np.arange(data_shape[0]), replace=False, size=amp_rows
        )
        amputed_data[na_ind] = np.NaN

    return amputed_data


def stratified_continuous_folds(y, nfold):
    """
    Create primitive stratified folds for continuous data.
    Should be digestible by lightgbm.cv function.
    """
    if isinstance(y, pd_Series):
        y = y.values
    elements = len(y)
    assert elements >= nfold, "more splits then elements."
    sorted = np.argsort(y)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    for v in val:
        yield (np.setdiff1d(range(elements), v), v)


def stratified_categorical_folds(y, nfold):
    """
    Create primitive stratified folds for categorical data.
    Should be digestible by lightgbm.cv function.
    """
    if isinstance(y, pd_Series):
        y = y.values
    y = y.reshape(
        y.shape[0],
    ).copy()
    elements = len(y)
    uniq, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    assert elements >= nfold, "more splits then elements."
    if any(counts < nfold):
        print("Decreasing nfold to lowest categorical level count...")
        nfold = min(counts)
    sorted = np.argsort(inv)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    for v in val:
        yield (np.setdiff1d(range(elements), v), v)


# https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
# We don't really need to worry that much about diffusion
# since we take % n at the end, and n (mmc) is usually
# very small. This hash performs well enough in testing.
def hash_int32(x):
    """
    A hash function which generates random uniform (enough)
    int32 integers. Used in mean matching and initialization.
    """
    assert isinstance(x, np.ndarray)
    assert x.dtype == "int32", "x must be int32"
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


def ensure_rng(
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> np.random.RandomState:
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


def load_kernel(filepath, n_threads=None):
    n_threads = blosc.detect_number_of_cores() if n_threads is None else n_threads
    blosc.set_nthreads(n_threads)
    with open(filepath, "rb") as f:
        kernel = dill.loads(blosc.decompress(dill.load(f)))

    if kernel.original_data_class == "pd_DataFrame":
        kernel.working_data = pd_read_parquet(kernel.working_data)
        for col in kernel.working_data.columns:
            kernel.working_data[col] = kernel.working_data[col].astype(
                kernel.working_dtypes[col]
            )

    return kernel


def _get_missing_stats(data: np.ndarray):
    """
    This function is seperate because this data is needed
    at different times depending on the datatype passed
    """
    na_where = np.isnan(data)
    data_shape = data.shape
    na_counts = na_where.sum(0).tolist()
    na_where = {col: np.where(na_where[:, col])[0] for col in range(data_shape[1])}
    vars_with_any_missing = [int(col) for col, ind in na_where.items() if len(ind > 0)]

    return na_where, data_shape, na_counts, vars_with_any_missing


def _get_default_mmc(candidates=None) -> int:
    if candidates is None:
        return 5
    else:
        percent = 0.001
        minimum = 5
        maximum = 10
        mean_match_candidates = min(maximum, max(minimum, int(percent * candidates)))
        return mean_match_candidates


def _ensure_iterable(x):
    """
    If the object is iterable, return the object.
    Else, return the object in a length 1 list.
    """
    return x if hasattr(x, "__iter__") else [x]


def _ensure_np_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd_DataFrame) | isinstance(x, pd_Series):
        return x.values
    else:
        raise ValueError("Can't cast to numpy array")


def _get_default_mms(candidates) -> int:
    return int(candidates)


def _setequal(a, b):
    if not hasattr(a, "__iter__"):
        return a == b
    else:
        return set(a) == set(b)


def _is_int(x):
    return isinstance(x, int) | isinstance(x, np.int_)


def _slice(dat, row_slice=slice(None), col_slice=slice(None)):
    """
    Returns a view of the subset data if possible.
    """

    if isinstance(dat, pd_DataFrame):
        return dat.iloc[row_slice, col_slice]
    elif isinstance(dat, np.ndarray):
        return dat[row_slice, col_slice]
    else:
        raise ValueError("Unknown data class passed.")


def _assign_col_values_without_copy(dat, row_ind, col_ind, val):
    """
    Insert values into different data frame objects.
    """

    row_ind = _ensure_iterable(row_ind)

    if isinstance(dat, pd_DataFrame):
        dat.iloc[row_ind, col_ind] = val
    elif isinstance(dat, np.ndarray):
        dat[row_ind, col_ind] = val
    else:
        raise ValueError("Unknown data class passed.")


def _subset_data(dat, row_ind=None, col_ind=None, return_1d=False):
    """
    Can subset data along 2 axis.
    Explicitly returns a copy.
    """

    row_ind = range(dat.shape[0]) if row_ind is None else row_ind
    col_ind = range(dat.shape[1]) if col_ind is None else col_ind

    if isinstance(dat, pd_DataFrame):
        data_copy = dat.iloc[row_ind, col_ind]
        return data_copy.to_numpy().flatten() if return_1d else data_copy
    elif isinstance(dat, np.ndarray):
        row_ind = _ensure_iterable(row_ind)
        col_ind = _ensure_iterable(col_ind)
        data_copy = dat[np.ix_(row_ind, col_ind)]
        return data_copy.flatten() if return_1d else data_copy
    else:
        raise ValueError("Unknown data class passed.")


# Not all aliases are stored somewhere retrievable in lightgbm..
# So they are manually stored here.
# List can be found at:
# https://github.com/microsoft/LightGBM/blob/master/src/io/config_auto.cpp
param_mapping = {
    "config_file": "config",
    "task_type": "task",
    "objective_type": "objective",
    "app": "objective",
    "application": "objective",
    "boosting_type": "boosting",
    "boost": "boosting",
    "train": "data",
    "train_data": "data",
    "train_data_file": "data",
    "data_filename": "data",
    "test": "valid",
    "valid_data": "valid",
    "valid_data_file": "valid",
    "test_data": "valid",
    "test_data_file": "valid",
    "valid_filenames": "valid",
    "num_iteration": "num_iterations",
    "n_iter": "num_iterations",
    "num_tree": "num_iterations",
    "num_trees": "num_iterations",
    "num_round": "num_iterations",
    "num_rounds": "num_iterations",
    "num_boost_round": "num_iterations",
    "n_estimators": "num_iterations",
    "shrinkage_rate": "learning_rate",
    "eta": "learning_rate",
    "num_leaf": "num_leaves",
    "max_leaves": "num_leaves",
    "max_leaf": "num_leaves",
    "tree": "tree_learner",
    "tree_type": "tree_learner",
    "tree_learner_type": "tree_learner",
    "num_thread": "num_threads",
    "nthread": "num_threads",
    "nthreads": "num_threads",
    "n_jobs": "num_threads",
    "device": "device_type",
    "random_seed": "seed",
    "random_state": "seed",
    "hist_pool_size": "histogram_pool_size",
    "min_data_per_leaf": "min_data_in_leaf",
    "min_data": "min_data_in_leaf",
    "min_child_samples": "min_data_in_leaf",
    "min_sum_hessian_per_leaf": "min_sum_hessian_in_leaf",
    "min_sum_hessian": "min_sum_hessian_in_leaf",
    "min_hessian": "min_sum_hessian_in_leaf",
    "min_child_weight": "min_sum_hessian_in_leaf",
    "sub_row": "bagging_fraction",
    "subsample": "bagging_fraction",
    "bagging": "bagging_fraction",
    "pos_sub_row": "pos_bagging_fraction",
    "pos_subsample": "pos_bagging_fraction",
    "pos_bagging": "pos_bagging_fraction",
    "neg_sub_row": "neg_bagging_fraction",
    "neg_subsample": "neg_bagging_fraction",
    "neg_bagging": "neg_bagging_fraction",
    "subsample_freq": "bagging_freq",
    "bagging_fraction_seed": "bagging_seed",
    "sub_feature": "feature_fraction",
    "colsample_bytree": "feature_fraction",
    "sub_feature_bynode": "feature_fraction_bynode",
    "colsample_bynode": "feature_fraction_bynode",
    "extra_tree": "extra_trees",
    "early_stopping_rounds": "early_stopping_round",
    "early_stopping": "early_stopping_round",
    "n_iter_no_change": "early_stopping_round",
    "max_tree_output": "max_delta_step",
    "max_leaf_output": "max_delta_step",
    "reg_alpha": "lambda_l1",
    "reg_lambda": "lambda_l2",
    "lambda": "lambda_l2",
    "min_split_gain": "min_gain_to_split",
    "rate_drop": "drop_rate",
    "topk": "top_k",
    "mc": "monotone_constraints",
    "monotone_constraint": "monotone_constraints",
    "monotone_constraining_method": "monotone_constraints_method",
    "mc_method": "monotone_constraints_method",
    "monotone_splits_penalty": "monotone_penalty",
    "ms_penalty": "monotone_penalty",
    "mc_penalty": "monotone_penalty",
    "feature_contrib": "feature_contri",
    "fc": "feature_contri",
    "fp": "feature_contri",
    "feature_penalty": "feature_contri",
    "fs": "forcedsplits_filename",
    "forced_splits_filename": "forcedsplits_filename",
    "forced_splits_file": "forcedsplits_filename",
    "forced_splits": "forcedsplits_filename",
    "verbose": "verbosity",
    "model_input": "input_model",
    "model_in": "input_model",
    "model_output": "output_model",
    "model_out": "output_model",
    "save_period": "snapshot_freq",
    "linear_trees": "linear_tree",
    "subsample_for_bin": "bin_construct_sample_cnt",
    "data_seed": "data_random_seed",
    "is_sparse": "is_enable_sparse",
    "enable_sparse": "is_enable_sparse",
    "sparse": "is_enable_sparse",
    "is_enable_bundle": "enable_bundle",
    "bundle": "enable_bundle",
    "is_pre_partition": "pre_partition",
    "two_round_loading": "two_round",
    "use_two_round_loading": "two_round",
    "has_header": "header",
    "label": "label_column",
    "weight": "weight_column",
    "group": "group_column",
    "group_id": "group_column",
    "query_column": "group_column",
    "query": "group_column",
    "query_id": "group_column",
    "ignore_feature": "ignore_column",
    "blacklist": "ignore_column",
    "cat_feature": "categorical_feature",
    "categorical_column": "categorical_feature",
    "cat_column": "categorical_feature",
    "is_save_binary": "save_binary",
    "is_save_binary_file": "save_binary",
    "is_predict_raw_score": "predict_raw_score",
    "predict_rawscore": "predict_raw_score",
    "raw_score": "predict_raw_score",
    "is_predict_leaf_index": "predict_leaf_index",
    "leaf_index": "predict_leaf_index",
    "is_predict_contrib": "predict_contrib",
    "contrib": "predict_contrib",
    "predict_result": "output_result",
    "prediction_result": "output_result",
    "predict_name": "output_result",
    "prediction_name": "output_result",
    "pred_name": "output_result",
    "name_pred": "output_result",
    "convert_model_file": "convert_model",
    "num_classes": "num_class",
    "unbalance": "is_unbalance",
    "unbalanced_sets": "is_unbalance",
    "metrics": "metric",
    "metric_types": "metric",
    "output_freq": "metric_freq",
    "training_metric": "is_provide_training_metric",
    "is_training_metric": "is_provide_training_metric",
    "train_metric": "is_provide_training_metric",
    "ndcg_eval_at": "eval_at",
    "ndcg_at": "eval_at",
    "map_eval_at": "eval_at",
    "map_at": "eval_at",
    "num_machine": "num_machines",
    "local_port": "local_listen_port",
    "port": "local_listen_port",
    "machine_list_file": "machine_list_filename",
    "machine_list": "machine_list_filename",
    "mlist": "machine_list_filename",
    "workers": "machines",
    "nodes": "machines",
}
