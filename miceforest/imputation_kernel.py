from miceforest.default_lightgbm_parameters import (
    default_parameters,
    make_default_tuning_space,
)
from miceforest.logger import Logger
from miceforest.imputed_data import ImputedData
from miceforest.utils import (
    logodds,
    _expand_value_to_dict,
    _list_union,
    _draw_random_int32,
    ensure_rng,
    stratified_categorical_folds,
    stratified_continuous_folds,
    _to_2d,
    _to_1d,
)
import numpy as np
from warnings import warn
from lightgbm import train, Dataset, cv, log_evaluation, early_stopping, Booster
from lightgbm.basic import _ConfigAliases
from io import BytesIO
from scipy.spatial import KDTree
from copy import copy
from typing import Union, List, Dict, Any, Optional, Tuple
from pandas import Series, DataFrame, MultiIndex, read_parquet, Categorical
from pandas.api.types import is_integer_dtype


_DEFAULT_DATA_SUBSET = 0
_DEFAULT_MEANMATCH_CANDIDATES = 5
_DEFAULT_MEANMATCH_STRATEGY = "normal"
_MICE_TIMED_LEVELS = ["Dataset", "Iteration", "Variable", "Event"]
_IMPUTE_NEW_DATA_TIMED_LEVELS = ["Dataset", "Iteration", "Variable", "Event"]
_PRE_LINK_DATATYPE = "float16"

# These can inherently be 2D, Series cannot.
_MEAN_MATCH_PRED_TYPE = Union[np.ndarray, DataFrame]


class ImputationKernel(ImputedData):
    """
    Creates a kernel dataset. This dataset can perform MICE on itself,
    and impute new data from models obtained during MICE.

    Parameters
    ----------
    data : pandas DataFrame.

        .. code-block:: text

            The data to be imputed.

    variable_schema : None or list or dict, default=None

        .. code-block:: text

            Specifies the feature - target relationships used to train models.
            This parameter also controls which models are built. Models can be built
            even if a variable contains no missing values, or is not being imputed.

                - If None, all columns with missing values will have models trained, and all
                    columns will be used as features in these models.
                - If list, all columns in data are used to impute the variables in the list
                - If dict the values will be used to impute the keys.

            No models will be trained for variables not specified by variable_schema
            (either by None, a list, or in dict keys).

    imputation_order: str, list[str], list[int], default="ascending"

        .. code-block:: text

            The order the imputations should occur in.
                - ascending: variables are imputed from least to most missing
                - descending: most to least missing
                - roman: from left to right in the dataset
                - arabic: from right to left in the dataset.

    data_subset: None or int or dict.

        .. code-block:: text

            Subsets the data used in each iteration, which can save a significant amount of time.
            This can also help with memory consumption, as the candidate data must be copied to
            make a feature dataset for lightgbm.

            The number of rows used for each variable is (# rows in raw data) - (# missing variable values)
            for each variable. data_subset takes a random sample of this.

            If int must be data_subset >= 0. Interpreted as the number of candidates.
            If 0, no subsetting is done.
            If dict, keys must be variable names, and values must follow two above rules.

            It is recommended to carefully select this value for each variable if dealing
            with very large data that barely fits into memory.

    mean_match_strategy: str or Dict[str, str]

        .. code-block:: text

            There are 3 mean matching strategies included in miceforest:
                - "normal" - this is the default. For all predictions, K-nearest-neighbors
                    is performed on the candidate predictions and bachelor predictions.
                    The top MMC closest candidate values are chosen at random.
                - "fast" - Only available for categorical and binary columns. A value
                    is selected at random weighted by the class probabilities.
                - "shap" - Similar to "normal" but more robust. A K-nearest-neighbors
                    search is performed on the shap values of the candidate predictions
                    and the bachelor predictions. A value from the top MMC closest candidate
                    values is chosen at random.

            A dict of strategies by variable can be passed as well. Any unmentioned variables
            will be set to the default, "normal".

            Special rules are enacted when mean_match_candidates == 0 for a variable. See the
            mean_match_candidates parameter for more information.

    mean_match_candidates: int or Dict[str, int]

        .. code-block:: text

            When mean matching relies on selecting one of the top N closest candidate predictions,
            this number is used for N.

            Special rules apply when this value is set to 0. This will skip mean matching entirely.
            The algorithm that applies depends on the objective type:
                - Regression: The bachelor predictions are used as the imputation values.
                - Binary: The class with the higher probability is chosen.
                - Multiclass: The class with the highest probability is chosen.

            Setting mmc to 0 will result in much faster process times, but at the cost of random
            variability that is desired when performing Multiple Imputation by Chained Equations.

    initialize_empty: bool, default = False

        .. code-block:: text

            If True, missing data is not filled in randomly before model training starts.

    save_all_iterations_data: boolean, optional(default=True)

        .. code-block:: text

            Setting to False will cause the process to not store the models and
            candidate values obtained at each iteration. This can save significant
            amounts of memory, but it means `impute_new_data()` will not be callable.

    copy_data: boolean (default = False)

        .. code-block:: text

            Should the dataset be referenced directly? If False, this will cause
            the dataset to be altered in place. If a copy is created, it is saved
            in self.working_data. There are different ways in which the dataset
            can be altered.

    random_state: None,int, or numpy.random.RandomState

        .. code-block:: text

            The random_state ensures script reproducibility. It only ensures reproducible
            results if the same script is called multiple times. It does not guarantee
            reproducible results at the record level, if a record is imputed multiple
            different times. If reproducible record-results are desired, a seed must be
            passed for each record in the random_seed_array parameter.

    """

    def __init__(
        self,
        data: DataFrame,
        num_datasets: int = 1,
        variable_schema: Union[List[str], Dict[str, str]] = None,
        imputation_order: str = "ascending",
        mean_match_candidates: Union[
            int, Dict[str, int]
        ] = _DEFAULT_MEANMATCH_CANDIDATES,
        mean_match_strategy: Optional[
            Union[str, Dict[str, str]]
        ] = _DEFAULT_MEANMATCH_STRATEGY,
        data_subset: Union[int, Dict[str, int]] = _DEFAULT_DATA_SUBSET,
        initialize_empty: bool = False,
        save_all_iterations_data: bool = True,
        copy_data: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):

        super().__init__(
            impute_data=data,
            num_datasets=num_datasets,
            variable_schema=variable_schema,
            save_all_iterations_data=save_all_iterations_data,
            copy_data=copy_data,
            random_seed_array=None,
        )

        # Model Training / Imputation Order:
        # Variables with missing data are always trained
        # first, according to imputation_order. Afterwards,
        # variables with no missing values have models trained.
        if imputation_order in ["ascending", "descending"]:
            _na_counts = {
                key: value
                for key, value in self.na_counts.items()
                if key in self.imputed_variables
            }
            self.imputation_order = list(
                Series(_na_counts).sort_values(ascending=False).index
            )
            if imputation_order == "decending":
                self.imputation_order.reverse()
        elif imputation_order == "roman":
            self.imputation_order = self.imputed_variables.copy()
        elif imputation_order == "arabic":
            self.imputation_order = self.imputed_variables.copy()
            self.imputation_order.reverse()
        else:
            raise ValueError("imputation_order not recognized.")

        modeled_but_not_imputed_variables = [
            col for col in self.modeled_variables if col not in self.imputed_variables
        ]
        model_training_order = self.imputation_order + modeled_but_not_imputed_variables
        self.model_training_order = model_training_order

        self.initialize_empty = initialize_empty
        self.save_all_iterations_data = save_all_iterations_data

        # Models are stored in a dict, keys are (variable, iteration, dataset)
        self.models: Dict[Tuple[str, int, int], Booster] = {}

        # Candidate preds are stored the same as models.
        self.candidate_preds: Dict[Tuple[str, int, int], Series] = {}

        # Optimal parameters can only be found on 1 dataset at the current iteration.
        self.optimal_parameters: Dict[str, Dict[str, Any]] = {}

        # Determine available candidates and interpret data subset.
        available_candidates = {
            v: (self.shape[0] - self.na_counts[v]) for v in self.model_training_order
        }
        data_subset = _expand_value_to_dict(
            _DEFAULT_DATA_SUBSET, data_subset, keys=self.model_training_order
        )
        for col in self.model_training_order:
            assert (
                data_subset[col] <= available_candidates[col]
            ), f"data_subset is more than available candidates for {col}"
        self.available_candidates = available_candidates
        self.data_subset = data_subset

        # Collect category information.
        categorical_columns: List[str] = [
            var
            for var, dtype in self.working_data.dtypes.items()
            if dtype.name == "category"
        ]
        category_counts = {
            col: len(self.working_data[col].cat.categories)
            for col in categorical_columns
        }
        numeric_columns = [
            col for col in self.working_data.columns if col not in categorical_columns
        ]
        binary_columns = []
        for col, count in category_counts.items():
            if count == 2:
                binary_columns.append(col)
                categorical_columns.remove(col)

        # Probably a better way of doing this
        assert set(categorical_columns).isdisjoint(set(numeric_columns))
        assert set(categorical_columns).isdisjoint(set(binary_columns))
        assert set(binary_columns).isdisjoint(set(numeric_columns))

        self.category_counts = category_counts
        self.modeled_categorical_columns = _list_union(
            categorical_columns, self.model_training_order
        )
        self.modeled_numeric_columns = _list_union(
            numeric_columns, self.model_training_order
        )
        self.modeled_binary_columns = _list_union(
            binary_columns, self.model_training_order
        )

        # Make sure all pandas categorical levels are used.
        rare_level_cols = []
        for col in self.modeled_categorical_columns:
            value_counts = data[col].value_counts(normalize=True)
            if np.any(value_counts < 0.002):
                rare_level_cols.append(col)
        if rare_level_cols:
            warn(
                f"{','.join(rare_level_cols)} have very rare categories, it is a good "
                "idea to group these, or set the min_data_in_leaf parameter to prevent "
                "lightgbm from outputting 0.0 probabilities."
            )

        self.mean_match_candidates = _expand_value_to_dict(
            _DEFAULT_MEANMATCH_CANDIDATES,
            mean_match_candidates,
            self.model_training_order,
        )
        self.mean_match_strategy = _expand_value_to_dict(
            _DEFAULT_MEANMATCH_STRATEGY, mean_match_strategy, self.model_training_order
        )

        for col in self.model_training_order:
            mmc = self.mean_match_candidates[col]
            mms = self.mean_match_strategy[col]
            assert not ((mmc == 0) and (mms == "shap")), (
                f"Failing because {col} mean_match_candidates == 0 and "
                "mean_match_strategy == shap. This implies an unintentional setup."
            )

        # Determine if the mean matching scheme will
        # require candidate information for each variable
        self.mean_matching_requires_candidates = []
        for variable in self.model_training_order:
            mean_match_strategy = self.mean_match_strategy[variable]
            if (mean_match_strategy in ["normal", "shap"]) or (
                variable in self.modeled_numeric_columns
            ):
                self.mean_matching_requires_candidates.append(variable)

        self.loggers = []

        # Manage randomness
        self._completely_random_kernel = random_state is None
        self._random_state = ensure_rng(random_state)

        # Set initial imputations (iteration 0).
        self._initialize_dataset(
            self, random_state=self._random_state
        )

    def __getstate__(self):
        """
        For pickling
        """
        # Copy the entire object, minus the big stuff

        special_handling = ["imputation_values"]
        if self.save_all_iterations_data:
            special_handling.append("candidate_preds")

        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in special_handling
        }.copy()

        state["imputation_values"] = {}
        state["candidate_preds"] = {}

        for col, df in self.imputation_values.items():
            byte_stream = BytesIO()
            df.to_parquet(byte_stream)
            state["imputation_values"][col] = byte_stream
        for col, df in self.candidate_preds.items():
            byte_stream = BytesIO()
            df.to_parquet(byte_stream)
            state["candidate_preds"][col] = byte_stream

        return state

    def __setstate__(self, state):
        """
        For unpickling
        """
        self.__dict__ = state

        for col, bytes in self.imputation_values.items():
            self.imputation_values[col] = read_parquet(bytes)

        if self.save_all_iterations_data:
            for col, bytes in self.candidate_preds.items():
                self.candidate_preds[col] = read_parquet(bytes)

    def __repr__(self):
        summary_string = f'\n{" " * 14}Class: ImputationKernel\n{self.__ids_info()}'
        return summary_string

    def _initialize_dataset(self, imputed_data, random_state):
        """
        Sets initial imputation values for iteration 0.
        If "random", draw values from the working data at random.
        If "empty", keep the values missing, since missing values
        can be handled natively by lightgbm.
        """

        assert not imputed_data.initialized, "dataset has already been initialized"

        if self.initialize_empty:
            # The default value when initialized is np.nan, nothing to do here
            pass
        else:
            for variable in imputed_data.imputed_variables:
                # Pulls from the kernel working data
                candidate_values = self._get_nonmissing_values(variable)
                candidate_num = candidate_values.shape[0]

                # Pulls from the ImputedData
                missing_ind = imputed_data.na_where[variable]
                missing_num = imputed_data.na_counts[variable]

                for dataset in range(imputed_data.num_datasets):
                    # Initialize using the random_state if no record seeds were passed.
                    if imputed_data.random_seed_array is None:
                        imputation_values = candidate_values.sample(
                            n=missing_num, replace=True, random_state=random_state
                        )
                        imputation_values.index = missing_ind
                        imputed_data[variable, 0, dataset] = imputation_values
                    else:
                        assert (
                            len(imputed_data.random_seed_array) == imputed_data.shape[0]
                        ), "The random_seed_array did not match the number of rows being imputed."
                        hashed_seeds = imputed_data._get_hashed_seeds(variable=variable)
                        selection_ind = hashed_seeds % candidate_num
                        imputation_values = candidate_values.iloc[selection_ind]
                        imputation_values.index = missing_ind
                        imputed_data[variable, 0, dataset] = imputation_values

        imputed_data.initialized = True

    def _get_lgb_params(self, variable, variable_parameters, random_state, **kwlgb):
        """
        Builds the parameters for a lightgbm model. Infers objective based on
        datatype of the response variable, assigns a random seed, finds
        aliases in the user supplied parameters, and returns a final dict.

        Parameters
        ----------
        var: int
            The variable to be modeled

        vsp: dict
            Variable specific parameters. These are supplied by the user.

        random_state: np.random.RandomState
            The random state to use (used to set the seed).

        kwlgb: dict
            Any additional parameters that should take presidence
            over the defaults or user supplied.
        """

        seed = _draw_random_int32(random_state, size=1)[0]

        if variable in self.modeled_categorical_columns:
            n_c = self.category_counts[variable]
            obj = {"objective": "multiclass", "num_class": n_c}
        elif variable in self.modeled_binary_columns:
            obj = {"objective": "binary"}
        else:
            obj = {"objective": "regression"}

        lgb_params = default_parameters.copy()
        lgb_params.update(obj)
        lgb_params["seed"] = seed

        # Priority is [variable specific] > [global in kwargs] > [defaults]
        lgb_params.update(kwlgb)
        lgb_params.update(variable_parameters)

        return lgb_params

    def _get_random_sample(self, parameters, random_state):
        """
        Searches through a parameter set and selects a random
        number between the values in any provided tuple of length 2.
        """

        parameters = parameters.copy()
        for p, v in parameters.items():
            if hasattr(v, "__iter__"):
                if isinstance(v, list):
                    parameters[p] = random_state.choice(v)
                elif isinstance(v, tuple):
                    parameters[p] = random_state.uniform(v[0], v[1], size=1)[0]
            else:
                pass
        parameters = self._make_params_digestible(parameters)
        return parameters

    def _make_params_digestible(self, params):
        """
        Cursory checks to force parameters to be digestible
        """

        int_params = [
            "num_leaves",
            "min_data_in_leaf",
            "num_threads",
            "max_depth",
            "num_iterations",
            "bagging_freq",
            "max_drop",
            "min_data_per_group",
            "max_cat_to_onehot",
        ]
        params = {
            key: int(val) if key in int_params else val for key, val in params.items()
        }
        return params

    def _get_oof_performance(
        self, parameters, folds, train_pointer, categorical_feature
    ):
        """
        Performance is gathered from built-in lightgbm.cv out of fold metric.
        Optimal number of iterations is also obtained.
        """

        num_iterations = parameters.pop("num_iterations")
        lgbcv = cv(
            params=parameters,
            train_set=train_pointer,
            folds=folds,
            num_boost_round=num_iterations,
            categorical_feature=categorical_feature,
            return_cvbooster=True,
            callbacks=[
                early_stopping(stopping_rounds=10, verbose=False),
                log_evaluation(period=0),
            ],
        )
        best_iteration = lgbcv["cvbooster"].best_iteration
        loss_metric_key = list(lgbcv)[0]
        loss = np.min(lgbcv[loss_metric_key])

        return loss, best_iteration

    def _get_nonmissing_subset_index(self, variable: str, seed: int):
        """
        Get random indices for a subset of the data in which variable is not missing.
        Used to create feature / label for training.

        replace = False because it would NOT mimic bagging for random forests.
        """

        data_subset = self.data_subset[variable]
        available_candidates = self.available_candidates[variable]
        nonmissing_ind = self._get_nonmissing_index(variable=variable)
        if (data_subset == 0) or (data_subset >= available_candidates):
            subset_index = nonmissing_ind
        else:
            rs = np.random.RandomState(seed)
            subset_index = rs.choice(nonmissing_ind, size=data_subset, replace=False)
        return subset_index

    def _make_label(self, variable: str, seed: int):
        """
        Returns a reproducible subset of the non-missing values of a variable.
        """
        # Don't subset at all if data_subset == 0 or we want more than there are candidates

        subset_index = self._get_nonmissing_subset_index(variable=variable, seed=seed)
        label = self.working_data.loc[subset_index, variable].copy()
        return label

    def _make_features_label(self, variable: str, seed: int):
        """
        Makes a reproducible set of features and
        target needed to train a lightgbm model.
        """
        subset_index = self._get_nonmissing_subset_index(variable=variable, seed=seed)
        predictor_columns = self.variable_schema[variable]
        features = self.working_data.loc[
            subset_index, predictor_columns + [variable]
        ].copy()
        label = features.pop(variable)
        return features, label

    def compile_candidate_preds(self):
        """
        Candidate predictions can be pre-generated before imputing new data.
        This can save a substantial amount of time, especially if save_models == 1.
        """

        compile_objectives = (
            self.mean_match_scheme.get_objectives_requiring_candidate_preds()
        )

        for key, model in self.models.items():
            already_compiled = key in self.candidate_preds.keys()
            objective = model.params["objective"]
            if objective in compile_objectives and not already_compiled:
                var = key[1]
                candidate_features, _, _ = self._make_features_label(
                    variable=var,
                    subset_count=self.data_subset[var],
                    random_seed=model.params["seed"],
                )
                self.candidate_preds[key] = self.mean_match_scheme.model_predict(
                    model, candidate_features
                )

            else:
                continue

    def delete_candidate_preds(self):
        """
        Deletes the pre-computed candidate predictions.
        """

        self.candidate_preds = {}

    def fit(self, X, y, **fit_params):
        """
        Method for fitting a kernel when used in a sklearn pipeline.
        Should not be called by the user directly.
        """
        assert self.num_datasets == 1, (
            "miceforest kernel should be initialized with datasets=1 if "
            "being used in a sklearn pipeline."
        )
        assert X.equals(self.working_data), (
            "It looks like this kernel is being used in a sklearn pipeline. "
            "The data passed in fit() should be the same as the data that "
            "was originally passed to the kernel. If this kernel is not being "
            "used in an sklearn pipeline, please just use the mice() method."
        )
        self.mice(**fit_params)
        return self

    @staticmethod
    def _mean_match_nearest_neighbors(
        mean_match_candidates: int,
        bachelor_preds: _MEAN_MATCH_PRED_TYPE,
        candidate_preds: _MEAN_MATCH_PRED_TYPE,
        candidate_values: Series,
        random_state: np.random.RandomState,
        hashed_seeds: Optional[np.ndarray] = None,
    ):
        """
        Determines the values of candidates which will be used to impute the bachelors
        """

        assert mean_match_candidates > 0, "Do not use nearest_neighbors with 0 mmc."

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

        imp_values = candidate_values.iloc[index_choice]

        return imp_values

    @staticmethod
    def _mean_match_binary_fast(
        mean_match_candidates: int,
        bachelor_preds: _MEAN_MATCH_PRED_TYPE,
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

    @staticmethod
    def _mean_match_multiclass_fast(
        mean_match_candidates: int,
        bachelor_preds: _MEAN_MATCH_PRED_TYPE,
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

            if hashed_seeds is None:
                # Turn bachelor_preds into discrete cdf, and choose
                bachelor_preds = bachelor_preds.cumsum(axis=1)
                compare = random_state.uniform(0, 1, size=(num_bachelors, 1))
                imp_values = (bachelor_preds < compare).sum(1)

            else:
                dtype = hashed_seeds.dtype
                dtype_max = np.iinfo(dtype).max
                compare = np.abs(hashed_seeds / dtype_max)
                imp_values = (bachelor_preds < compare).sum(1)

        return imp_values

    def _mean_match_fast(
        self,
        variable: str,
        mean_match_candidates: int,
        bachelor_preds: np.ndarray,
        random_state: np.random.RandomState,
        hashed_seeds: Optional[np.ndarray],
    ):
        """
        Dispatcher and formatter for the fast mean matching functions
        """
        if variable in self.modeled_categorical_columns:
            imputation_values = self._mean_match_multiclass_fast(
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                random_state=random_state,
                hashed_seeds=hashed_seeds,
            )
        elif variable in self.modeled_binary_columns:
            imputation_values = self._mean_match_binary_fast(
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                random_state=random_state,
                hashed_seeds=hashed_seeds,
            )
        else:
            raise ValueError("Shouldnt be able to get here")

        _to_1d(imputation_values)
        dtype = self.working_data[variable].dtype
        imputation_values = Categorical.from_codes(codes=imputation_values, dtype=dtype)

        return imputation_values

    def _impute_with_predictions(
        self,
        variable: str,
        lgbmodel: Booster,
        bachelor_features: DataFrame,
    ):
        bachelor_preds = lgbmodel.predict(
            bachelor_features,
            pred_contrib=False,
            raw_score=False,
        )
        dtype = self.working_data[variable].dtype
        if variable in self.modeled_numeric_columns:
            if is_integer_dtype(dtype):
                bachelor_preds = bachelor_preds.round(0)
            return Series(bachelor_preds, dtype=dtype)
        else:
            if variable in self.modeled_binary_columns:
                selection_ind = (bachelor_preds > 0.5).astype("uint8")
            else:
                assert (
                    variable in self.modeled_categorical_columns
                ), f"{variable} is not in numeric, binary or categorical columns"
                selection_ind = np.argmax(bachelor_preds, axis=1)
            values = dtype.categories[selection_ind]
            return Series(values, dtype=dtype)

    def _get_candidate_preds_mice(
        self,
        variable: str,
        lgbmodel: Booster,
        candidate_features: DataFrame,
        dataset: int,
        iteration: int,
    ):
        """
        This function also records the candidate predictions
        """
        shap = self.mean_match_strategy[variable] == "shap"
        fast = self.mean_match_strategy[variable] == "fast"
        logistic = variable not in self.modeled_numeric_columns

        assert hasattr(
            lgbmodel, "train_set"
        ), "Model was passed that does not have training data."
        if shap:
            candidate_preds = lgbmodel.predict(
                candidate_features,
                pred_contrib=True,
            ).astype(_PRE_LINK_DATATYPE)
        else:
            candidate_preds = lgbmodel._Booster__inner_predict(0)
            if logistic and not (shap or fast):
                candidate_preds = logodds(candidate_preds).astype(_PRE_LINK_DATATYPE)

        candidate_preds = self._prepare_prediction_multiindex(
            variable=variable,
            preds=candidate_preds,
            shap=shap,
            dataset=dataset,
            iteration=iteration,
        )

        if self.save_all_iterations_data:
            self._record_candidate_preds(
                variable=variable,
                candidate_preds=candidate_preds,
            )

        return candidate_preds

    def _get_candidate_preds_from_store(
        self,
        variable: str,
        dataset: int,
        iteration: int,
    ) -> DataFrame:
        """
        Mean matching requires 2D array, so always return a dataframe
        """
        ret = self.candidate_preds[variable][iteration][[dataset]]
        assert isinstance(ret, DataFrame)
        return ret

    def _get_bachelor_preds(
        self,
        variable: str,
        lgbmodel: Booster,
        bachelor_features: DataFrame,
        dataset: int,
        iteration: int,
    ) -> np.ndarray:

        shap = self.mean_match_strategy[variable] == "shap"
        fast = self.mean_match_strategy[variable] == "fast"
        logistic = variable not in self.modeled_numeric_columns

        bachelor_preds = lgbmodel.predict(
            bachelor_features,
            pred_contrib=shap,
        )

        if shap:
            bachelor_preds = bachelor_preds.astype(_PRE_LINK_DATATYPE)

        # We want the logods if running k-nearest
        # neighbors on logistic-link predictions
        if logistic and not (shap or fast):
            bachelor_preds = logodds(bachelor_preds).astype(_PRE_LINK_DATATYPE)

        bachelor_preds = self._prepare_prediction_multiindex(
            variable=variable,
            preds=bachelor_preds,
            shap=shap,
            dataset=dataset,
            iteration=iteration,
        )

        return bachelor_preds

    def mean_match_mice(
        self,
        variable: str,
        lgbmodel: Booster,
        bachelor_features: DataFrame,
        candidate_features: DataFrame,
        candidate_values: Series,
        dataset: int,
        iteration: int,
    ):
        mean_match_candidates = self.mean_match_candidates[variable]
        using_candidate_data = variable in self.mean_matching_requires_candidates

        use_mean_matching = mean_match_candidates > 0
        if not use_mean_matching:
            imputation_values = self._impute_with_predictions(
                variable=variable,
                lgbmodel=lgbmodel,
                bachelor_features=bachelor_features,
            )
            return imputation_values

        # Get bachelor predictions
        bachelor_preds = self._get_bachelor_preds(
            variable=variable,
            lgbmodel=lgbmodel,
            bachelor_features=bachelor_features,
            dataset=dataset,
            iteration=iteration,
        )

        if using_candidate_data:

            candidate_preds = self._get_candidate_preds_mice(
                variable=variable,
                lgbmodel=lgbmodel,
                candidate_features=candidate_features,
                dataset=dataset,
                iteration=iteration,
            )

            # By now, a numeric variable will be post-link, and
            # categorical / binary variables will be pre-link.
            imputation_values = self._mean_match_nearest_neighbors(
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                candidate_preds=candidate_preds,
                candidate_values=candidate_values,
                random_state=self._random_state,
                hashed_seeds=None,
            )

        else:

            imputation_values = self._mean_match_fast(
                variable=variable,
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                random_state=self._random_state,
                hashed_seeds=None,
            )

        return imputation_values

    def mean_match_ind(
        self,
        variable: str,
        lgbmodel: Booster,
        bachelor_features: DataFrame,
        dataset: int,
        iteration: int,
        hashed_seeds: Optional[np.ndarray] = None,
    ):
        mean_match_candidates = self.mean_match_candidates[variable]
        using_candidate_data = variable in self.mean_matching_requires_candidates
        use_mean_matching = mean_match_candidates > 0

        if not use_mean_matching:
            imputation_values = self._impute_with_predictions(
                variable=variable,
                lgbmodel=lgbmodel,
                bachelor_features=bachelor_features,
            )
            return imputation_values

        # Get bachelor predictions
        bachelor_preds = self._get_bachelor_preds(
            variable=variable,
            lgbmodel=lgbmodel,
            bachelor_features=bachelor_features,
            dataset=dataset,
            iteration=iteration,
        )

        if using_candidate_data:

            print(f"Mean matching {variable} using nearest neighbor")

            candidate_preds = self._get_candidate_preds_from_store(
                variable=variable,
                dataset=dataset,
                iteration=iteration,
            )

            candidate_values = self._make_label(
                variable=variable, seed=lgbmodel.params["seed"]
            )

            # By now, a numeric variable will be post-link, and
            # categorical / binary variables will be pre-link.
            imputation_values = self._mean_match_nearest_neighbors(
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                candidate_preds=candidate_preds,
                candidate_values=candidate_values,
                random_state=self._random_state,
                hashed_seeds=hashed_seeds,
            )

        else:

            imputation_values = self._mean_match_fast(
                variable=variable,
                mean_match_candidates=mean_match_candidates,
                bachelor_preds=bachelor_preds,
                random_state=self._random_state,
                hashed_seeds=hashed_seeds,
            )

        return imputation_values

    def _record_candidate_preds(
        self,
        variable: str,
        candidate_preds: DataFrame,
    ):

        assign_col_index = candidate_preds.columns

        if variable not in self.candidate_preds.keys():
            inferred_iteration = assign_col_index.get_level_values("iteration").unique()
            assert (
                len(inferred_iteration) == 1
            ), f"Malformed iteration multiindex for {variable}: {print(assign_col_index)}"
            inferred_iteration = inferred_iteration[0]
            assert (
                inferred_iteration == 1
            ), "Adding initial candidate preds after iteration 1."
            self.candidate_preds[variable] = candidate_preds
        else:
            self.candidate_preds[variable][assign_col_index] = candidate_preds

    def _prepare_prediction_multiindex(
        self,
        variable: str,
        preds: np.ndarray,
        shap: bool,
        dataset: int,
        iteration: int,
    ) -> DataFrame:

        multiclass = variable in self.modeled_categorical_columns
        cols = self.variable_schema[variable] + ["Intercept"]

        if shap:

            if multiclass:

                categories = self.working_data[variable].dtype.categories
                cat_count = self.category_counts[variable]
                preds = DataFrame(preds, columns=cols * cat_count)
                del preds["Intercept"]
                cols.remove("Intercept")
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], categories, cols],
                    names=("iteration", "dataset", "categories", "predictor"),
                )
                preds.columns = assign_col_index

            else:
                preds = DataFrame(preds, columns=cols)
                del preds["Intercept"]
                cols.remove("Intercept")
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], cols],
                    names=("iteration", "dataset", "predictor"),
                )
                preds.columns = assign_col_index

        else:

            if multiclass:

                categories = self.working_data[variable].dtype.categories
                preds = DataFrame(preds, columns=categories)
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], categories],
                    names=("iteration", "dataset", "categories"),
                )
                preds.columns = assign_col_index

            else:

                preds = DataFrame(preds, columns=[variable])
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset]], names=("iteration", "dataset")
                )
                preds.columns = assign_col_index

        return preds

    def mice(
        self,
        iterations: int,
        verbose: bool = False,
        variable_parameters: Dict[str, Any] = {},
        **kwlgb,
    ):
        """
        Perform mice on a given dataset.

        Multiple Imputation by Chained Equations (MICE) is an
        iterative method which fills in (imputes) missing data
        points in a dataset by modeling each column using the
        other columns, and then inferring the missing data.

        For more information on MICE, and missing data in
        general, see Stef van Buuren's excellent online book:
        https://stefvanbuuren.name/fimd/ch-introduction.html

        For detailed usage information, see this project's
        README on the github repository:
        https://github.com/AnotherSamWilson/miceforest

        Parameters
        ----------
        iterations: int
            The number of iterations to run.

        verbose: bool
            Should information about the process be printed?

        variable_parameters: None or dict
            Model parameters can be specified by variable here. Keys should
            be variable names or indices, and values should be a dict of
            parameter which should apply to that variable only.

        compile_candidates: bool
            Candidate predictions can be stored as they are created while
            performing mice. This prevents kernel.compile_candidate_preds()
            from having to be called separately, and can save a significant
            amount of time if compiled candidate predictions are desired.

        kwlgb:
            Additional arguments to pass to lightgbm. Applied to all models.

        """

        current_iterations = self.iteration_count()
        start_iter = current_iterations + 1
        end_iter = current_iterations + iterations + 1
        logger = Logger(
            name=f"MICE Iterations {current_iterations + 1} - {current_iterations + iterations}",
            timed_levels=_MICE_TIMED_LEVELS,
            verbose=verbose,
        )

        if len(variable_parameters) > 0:
            assert isinstance(
                variable_parameters, dict
            ), "variable_parameters should be a dict."
            assert set(variable_parameters).issubset(self.model_training_order), (
                "Variables in variable_parameters will not have models trained. "
                "Check kernel.model_training_order"
            )

        for iteration in range(start_iter, end_iter, 1):
            # absolute_iteration = self.iteration_count(datasets=dataset)
            logger.log(str(iteration) + " ", end="")

            for dataset in range(self.num_datasets):
                logger.log("Dataset " + str(dataset))

                # Set self.working_data to the most current iteration.
                self.complete_data(dataset=dataset, inplace=True)

                for variable in self.model_training_order:
                    logger.log(" | " + variable, end="")

                    # Define the lightgbm parameters
                    lgbpars = self._get_lgb_params(
                        variable,
                        variable_parameters.get(variable, {}),
                        self._random_state,
                        **kwlgb,
                    )

                    time_key = dataset, iteration, variable, "Prepare XY"
                    logger.set_start_time(time_key)
                    (
                        candidate_features,
                        candidate_values,
                    ) = self._make_features_label(
                        variable=variable, seed=lgbpars["seed"]
                    )

                    # lightgbm requires integers for label. Categories won't work.
                    if candidate_values.dtype.name == "category":
                        label = candidate_values.cat.codes
                    else:
                        label = candidate_values

                    num_iterations = lgbpars.pop("num_iterations")
                    train_pointer = Dataset(
                        data=candidate_features,
                        label=label,
                    )
                    logger.record_time(time_key)

                    time_key = dataset, iteration, variable, "Training"
                    logger.set_start_time(time_key)
                    current_model = train(
                        params=lgbpars,
                        train_set=train_pointer,
                        num_boost_round=num_iterations,
                        keep_training_booster=True,
                    )
                    logger.record_time(time_key)

                    # Only perform mean matching and insertion
                    # if variable is being imputed.
                    if variable in self.imputation_order:
                        time_key = dataset, iteration, variable, "Mean Matching"
                        logger.set_start_time(time_key)
                        bachelor_features = self.get_bachelor_features(
                            variable=variable
                        )
                        imputation_values = self.mean_match_mice(
                            variable=variable,
                            lgbmodel=current_model,
                            bachelor_features=bachelor_features,
                            candidate_features=candidate_features,
                            candidate_values=candidate_values,
                            dataset=dataset,
                            iteration=iteration,
                        )
                        imputation_values.index = self.na_where[variable]
                        logger.record_time(time_key)

                        assert imputation_values.shape == (
                            self.na_counts[variable],
                        ), f"{variable} mean matching returned malformed array"

                        # Insert the imputation_values we obtained
                        self[variable, iteration, dataset] = imputation_values

                        if not self.save_all_iterations_data:
                            del self[variable, iteration - 1, dataset]

                    else:

                        # This is called to save the candidate predictions
                        _ = self._get_candidate_preds_mice(
                            variable=variable,
                            lgbmodel=current_model,
                            candidate_features=candidate_features,
                            dataset=dataset,
                            iteration=iteration,
                        )
                        del _

                    # Save the model, if we should be
                    if self.save_all_iterations_data:
                        self.models[variable, iteration, dataset] = (
                            current_model.free_dataset()
                        )

                    self.iteration_tab[variable, dataset] += 1

                logger.log("\n", end="")

        self._ampute_original_data()
        self.loggers.append(logger)

    def get_model(
        self,
        variable: str,
        dataset: int,
        iteration: Optional[int] = None,
    ):
        # Allow passing -1 to get the latest iteration's model
        if (iteration is None) or (iteration == -1):
            iteration = self.iteration_count(dataset=dataset, variable=variable)
        try:
            model = self.models[variable, iteration, dataset]
        except KeyError:
            raise ValueError("Model was not saved.")
        return model

    def transform(self, X, y=None):
        """
        Method for calling a kernel when used in a sklearn pipeline.
        Should not be called by the user directly.
        """

        new_dat = self.impute_new_data(X, datasets=[0])
        return new_dat.complete_data(dataset=0, inplace=False)

    def tune_parameters(
        self,
        dataset: int,
        variables: Optional[List[str]] = None,
        variable_parameters: Optional[Dict[str, Any]] = None,
        parameter_sampling_method: str = "random",
        nfold: int = 10,
        optimization_steps: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
        **kwbounds,
    ):
        """
        Perform hyperparameter tuning on models at the current iteration.
        This method is not meant to be robust, but to get a decent set of
        parameters to help with imputation.

        .. code-block:: text

            A few notes:
            - Underlying models will now be gradient boosted trees by default (or any
                other boosting type compatible with lightgbm.cv).
            - The parameters are tuned on the data that would currently be returned by
                complete_data(dataset). It is usually a good idea to run at least 1 iteration
                of mice with the default parameters to get a more accurate idea of the
                real optimal parameters, since Missing At Random (MAR) data imputations
                tend to converge over time.
            - num_iterations is treated as the maximum number of boosting rounds to run
                in lightgbm.cv. It is NEVER optimized. The num_iterations that is returned
                is the best_iteration returned by lightgbm.cv. num_iterations can be passed to
                limit the boosting rounds, but the returned value will always be obtained
                from best_iteration.
            - lightgbm parameters are chosen in the following order of priority:
                1) Anything specified in variable_parameters
                2) Parameters specified globally in **kwbounds
                3) Default tuning space (miceforest.default_lightgbm_parameters.make_default_tuning_space)
                4) Default parameters (miceforest.default_lightgbm_parameters.default_parameters)
            - See examples for a detailed run-through. See
                https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters
                for even more detailed examples.


        Parameters
        ----------

        dataset: int (required)

            .. code-block:: text

                The dataset to run parameter tuning on. Tuning parameters on 1 dataset usually results
                in acceptable parameters for all datasets. However, tuning results are still stored
                seperately for each dataset.

        variables: None or list

            .. code-block:: text

                - If None, default hyper-parameter spaces are selected based on kernel data, and
                all variables with missing values are tuned.
                - If list, must either be indexes or variable names corresponding to the variables
                that are to be tuned.

        variable_parameters: None or dict

            .. code-block:: text

                Defines the tuning space. Dict keys must be variable names or indices, and a subset
                of the variables parameter. Values must be a dict with lightgbm parameter names as
                keys, and values that abide by the following rules:
                    scalar: If a single value is passed, that parameter will be used to build the
                        model, and will not be tuned.
                    tuple: If a tuple is passed, it must have length = 2 and will be interpreted as
                        the bounds to search within for that parameter.
                    list: If a list is passed, values will be randomly selected from the list.
                        NOTE: This is only possible with method = 'random'.

                example: If you wish to tune the imputation model for the 4th variable with specific
                bounds and parameters, you could pass:
                    variable_parameters = {
                        4: {
                            'learning_rate: 0.01',
                            'min_sum_hessian_in_leaf: (0.1, 10),
                            'extra_trees': [True, False]
                        }
                    }
                All models for variable 4 will have a learning_rate = 0.01. The process will randomly
                search within the bounds (0.1, 10) for min_sum_hessian_in_leaf, and extra_trees will
                be randomly selected from the list. Also note, the variable name for the 4th column
                could also be passed instead of the integer 4. All other variables will be tuned with
                the default search space, unless **kwbounds are passed.

        parameter_sampling_method: str

            .. code-block:: text

                If 'random', parameters are randomly selected.
                Other methods will be added in future releases.

        nfold: int

            .. code-block:: text

                The number of folds to perform cross validation with. More folds takes longer, but
                Gives a more accurate distribution of the error metric.

        optimization_steps:

            .. code-block:: text

                How many steps to run the process for.

        random_state: int or np.random.RandomState or None (default=None)

            .. code-block:: text

                The random state of the process. Ensures reproduceability. If None, the random state
                of the kernel is used. Beware, this permanently alters the random state of the kernel
                and ensures non-reproduceable results, unless the entire process up to this point
                is re-run.

        kwbounds:

            .. code-block:: text

                Any additional arguments that you want to apply globally to every variable.
                For example, if you want to limit the number of iterations, you could pass
                num_iterations = x to this functions, and it would apply globally. Custom
                bounds can also be passed.


        Returns
        -------
        2 dicts: optimal_parameters, optimal_parameter_losses

        - optimal_parameters: dict
            A dict of the optimal parameters found for each variable.
            This can be passed directly to the variable_parameters parameter in mice()

            .. code-block:: text

                {variable: {parameter_name: parameter_value}}

        - optimal_parameter_losses: dict
            The average out of fold cv loss obtained directly from
            lightgbm.cv() associated with the optimal parameter set.

            .. code-block:: text

                {variable: loss}

        """

        if random_state is None:
            random_state = self._random_state
        else:
            random_state = ensure_rng(random_state)

        if variables is None:
            variables = self.imputation_order
        else:
            variables = self._get_var_ind_from_list(variables)

        self.complete_data(dataset, inplace=True)

        logger = Logger(
            name=f"tune: {optimization_steps}",
            verbose=verbose,
        )

        vsp = self._format_variable_parameters(variable_parameters)
        variable_parameter_space = {}

        for var in variables:
            default_tuning_space = make_default_tuning_space(
                self.category_counts[var] if var in self.categorical_variables else 1,
                int((self.shape[0] - len(self.na_where[var])) / 10),
            )

            variable_parameter_space[var] = self._get_lgb_params(
                var=var,
                vsp={**kwbounds, **vsp[var]},
                random_state=random_state,
                **default_tuning_space,
            )

        if parameter_sampling_method == "random":
            for var, parameter_space in variable_parameter_space.items():
                logger.log(self._get_var_name_from_scalar(var) + " | ", end="")

                (
                    candidate_features,
                    candidate_values,
                    feature_cat_index,
                ) = self._make_features_label(
                    variable=var,
                    subset_count=self.data_subset[var],
                    random_seed=_draw_random_int32(
                        random_state=self._random_state, size=1
                    )[0],
                )

                # lightgbm requires integers for label. Categories won't work.
                if candidate_values.dtype.name == "category":
                    candidate_values = candidate_values.cat.codes

                is_categorical = var in self.categorical_variables

                for step in range(optimization_steps):
                    logger.log(str(step), end="")

                    # Make multiple attempts to learn something.
                    non_learners = 0
                    learning_attempts = 10
                    while non_learners < learning_attempts:
                        # Pointer and folds need to be re-initialized after every run.
                        train_pointer = Dataset(
                            data=candidate_features,
                            label=candidate_values,
                            categorical_feature=feature_cat_index,
                            free_raw_data=False,
                        )
                        if is_categorical:
                            folds = stratified_categorical_folds(
                                candidate_values, nfold
                            )
                        else:
                            folds = stratified_continuous_folds(candidate_values, nfold)
                        sampling_point = self._get_random_sample(
                            parameters=parameter_space, random_state=random_state
                        )
                        try:
                            loss, best_iteration = self._get_oof_performance(
                                parameters=sampling_point.copy(),
                                folds=folds,
                                train_pointer=train_pointer,
                                categorical_feature=feature_cat_index,
                            )
                        except:
                            loss, best_iteration = np.Inf, 0

                        if best_iteration > 1:
                            break
                        else:
                            non_learners += 1

                    if loss < self.optimal_parameter_losses[dataset][var]:
                        del sampling_point["seed"]
                        sampling_point["num_iterations"] = best_iteration
                        self.optimal_parameters[dataset][var] = sampling_point
                        self.optimal_parameter_losses[dataset][var] = loss

                    logger.log(" - ", end="")

                logger.log("\n", end="")

            self._ampute_original_data()
            return (
                self.optimal_parameters[dataset],
                self.optimal_parameter_losses[dataset],
            )

    def impute_new_data(
        self,
        new_data: DataFrame,
        datasets: Optional[List[int]] = None,
        iterations: Optional[int] = None,
        save_all_iterations_data: bool = True,
        copy_data: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        random_seed_array: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> ImputedData:
        """
        Impute a new dataset

        Uses the models obtained while running MICE to impute new data,
        without fitting new models. Pulls mean matching candidates from
        the original data.

        save_models must be > 0. If save_models == 1, the last model
        obtained in mice is used for every iteration. If save_models > 1,
        the model obtained at each iteration is used to impute the new
        data for that iteration. If specified iterations is greater than
        the number of iterations run so far using mice, the last model
        is used for each additional iteration.

        Type checking is not done. It is up to the user to ensure that the
        kernel data matches the new data being imputed.

        Parameters
        ----------
        new_data: pandas DataFrame or numpy ndarray
            The new data to impute

        datasets: int or List[int] (default = None)
            The datasets from the kernel to use to impute the new data.
            If None, all datasets from the kernel are used.

        iterations: int
            The number of iterations to run.
            If None, the same number of iterations run so far in mice is used.

        save_all_iterations: bool
            Should the imputation values of all iterations be archived?
            If False, only the latest imputation values are saved.

        copy_data: boolean
            Should the dataset be referenced directly? This will cause the dataset to be altered
            in place. If a copy is created, it is saved in self.working_data. There are different
            ways in which the dataset can be altered:

            1) complete_data() will fill in missing values
            2) mice() references and manipulates self.working_data directly.

        random_state: int or np.random.RandomState or None (default=None)
            The random state of the process. Ensures reproducibility. If None, the random state
            of the kernel is used. Beware, this permanently alters the random state of the kernel
            and ensures non-reproduceable results, unless the entire process up to this point
            is re-run.

        random_seed_array: None or np.ndarray (int32)

            .. code-block:: text

                Record-level seeds.

                Ensures deterministic imputations at the record level. random_seed_array causes
                deterministic imputations for each record no matter what dataset each record is
                imputed with, assuming the same number of iterations and datasets are used.
                If random_seed_array os passed, random_state must also be passed.

                Record-level imputations are deterministic if the following conditions are met:
                    1) The associated seed is the same.
                    2) The same kernel is used.
                    3) The same number of iterations are run.
                    4) The same number of datasets are run.

                Notes:
                    a) This will slightly slow down the imputation process, because random
                    number generation in numpy can no longer be vectorized. If you don't have a
                    specific need for deterministic imputations at the record level, it is better to
                    keep this parameter as None.

                    b) Using this parameter may change the global numpy seed by calling np.random.seed().

                    c) Internally, these seeds are hashed each time they are used, in order
                    to obtain different results for each dataset / iteration.


        verbose: boolean
            Should information about the process be printed?

        Returns
        -------
        miceforest.ImputedData

        """

        assert self.save_all_iterations_data, (
            "Cannot recreate imputation procedure, data was not saved during MICE. "
            "To save this data, set save_all_iterations_data to True when making kernel."
        )

        datasets = list(range(self.num_datasets)) if datasets is None else datasets
        kernel_iterations = self.iteration_count()
        iterations = kernel_iterations if iterations is None else iterations
        logger = Logger(
            name=f"Impute New Data {0}-{iterations}",
            timed_levels=_IMPUTE_NEW_DATA_TIMED_LEVELS,
            verbose=verbose,
        )

        assert isinstance(new_data, DataFrame)
        assert self.working_data.columns.equals(
            new_data.columns
        ), "Different columns from original dataset."
        assert np.all(
            [
                self.working_data[col].dtype == new_data[col].dtype
                for col in self.column_names
            ]
        ), "Column types are not the same as the original data. Check categorical columns."

        imputed_data = ImputedData(
            impute_data=new_data,
            num_datasets=len(datasets),
            variable_schema=self.variable_schema.copy(),
            save_all_iterations_data=save_all_iterations_data,
            copy_data=copy_data,
            random_seed_array=random_seed_array,
        )
        new_imputation_order = [
            col
            for col in self.model_training_order
            if col in imputed_data.vars_with_any_missing
        ]

        ### Manage Randomness.
        if random_state is None:
            assert (
                random_seed_array is None
            ), "random_state is also required when using random_seed_array"
            random_state = self._random_state
        else:
            random_state = ensure_rng(random_state)

        self._initialize_dataset(
            imputed_data,
            random_state=random_state,
        )

        for iteration in range(1, iterations + 1):
            logger.log(str(iteration) + " ", end="")

            for dataset in datasets:
                logger.log("Dataset " + str(dataset))
                self.complete_data(dataset=dataset, inplace=True)
                ds_new = datasets.index(dataset)
                imputed_data.complete_data(dataset=ds_new, inplace=True)

                for variable in new_imputation_order:
                    logger.log(" | " + variable, end="")

                    # Select our model.
                    current_model = self.get_model(
                        variable=variable, dataset=dataset, iteration=iteration
                    )

                    time_key = dataset, iteration, variable, "Getting Bachelor Features"
                    logger.set_start_time(time_key)
                    bachelor_features = imputed_data.get_bachelor_features(variable)
                    hashed_seeds = imputed_data._get_hashed_seeds(variable)
                    logger.record_time(time_key)

                    time_key = dataset, iteration, variable, "Mean Matching"
                    logger.set_start_time(time_key)
                    na_where = imputed_data.na_where[variable]
                    imputation_values = self.mean_match_ind(
                        variable=variable,
                        lgbmodel=current_model,
                        bachelor_features=bachelor_features,
                        dataset=dataset,
                        iteration=iteration,
                        hashed_seeds=hashed_seeds,
                    )
                    # self.cycle_random_seed_array(variable)
                    imputation_values.index = na_where
                    logger.record_time(time_key)

                    assert imputation_values.shape == (
                        imputed_data.na_counts[variable],
                    ), f"{variable} mean matching returned malformed array"

                    # Insert the imputation_values we obtained
                    imputed_data[variable, iteration, dataset] = imputation_values

                    if not imputed_data.save_all_iterations_data:
                        del imputed_data[variable, iteration - 1, dataset]

                logger.log("\n", end="")

        imputed_data._ampute_original_data()
        self.loggers.append(logger)

        return imputed_data

    # def save_kernel(
    #     self, filepath, clevel=None, cname=None, n_threads=None, copy_while_saving=True
    # ):
    #     """
    #     Compresses and saves the kernel to a file.

    #     Parameters
    #     ----------
    #     filepath: str
    #         The file to save to.

    #     clevel: int
    #         The compression level, sent to clevel argument in blosc.compress()

    #     cname: str
    #         The compression algorithm used.
    #         Sent to cname argument in blosc.compress.
    #         If None is specified, the default is lz4hc.

    #     n_threads: int
    #         The number of threads to use for compression.
    #         By default, all threads are used.

    #     copy_while_saving: boolean
    #         Should the kernel be copied while saving? Copying is safer, but
    #         may take more memory.

    #     """

    #     clevel = 9 if clevel is None else clevel
    #     cname = "lz4hc" if cname is None else cname
    #     n_threads = blosc2.detect_number_of_cores() if n_threads is None else n_threads

    #     if copy_while_saving:
    #         kernel = copy(self)
    #     else:
    #         kernel = self

    #     # convert working data to parquet bytes object
    #     if kernel.original_data_class == "pd_DataFrame":
    #         working_data_bytes = BytesIO()
    #         kernel.working_data.to_parquet(working_data_bytes)
    #         kernel.working_data = working_data_bytes

    #     blosc2.set_nthreads(n_threads)

    #     with open(filepath, "wb") as f:
    #         dill.dump(
    #             blosc2.compress(
    #                 dill.dumps(kernel),
    #                 clevel=clevel,
    #                 typesize=8,
    #                 shuffle=blosc2.NOSHUFFLE,
    #                 cname=cname,
    #             ),
    #             f,
    #         )

    def get_feature_importance(self, dataset, iteration=None) -> np.ndarray:
        """
        Return a matrix of feature importance. The cells
        represent the normalized feature importance of the
        columns to impute the rows. This is calculated
        internally by lightgbm.Booster.feature_importance().

        Parameters
        ----------
        dataset: int
            The dataset to get the feature importance for.

        iteration: int
            The iteration to return the feature importance for.
            Right now, the model must be saved to return importance

        Returns
        -------
        np.ndarray of importance values. Rows are imputed variables, and
        columns are predictor variables.

        """

        if iteration is None:
            iteration = self.iteration_count(dataset=dataset)

        importance_matrix = np.full(
            shape=(len(self.imputation_order), len(self.predictor_vars)),
            fill_value=np.NaN,
        )

        for ivar in self.imputation_order:
            importance_dict = dict(
                zip(
                    self.variable_schema[ivar],
                    self.get_model(dataset, ivar, iteration).feature_importance(),
                )
            )
            for pvar in importance_dict:
                importance_matrix[
                    np.sort(self.imputation_order).tolist().index(ivar),
                    np.sort(self.predictor_vars).tolist().index(pvar),
                ] = importance_dict[pvar]

        return importance_matrix

    def plot_feature_importance(
        self,
        dataset,
        normalize: bool = True,
        iteration: Optional[int] = None,
        **kw_plot,
    ):
        """
        Plot the feature importance. See get_feature_importance()
        for more details.

        Parameters
        ----------
        dataset: int
            The dataset to plot the feature importance for.

        iteration: int
            The iteration to plot the feature importance of.

        normalize: book
            Should the values be normalize from 0-1?
            If False, values are raw from Booster.feature_importance()

        kw_plot
            Additional arguments sent to sns.heatmap()

        """

        # Move this to .compat at some point.
        try:
            from seaborn import heatmap
        except ImportError:
            raise ImportError("seaborn must be installed to plot importance")

        importance_matrix = self.get_feature_importance(
            dataset=dataset, iteration=iteration
        )
        if normalize:
            importance_matrix = (
                importance_matrix / np.nansum(importance_matrix, 1).reshape(-1, 1)
            ).round(2)

        imputed_var_names = [
            self._get_var_name_from_scalar(int(i))
            for i in np.sort(self.imputation_order)
        ]
        predictor_var_names = [
            self._get_var_name_from_scalar(int(i)) for i in np.sort(self.predictor_vars)
        ]

        params = {
            **{
                "cmap": "coolwarm",
                "annot": True,
                "fmt": ".2f",
                "xticklabels": predictor_var_names,
                "yticklabels": imputed_var_names,
                "annot_kws": {"size": 16},
            },
            **kw_plot,
        }

        print(heatmap(importance_matrix, **params))
