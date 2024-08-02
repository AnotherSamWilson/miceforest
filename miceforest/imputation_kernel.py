from copy import copy
from io import BytesIO
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
from lightgbm import Booster, Dataset, cv, early_stopping, log_evaluation, train
from lightgbm.basic import _ConfigAliases
from pandas import Categorical, DataFrame, MultiIndex, Series, read_parquet
from pandas.api.types import is_integer_dtype
from scipy.spatial import KDTree

from .default_lightgbm_parameters import _DEFAULT_LGB_PARAMS, _sample_parameters
from .imputed_data import ImputedData
from .logger import Logger
from .utils import (
    _draw_random_int32,
    _expand_value_to_dict,
    _list_union,
    ensure_rng,
    logodds,
    stratified_categorical_folds,
    stratified_continuous_folds,
)

_DEFAULT_DATA_SUBSET = 0
_DEFAULT_MEANMATCH_CANDIDATES = 5
_DEFAULT_MEANMATCH_STRATEGY = "normal"
_MICE_TIMED_LEVELS = ["Dataset", "Iteration", "Variable", "Event"]
_IMPUTE_NEW_DATA_TIMED_LEVELS = ["Dataset", "Iteration", "Variable", "Event"]
_TUNING_TIMED_LEVELS = ["Variable", "Iteration"]
_PRE_LINK_DATATYPE = "float16"


class ImputationKernel(ImputedData):
    """
    Creates a kernel dataset. This dataset can perform MICE on itself,
    and impute new data from models obtained during MICE.

    Parameters
    ----------
    data : pandas.DataFrame.
        The data to be imputed.
    variable_schema : None or List[str] or Dict[str, str], default=None
        Specifies the feature - target relationships used to train models.
        This parameter also controls which models are built. Models can be built
        even if a variable contains no missing values, or is not being imputed.

            - If :code:`None`, all columns with missing values will have models trained, and all
              columns will be used as features in these models.
            - If :code:`List[str]`, all columns in data are used to impute the variables in the list
            - If :code:`Dict[str, str]` the values will be used to impute the keys.

        No models will be trained for variables not specified by variable_schema
        (either by None, a list, or in dict keys).
    imputation_order : str, default="ascending"
        The order the imputations should occur in:

        - :code:`ascending`: variables are imputed from least to most missing
        - :code:`descending`: most to least missing
        - :code:`roman`: from left to right in the dataset
        - :code:`arabic`: from right to left in the dataset.

    data_subset: None or int or Dict[str, int], default=0
        Subsets the data used to train the model for each variable, which can save a significant amount of time.
        The number of rows used for model training and mean matching (candidates) is
        :code:`(# rows in raw data) - (# missing variable values)`
        for each variable. :code:`data_subset` takes a random sample from these candidates.

        - If :code:`int`, must be >= 0. Interpreted as the number of candidates.
        - If :code:`0`, no subsetting is done.
        - If :code:`Dict[str, int]`, keys must be variable names, and values must follow two above rules.

        This can also help with memory consumption, as the candidate data must be copied to
        make a feature dataset for lightgbm. It is recommended to carefully select this value
        for each variable if dealing with very large data that barely fits into memory.

    mean_match_strategy: str or Dict[str, str], default="normal"
        There are 3 mean matching strategies included in miceforest:

        - :code:`normal` - this is the default. For all predictions, K-nearest-neighbors
          is performed on the candidate predictions and bachelor predictions.
          The top MMC closest candidate values are chosen at random.
        - :code:`fast` - Only available for categorical and binary columns. A value
          is selected at random weighted by the class probabilities.
        - :code:`shap` - Similar to "normal" but more robust. A K-nearest-neighbors
          search is performed on the shap values of the candidate predictions
          and the bachelor predictions. A value from the top MMC closest candidate
          values is chosen at random.

        A dict of strategies by variable can be passed as well. Any unmentioned variables
        will be set to the default, "normal".

            .. code-block:: python

                mean_match_strategy = {
                    'column_1': 'fast',
                    'column_2': 'shap',
                }

        Special rules are enacted when :code:`mean_match_candidates==0` for a
        variable. See the mean_match_candidates parameter for more information.

    mean_match_candidates: int or Dict[str, int]
        The number of nearest neighbors to choose an imputation value from randomly when mean matching.

        Special rules apply when this value is set to 0. This will skip mean matching entirely.
        The algorithm that applies depends on the objective type:

        - :code:`Regression`: The bachelor predictions are used as the imputation values.
        - :code:`Binary`: The class with the higher probability is chosen.
        - :code:`Multiclass`: The class with the highest probability is chosen.

        Setting mmc to 0 will result in much faster process times, but has a few downsides:

        - Imputation values for regression variables might no longer be valid values.
          Mean matching ensures that the imputed values have been realized in the data before.
        - Random variability from mean matching is often desired to get a more accurate
          view of the variability in imputed "confidence"

    initialize_empty: bool, default=False
        If :code:`True`, missing data is not filled in randomly before model training starts.

    save_all_iterations_data: bool, default=True
        Setting to False will cause the process to not store the models and
        candidate values obtained at each iteration. This can save significant
        amounts of memory, but it means :code:`impute_new_data()` will not be callable.

    copy_data: bool, default=True
        Should the dataset be referenced directly? If False, this will cause
        the dataset to be altered in place. If a copy is created, it is saved
        in self.working_data. There are different ways in which the dataset
        can be altered.

    random_state: None, int, or numpy.random.RandomState
        The random_state ensures script reproducibility. It only ensures reproducible
        results if the same script is called multiple times. It does not guarantee
        reproducible results at the record level if a record is imputed multiple
        different times. If reproducible record-results are desired, a seed must be
        passed for each record in the :code:`random_seed_array` parameter.
    """

    def __init__(
        self,
        data: DataFrame,
        num_datasets: int = 1,
        variable_schema: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        imputation_order: Literal[
            "ascending", "descending", "roman", "latin"
        ] = "ascending",
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

        datasets = list(range(num_datasets))

        super().__init__(
            impute_data=data,
            datasets=datasets,
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
        self.candidate_preds: Dict[str, DataFrame] = {}

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
        predictor_columns = sum(self.variable_schema.values(), [])
        self.predictor_columns = [
            col for col in data.columns if col in predictor_columns
        ]

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

        self.loggers: List[Logger] = []

        # Manage randomness
        self._completely_random_kernel = random_state is None
        self._random_state = ensure_rng(random_state)

        # Set initial imputations (iteration 0).
        self._initialize_dataset(self, random_state=self._random_state)

        # Save for use later
        self.optimal_parameter_losses: Dict[str, float] = dict()
        self.optimal_parameters = dict()

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
        summary_string = f'\n{" " * 14}Class: ImputationKernel\n{self._ids_info()}'
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

                for dataset in imputed_data.datasets:
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

    @staticmethod
    def _uncover_aliases(params):
        """
        Switches all aliases in the parameter dict to their
        True name, easiest way to avoid duplicate parameters.
        """
        alias_dict = _ConfigAliases._get_all_param_aliases()
        for param in list(params):
            for true_name, aliases in alias_dict.items():
                if param in aliases:
                    params[true_name] = params.pop(param)

    def _make_lgb_params(
        self,
        variable: str,
        default_parameters: dict,
        variable_parameters: dict,
        **kwlgb,
    ):
        """
        Builds the parameters for a lightgbm model. Infers objective based on
        datatype of the response variable, assigns a random seed, finds
        aliases in the user supplied parameters, and returns a final dict.

        Parameters
        ----------
        variable: int
            The variable to be modeled

        default_parameters: dict
            The base set of parameters that should be used.

        variable_parameters: dict
            Variable specific parameters. These are supplied by the user.

        kwlgb: dict
            Any additional parameters that should take presidence
            over the defaults.
        """

        seed = _draw_random_int32(self._random_state, size=1)[0]

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

        self._uncover_aliases(lgb_params)
        self._uncover_aliases(kwlgb)
        self._uncover_aliases(variable_parameters)

        # Priority is [variable specific] > [global in kwargs] > [defaults]
        lgb_params.update(kwlgb)
        lgb_params.update(variable_parameters)

        return lgb_params

    # WHEN TUNING, THESE PARAMETERS OVERWRITE THE DEFAULTS ABOVE
    # These need to be main parameter names, not aliases
    def _make_tuning_space(
        self,
        variable: str,
        variable_parameters: dict,
        use_gbdt: bool,
        min_samples: int,
        max_samples: int,
        **kwargs,
    ):

        # Start with the default parameters, update with the search space
        params = _DEFAULT_LGB_PARAMS.copy()
        search_space = {
            "min_data_in_leaf": (min_samples, max_samples),
            "max_depth": (2, 6),
            "num_leaves": (2, 25),
            "bagging_fraction": (0.1, 1.0),
            "feature_fraction_bynode": (0.1, 1.0),
        }
        params.update(search_space)

        # Set our defaults if using gbdt
        if use_gbdt:
            params["boosting"] = "gbdt"
            params["learning_rate"] = 0.02
            params["num_iterations"] = 250

        params = self._make_lgb_params(
            variable=variable,
            default_parameters=params,
            variable_parameters=variable_parameters,
            **kwargs,
        )

        return params

    @staticmethod
    def _get_oof_performance(
        parameters: dict,
        folds: Generator,
        train_set: Dataset,
    ):
        """
        Performance is gathered from built-in lightgbm.cv out of fold metric.
        Optimal number of iterations is also obtained.
        """

        num_iterations = parameters.pop("num_iterations")
        lgbcv = cv(
            params=parameters,
            train_set=train_set,
            folds=folds,
            num_boost_round=num_iterations,
            return_cvbooster=True,
            callbacks=[
                early_stopping(stopping_rounds=10, verbose=False),
                log_evaluation(period=0),
            ],
        )
        best_iteration = lgbcv["cvbooster"].best_iteration  # type: ignore
        loss_metric_key = list(lgbcv)[0]
        loss: float = np.min(lgbcv[loss_metric_key])  # type: ignore

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

    @staticmethod
    def _mean_match_nearest_neighbors(
        mean_match_candidates: int,
        bachelor_preds: DataFrame,
        candidate_preds: DataFrame,
        candidate_values: Series,
        random_state: np.random.RandomState,
        hashed_seeds: Optional[np.ndarray] = None,
    ) -> Series:
        """
        Determines the values of candidates which will be used to impute the bachelors
        """

        assert mean_match_candidates > 0, "Do not use nearest_neighbors with 0 mmc."
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
        bachelor_preds: DataFrame,
        random_state: np.random.RandomState,
        hashed_seeds: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Chooses 0/1 randomly weighted by probability obtained from prediction.
        If mean_match_candidates is 0, choose class with highest probability.

        Returns a np.ndarray, because these get set to categorical later on.
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
                    imp_values.append(np.random.binomial(n=1, p=bachelor_preds.iloc[i]))

                imp_values = np.array(imp_values)

        imp_values.shape = (-1,)

        return imp_values

    @staticmethod
    def _mean_match_multiclass_fast(
        mean_match_candidates: int,
        bachelor_preds: DataFrame,
        random_state: np.random.RandomState,
        hashed_seeds: Optional[np.ndarray],
    ):
        """
        If mean_match_candidates is 0, choose class with highest probability.
        Otherwise, randomly choose class weighted by class probabilities.

        Returns a np.ndarray, because these get set to categorical later on.
        """
        if mean_match_candidates == 0:
            imp_values = np.argmax(bachelor_preds, axis=1)

        else:
            num_bachelors = bachelor_preds.shape[0]
            bachelor_preds = bachelor_preds.cumsum(axis=1).to_numpy()

            if hashed_seeds is None:
                compare = random_state.uniform(0, 1, size=(num_bachelors, 1))
                imp_values = (bachelor_preds < compare).sum(1)

            else:
                dtype = hashed_seeds.dtype
                dtype_max = np.iinfo(dtype).max
                compare = np.abs(hashed_seeds / dtype_max)
                compare.shape = (-1, 1)
                imp_values = (bachelor_preds < compare).sum(1)

        imp_values.shape = (-1,)

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
        assert isinstance(bachelor_preds, np.ndarray)
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
    ) -> DataFrame:
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
            )
            candidate_preds = candidate_preds.astype(_PRE_LINK_DATATYPE)  # type: ignore
        else:
            candidate_preds = lgbmodel._Booster__inner_predict(0)  # type: ignore
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
    ) -> DataFrame:

        shap = self.mean_match_strategy[variable] == "shap"
        fast = self.mean_match_strategy[variable] == "fast"
        logistic = variable not in self.modeled_numeric_columns

        bachelor_preds = lgbmodel.predict(
            bachelor_features,
            pred_contrib=shap,
        )
        assert isinstance(bachelor_preds, np.ndarray)

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
            ), f"Malformed iteration multiindex for {variable}: {assign_col_index}"
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
                preds_df = DataFrame(preds, columns=cols * cat_count)
                del preds_df["Intercept"]
                cols.remove("Intercept")
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], categories, cols],
                    names=("iteration", "dataset", "categories", "predictor"),
                )
                preds_df.columns = assign_col_index

            else:
                preds_df = DataFrame(preds, columns=cols)
                del preds_df["Intercept"]
                cols.remove("Intercept")
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], cols],
                    names=("iteration", "dataset", "predictor"),
                )
                preds_df.columns = assign_col_index

        else:

            if multiclass:

                categories = self.working_data[variable].dtype.categories
                preds_df = DataFrame(preds, columns=categories)
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset], categories],
                    names=("iteration", "dataset", "categories"),
                )
                preds_df.columns = assign_col_index

            else:

                preds_df = DataFrame(preds, columns=[variable])
                assign_col_index = MultiIndex.from_product(
                    [[iteration], [dataset]], names=("iteration", "dataset")
                )
                preds_df.columns = assign_col_index

        return preds_df

    def _mean_match_mice(
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

    def _mean_match_ind(
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

    def mice(
        self,
        iterations: int,
        verbose: bool = False,
        variable_parameters: Dict[str, Any] = {},
        **kwlgb,
    ):
        """
        Perform MICE on a given dataset.

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

            .. code-block:: python

                variable_parameters = {
                    'column': {
                        'min_sum_hessian_in_leaf: 25.0,
                        'extra_trees': True,
                    }
                }

        kwlgb:
            Additional parameters to pass to lightgbm. Applied to all models.

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

            for dataset in self.datasets:
                logger.log("Dataset " + str(dataset))

                # Set self.working_data to the most current iteration.
                self.complete_data(dataset=dataset, inplace=True)

                for variable in self.model_training_order:
                    logger.log(" | " + variable, end="")

                    # Define the lightgbm parameters
                    lgbpars = self._make_lgb_params(
                        variable=variable,
                        default_parameters=_DEFAULT_LGB_PARAMS.copy(),
                        variable_parameters=variable_parameters.get(variable, dict()),
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
                        bachelor_features = self._get_bachelor_features(
                            variable=variable
                        )
                        imputation_values = self._mean_match_mice(
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
        iteration: int = -1,
    ):
        """
        Returns the model trained for the specified variable, dataset, iteration.
        Model must have been saved.

        Parameters
        ----------
        variable: str
            The variable

        dataset: int
            The dataset

        iteration: str
            The iteration. Use -1 for the latest.
        """
        # Allow passing -1 to get the latest iteration's model
        if iteration == -1:
            iteration = self.iteration_count(dataset=dataset, variable=variable)
        try:
            model = self.models[variable, iteration, dataset]
        except KeyError:
            raise ValueError("Model was not saved.")
        return model

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

    def transform(self, X, y=None):
        """
        Method for calling a kernel when used in a sklearn pipeline.
        Should not be called by the user directly.
        """

        new_dat = self.impute_new_data(X, datasets=[0])
        return new_dat.complete_data(dataset=0, inplace=False)

    def tune_parameters(
        self,
        dataset: int = 0,
        variables: Optional[List[str]] = None,
        variable_parameters: Dict[str, Any] = dict(),
        parameter_sampling_method: Literal["random"] = "random",
        max_reattempts: int = 5,
        use_gbdt: bool = True,
        nfold: int = 10,
        optimization_steps: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Perform hyperparameter tuning on models at the current iteration.
        This method is not meant to be robust, but to get a decent set of
        parameters to help with imputation. A few notes:

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
            - Anything specified in variable_parameters
            - Parameters specified globally in `**kwbounds`
            - Default tuning space (miceforest.default_lightgbm_parameters)
            - Default parameters (miceforest.default_lightgbm_parameters.default_parameters)
        - See examples for a detailed run-through. See
          https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters
          for even more detailed examples.

        Parameters
        ----------
        dataset: int (required)
            The dataset to run parameter tuning on. Tuning parameters on 1 dataset usually results
            in acceptable parameters for all datasets. However, tuning results are still stored
            seperately for each dataset.
        variables: None or List[str]
            - If None, default hyper-parameter spaces are selected based on kernel data, and
              all variables with missing values are tuned.
            - If list, must either be indexes or variable names corresponding to the variables
              that are to be tuned.

        variable_parameters: None or dict
            Defines the tuning space. Dict keys must be variable names or indices, and a subset
            of the variables parameter. Values must be a dict with lightgbm parameter names as
            keys, and values that abide by the following rules:

            - **scalar**: If a single value is passed, that parameter will be used to build the
              model, and will not be tuned.
            - **tuple**: If a tuple is passed, it must have length = 2 and will be interpreted as
              the bounds to search within for that parameter.
            - **list**: If a list is passed, values will be randomly selected from the list.

            example: If you wish to tune the imputation model for the 4th variable with specific
            bounds and parameters, you could pass:

            .. code-block:: python

                variable_parameters = {
                    'column': {
                        'learning_rate: 0.01',
                        'min_sum_hessian_in_leaf: (0.1, 10),
                        'extra_trees': [True, False]
                    }
                }

            All models for variable 'column' will have a learning_rate = 0.01. The process will randomly
            search within the bounds (0.1, 10) for min_sum_hessian_in_leaf, and extra_trees will
            be randomly selected from the list. Also note, the variable name for the 4th column
            could also be passed instead of the integer 4. All other variables will be tuned with
            the default search space, unless `**kwbounds` are passed.

        parameter_sampling_method: str
            If :code:`random`, parameters are randomly selected.
            Other methods will be added in future releases.

        max_reattempts: int
            The maximum number of failures (or non-learners) before the process stops, and moves to the
            next variable. Failures can be caused by bad parameters passed to lightgbm. Non-learners
            occur when trees cannot possibly be built (i.e. if :code:`min_data_in_leaf > dataset.shape[0]`).

        use_gbdt: bool
            Whether the models should use gradient boosting instead of random forests.
            If True, the optimal number of iterations will be found in lgb.cv, along
            with the other parameters.

        nfold: int
            The number of folds to perform cross validation with. More folds takes longer, but
            Gives a more accurate distribution of the error metric.

        optimization_steps: int
            How many steps to run the process for.

        random_state: int or np.random.RandomState or None (default=None)
            The random state of the process. Ensures reproduceability. If None, the random state
            of the kernel is used. Beware, this permanently alters the random state of the kernel
            and ensures non-reproduceable results, unless the entire process up to this point
            is re-run.

        verbose: bool
            Whether to print progress.

        kwbounds:
            Any additional arguments that you want to apply globally to every variable.
            For example, if you want to limit the number of iterations, you could pass
            num_iterations = x to this functions, and it would apply globally. Custom
            bounds can also be passed.


        Returns
        -------
        optimal_parameters: dict
            A dict of the optimal parameters found for each variable.
            This can be passed directly to the :code:`variable_parameters` parameter in :code:`mice()`

        """

        random_state = ensure_rng(random_state)

        if variables is None:
            variables = self.imputation_order

        self.complete_data(dataset, inplace=True)

        logger = Logger(
            name=f"tune: {optimization_steps}",
            timed_levels=_TUNING_TIMED_LEVELS,
            verbose=verbose,
        )

        for variable in variables:

            logger.log(f"Optimizing {variable}")

            seed = _draw_random_int32(random_state=random_state, size=1)

            (
                candidate_features,
                candidate_values,
            ) = self._make_features_label(variable=variable, seed=seed)

            min_samples = (
                self.category_counts[variable]
                if variable in self.modeled_categorical_columns
                else 1
            )
            max_samples = int(candidate_features.shape[0] / 5)

            assert isinstance(
                variable_parameters, dict
            ), "variable_parameters should be a dict"
            vp = variable_parameters.get(variable, dict()).copy()

            tuning_space = self._make_tuning_space(
                variable=variable,
                variable_parameters=vp,
                use_gbdt=use_gbdt,
                min_samples=min_samples,
                max_samples=max_samples,
                **kwargs,
            )

            # lightgbm requires integers for label. Categories won't work.
            if candidate_values.dtype.name == "category":
                cat_cols = (
                    self.modeled_categorical_columns + self.modeled_binary_columns
                )
                assert variable in cat_cols, (
                    "Something went wrong in definining categorical "
                    f"status of variable {variable}. Please open an issue."
                )
                candidate_values = candidate_values.cat.codes
                is_cat = True
            else:
                is_cat = False

            for step in range(optimization_steps):

                # Make multiple attempts to learn something.
                non_learners = 0
                while non_learners < max_reattempts:

                    # Sample parameters
                    sampled_parameters = _sample_parameters(
                        parameters=tuning_space,
                        random_state=random_state,
                        parameter_sampling_method=parameter_sampling_method,
                    )
                    logger.log(
                        f"   Step {step} - Parameters: {sampled_parameters}", end=""
                    )

                    # Pointer and folds need to be re-initialized after every run.
                    train_set = Dataset(
                        data=candidate_features,
                        label=candidate_values,
                    )
                    if is_cat:
                        folds = stratified_categorical_folds(candidate_values, nfold)
                    else:
                        folds = stratified_continuous_folds(candidate_values, nfold)

                    try:
                        loss, best_iteration = self._get_oof_performance(
                            parameters=sampled_parameters.copy(),
                            folds=folds,
                            train_set=train_set,
                        )
                    except Exception as err:
                        non_learners += 1
                        logger.log(f" - Lightgbm Error {err=}, {type(err)=}")
                        continue

                    if best_iteration > 1:
                        logger.log(f" - Success - Loss: {loss}")
                        break
                    else:
                        logger.log(" - Non-Learner")
                        non_learners += 1

                best_loss = self.optimal_parameter_losses.get(variable, np.inf)
                if loss < best_loss:
                    del sampled_parameters["seed"]
                    sampled_parameters["num_iterations"] = best_iteration
                    self.optimal_parameters[variable] = sampled_parameters
                    self.optimal_parameter_losses[variable] = loss

        self._ampute_original_data()
        return self.optimal_parameters

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
        new_data: pandas.DataFrame
            The new data to impute

        datasets: int or List[int], default = None
            The datasets from the kernel to use to impute the new data.
            If :code:`None`, all datasets from the kernel are used.

        iterations: int, default=None
            The number of iterations to run.
            If :code:`None`, the same number of iterations run so far in mice is used.

        save_all_iterations_data: bool, default=True
            Should the imputation values of all iterations be archived?
            If :code:`False`, only the latest imputation values are saved.

        copy_data: boolean, default=True
            Should the dataset be referenced directly? This will cause the dataset to be altered
            in place.

        random_state: None or int or np.random.RandomState (default=None)
            The random state of the process. Ensures reproducibility. If :code:`None`, the random state
            of the kernel is used. Beware, this permanently alters the random state of the kernel
            and ensures non-reproduceable results, unless the entire process up to this point
            is re-run.

        random_seed_array: None or np.ndarray[uint32, int32, uint64]
            Record-level seeds.

            Ensures deterministic imputations at the record level. random_seed_array causes
            deterministic imputations for each record no matter what dataset each record is
            imputed with, assuming the same number of iterations and datasets are used.
            If :code:`random_seed_array` is passed, random_state must also be passed.

            Record-level imputations are deterministic if the following conditions are met:
                1) The associated value in :code:`random_seed_array` is the same.
                2) The same kernel is used.
                3) The same number of iterations are run.
                4) The same number of datasets are run.

            Note: Using this parameter may change the global numpy seed by calling :code:`np.random.seed()`

        verbose: boolean, default=False
            Should information about the process be printed?

        Returns
        -------
        miceforest.ImputedData

        """

        assert self.save_all_iterations_data, (
            "Cannot recreate imputation procedure, data was not saved during MICE. "
            "To save this data, set save_all_iterations_data to True when making kernel."
        )

        # datasets = list(range(self.num_datasets)) if datasets is None else datasets
        datasets = self.datasets if datasets is None else datasets
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
            # num_datasets=len(datasets),
            datasets=datasets,
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
                imputed_data.complete_data(dataset=dataset, inplace=True)

                for variable in new_imputation_order:
                    logger.log(" | " + variable, end="")

                    # Select our model.
                    current_model = self.get_model(
                        variable=variable, dataset=dataset, iteration=iteration
                    )

                    time_key = dataset, iteration, variable, "Getting Bachelor Features"
                    logger.set_start_time(time_key)
                    bachelor_features = imputed_data._get_bachelor_features(variable)
                    hashed_seeds = imputed_data._get_hashed_seeds(variable)
                    logger.record_time(time_key)

                    time_key = dataset, iteration, variable, "Mean Matching"
                    logger.set_start_time(time_key)
                    na_where = imputed_data.na_where[variable]
                    imputation_values = self._mean_match_ind(
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

    def get_feature_importance(
        self,
        dataset: int = 0,
        iteration: int = -1,
        importance_type: str = "split",
        normalize: bool = True,
    ) -> DataFrame:
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
            The model must be saved to return importance.
            Use -1 to specify the latest iteration.

        importance_type: str
            Passed to :code:`lgb.feature_importance()`

        normalize: bool
            Whether to normalize the values within
            each modeled variable to sum to 1.

        Returns
        -------
        pandas.DataFrame of importance values. Rows are imputed variables, and columns are predictor variables.

        """

        if iteration == -1:
            iteration = self.iteration_count(dataset=dataset)

        modeled_vars = [
            col for col in self.working_data.columns if col in self.model_training_order
        ]

        importance_matrix = DataFrame(
            index=modeled_vars, columns=self.predictor_columns
        )
        for modeled_variable in modeled_vars:
            predictor_vars = self.variable_schema[modeled_variable]
            importances = self.get_model(
                variable=modeled_variable, dataset=dataset, iteration=iteration
            ).feature_importance(importance_type=importance_type)
            importances = Series(importances, index=predictor_vars)
            importance_matrix.loc[modeled_variable, predictor_vars] = importances

        importance_matrix = importance_matrix.astype("float64")

        if normalize:
            importance_matrix /= importance_matrix.sum(1).to_numpy().reshape(-1, 1)

        return importance_matrix

    def plot_feature_importance(
        self,
        dataset,
        importance_type: str = "split",
        normalize: bool = True,
        iteration: int = -1,
    ):
        """
        Plot the feature importance. See get_feature_importance()
        for more details.

        Parameters
        ----------
        dataset: int
            The dataset to plot the feature importance for.

        importance_type: str
            Passed to lgb.feature_importance()

        normalize: book
            Should the values be normalize from 0-1?
            If False, values are raw from Booster.feature_importance()

        kw_plot
            Additional arguments sent to sns.heatmap()

        """

        try:
            from plotnine import (
                aes,
                element_blank,
                element_text,
                geom_label,
                geom_tile,
                ggplot,
                ggtitle,
                scale_fill_distiller,
                theme,
                xlab,
                ylab,
            )
        except ImportError:
            raise ImportError("plotnine must be installed to plot importance")

        importance_matrix = self.get_feature_importance(
            dataset=dataset,
            iteration=iteration,
            normalize=normalize,
            importance_type=importance_type,
        )
        importance_matrix = importance_matrix.reset_index().melt(id_vars="index")
        importance_matrix["Importance"] = importance_matrix["value"].round(2)
        importance_matrix = importance_matrix.dropna()

        fig = (
            ggplot(importance_matrix, aes(x="variable", y="index", fill="Importance"))
            + geom_tile(show_legend=False)
            + ylab("Modeled Variable")
            + xlab("Predictor")
            + ggtitle("Feature Importance")
            + geom_label(aes(label="Importance"), fill="white", size=8)
            + scale_fill_distiller(palette=1, direction=1)
            + theme(
                axis_text_x=element_text(rotation=30, hjust=1),
                plot_title=element_text(ha="left", size=20),
                panel_background=element_blank(),
                figure_size=(6, 6),
            )
        )

        return fig
