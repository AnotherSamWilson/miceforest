from .ImputedData import ImputedData
from .MeanMatchScheme import MeanMatchScheme
from .utils import (
    _t_dat,
    _t_var_list,
    _t_var_dict,
    _t_var_sub,
    _t_random_state,
    _assert_dataset_equivalent,
    _draw_random_int32,
    _interpret_ds,
    _subset_data,
    ensure_rng,
    hash_int32,
    stratified_categorical_folds,
    stratified_continuous_folds,
    stratified_subset,
)
from .compat import pd_DataFrame, pd_Series
from .default_lightgbm_parameters import default_parameters, make_default_tuning_space
from .logger import Logger
import numpy as np
from numpy.random import RandomState
from warnings import warn
from lightgbm import train, Dataset, cv, log_evaluation, early_stopping, Booster
from lightgbm.basic import _ConfigAliases
from io import BytesIO
import blosc
import dill
from copy import copy
from typing import Union, List, Dict, Any, Optional

_DEFAULT_DATA_SUBSET = 1.0


class ImputationKernel(ImputedData):
    """
    Creates a kernel dataset. This dataset can perform MICE on itself,
    and impute new data from models obtained during MICE.

    Parameters
    ----------
    data : np.ndarray or pandas DataFrame.

        .. code-block:: text

            The data to be imputed.

    variable_schema : None or list or dict, default=None

        .. code-block:: text

            Specifies the feature - target relationships used to train models.
            This parameter also controls which models are built. Models can be built
            even if a variable contains no missing values, or is not being imputed
            (train_nonmissing must be set to True).

                - If None, all columns will be used as features in the training of each model.
                - If list, all columns in data are used to impute the variables in the list
                - If dict the values will be used to impute the keys. Can be either column
                    indices or names (if data is a pd.DataFrame).

            No models will be trained for variables not specified by variable_schema
            (either by None, a list, or in dict keys).

    imputation_order: str, list[str], list[int], default="ascending"

        .. code-block:: text

            The order the imputations should occur in. If a string from the
            items below, all variables specified by variable_schema with
            missing data are imputed:
                ascending: variables are imputed from least to most missing
                descending: most to least missing
                roman: from left to right in the dataset
                arabic: from right to left in the dataset.
            If a list is provided:
                - the variables will be imputed in that order.
                - only variables with missing values should be included in the list.
                - must be a subset of variables specified by variable_schema.
            If a variable with missing values is in variable_schema, but not in
            imputation_order, then models to impute that variable will be trained,
            but the actual values will not be imputed. See examples for details.

    train_nonmissing: boolean

        .. code-block:: text

            Should models be trained for variables with no missing values? Useful if you
            expect you will need to impute new data which will have missing values, but
            the training data is fully recognized.

            If True, parameters are interpreted like so:
                - models are run for all variables specified by variable_schema
                - if variable_schema is None, models are run for all variables
                - each iteration, models build for fully recognized variables are
                    always trained after the models trained during mice.
                - imputation_order does not have any affect on fully recognized
                    variable model training.

            WARNING: Setting this to True without specifying a variable schema will build
            models for all variables in the dataset, whether they have missing values or
            not. This may or may not be what you want.

    data_subset: None or int or float or dict.

        .. code-block:: text

            Subsets the data used in each iteration, which can save a significant amount of time.
            This can also help with memory consumption, as the candidate data must be copied to
            make a feature dataset for lightgbm.

            The number of rows used for each variable is (# rows in raw data) - (# missing variable values)
            for each variable. data_subset takes a random sample of this.

            If float, must be 0.0 < data_subset <= 1.0. Interpreted as a percentage of available candidates
            If int must be data_subset >= 0. Interpreted as the number of candidates.
            If 0, no subsetting is done.
            If dict, keys must be variable names, and values must follow two above rules.

            It is recommended to carefully select this value for each variable if dealing
            with very large data that barely fits into memory.

    mean_match_scheme: Dict, default = None

        .. code-block:: text

            An instance of the miceforest.MeanMatchScheme class.

            If None is passed, a sensible default scheme is used. There are multiple helpful
            schemes that can be accessed from miceforest.builtin_mean_match_schemes, or
            you can build your own.

            A description of the defaults:
            - mean_match_default (default, if mean_match_scheme is None))
                This scheme is has medium speed and accuracy for most data.

                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, get N nearest neighbors from class probabilities.
                        Select 1 at random.
                Numeric:
                    If mmc = 0, the predicted value is used
                    If mmc > 0, obtain the mmc closest candidate
                        predictions and collect the associated
                        real candidate values. Choose 1 randomly.

            - mean_match_shap
                This scheme is the most accurate, but takes the longest.
                It works the same as mean_match_default, except all nearest
                neighbor searches are performed on the shap values of the
                predictions, instead of the predictions themselves.

            - mean_match_scheme_fast_cat:
                This scheme is faster for categorical variables,
                but may be less accurate as well..

                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, return class based on random draw weighted by
                        class probability for each sample.
                Numeric or binary:
                    If mmc = 0, the predicted value is used
                    If mmc > 0, obtain the mmc closest candidate
                        predictions and collect the associated
                        real candidate values. Choose 1 randomly.

    categorical_feature: str or list, default="auto"

        .. code-block:: text

            The categorical features in the dataset. This handling depends on class of impute_data:

                pandas DataFrame:
                    - "auto": categorical information is inferred from any columns with
                        datatype category or object.
                    - list of column names (or indices): Useful if all categorical columns
                        have already been cast to numeric encodings of some type, otherwise you
                        should just use "auto". Will throw an error if a list is provided AND
                        categorical dtypes exist in data. If a list is provided, values in the
                        columns must be consecutive integers starting at 0, as required by lightgbm.

                numpy ndarray:
                    - "auto": no categorical information is stored.
                    - list of column indices: Specified columns are treated as categorical. Column
                        values must be consecutive integers starting at 0, as required by lightgbm.

    initialization: str

        .. code-block:: text

            "random" - missing values will be filled in randomly from existing values.
            "empty" - lightgbm will start MICE without initial imputation

    save_all_iterations: boolean, optional(default=True)

        .. code-block:: text

            Save all the imputation values from all iterations, or just
            the latest. Saving all iterations allows for additional
            plotting, but may take more memory

    save_models: int

        .. code-block:: text

            Which models should be saved:
                = 0: no models are saved. Cannot get feature importance or
                    impute new data.
                = 1: only the last model iteration is saved. Can only get
                    feature importance of last iteration. New data is
                    imputed using the last model for all specified iterations.
                    This is only an issue if data is heavily Missing At Random.
                = 2: all model iterations are saved. Can get feature importance
                    for any iteration. When imputing new data, each iteration is
                    imputed using the model obtained at that iteration in mice.
                    This allows for imputations that most closely resemble those
                    that would have been obtained in mice.

    copy_data: boolean (default = False)

        .. code-block:: text

            Should the dataset be referenced directly? If False, this will cause
            the dataset to be altered in place. If a copy is created, it is saved
            in self.working_data. There are different ways in which the dataset
            can be altered:

            1) complete_data() will fill in missing values
            2) To save space, mice() references and manipulates self.working_data directly.
                If self.working_data is a reference to the original dataset, the original
                dataset will undergo these manipulations during the mice process.
                At the end of the mice process, missing values will be set back to np.NaN
                where they were originally missing.

    save_loggers: boolean (default = False)

        .. code-block:: text

            A logger is created each time mice() or impute_new_data() is called.
            If True, the loggers are stored in a list ImputationKernel.loggers.
            If you wish to start saving logs, call ImputationKernel.start_logging().
            If you wish to stop saving logs, call ImputationKernel.stop_logging().

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
        data: _t_dat,
        datasets: int = 1,
        variable_schema: Union[_t_var_list, _t_var_dict, None] = None,
        imputation_order: Union[str, _t_var_list] = "ascending",
        train_nonmissing: bool = False,
        mean_match_scheme: Optional[MeanMatchScheme] = None,
        data_subset: Union[int, float, _t_var_sub, None] = None,
        categorical_feature: Union[str, _t_var_list] = "auto",
        initialization: str = "random",
        save_all_iterations: bool = True,
        save_models: int = 1,
        copy_data: bool = True,
        save_loggers: bool = False,
        random_state: _t_random_state = None,
    ):

        super().__init__(
            impute_data=data,
            datasets=datasets,
            variable_schema=variable_schema,
            imputation_order=imputation_order,
            train_nonmissing=train_nonmissing,
            categorical_feature=categorical_feature,
            save_all_iterations=save_all_iterations,
            copy_data=copy_data,
        )

        self.initialization = initialization
        self.train_nonmissing = train_nonmissing
        self.save_models = save_models
        self.save_loggers = save_loggers
        self.loggers: List[Logger] = []
        self.models: Dict[Any, Booster] = {}
        self.candidate_preds: Dict[Any, np.ndarray] = {}
        self.optimal_parameters: Dict[Any, Any] = {
            ds: {var: {} for var in self.variable_training_order}
            for ds in range(datasets)
        }
        self.optimal_parameter_losses: Dict[Any, Any] = {
            ds: {var: np.Inf for var in self.variable_training_order}
            for ds in range(datasets)
        }

        # Format data_subset and available_candidates
        available_candidates = {
            v: (self.data_shape[0] - self.na_counts[v])
            for v in self.variable_training_order
        }
        data_subset = _DEFAULT_DATA_SUBSET if data_subset is None else data_subset
        if not isinstance(data_subset, dict):
            data_subset = {v: data_subset for v in self.variable_training_order}
        if set(data_subset) != set(self.variable_training_order):
            # Change variable names to indices
            for v in list(data_subset):
                data_subset[self._get_var_ind_from_scalar(v)] = data_subset.pop(v)
            ds_supplement = {
                v: _DEFAULT_DATA_SUBSET
                for v in self.variable_training_order
                if v not in data_subset.keys()
            }
            data_subset.update(ds_supplement)
        for v, ds in data_subset.items():
            assert v in self.variable_training_order, (
                f"Variable {self._get_var_name_from_scalar(v)} will not have a model trained "
                + "but it is in data_subset."
            )
            data_subset[v] = _interpret_ds(data_subset[v], available_candidates[v])

        self.available_candidates = available_candidates
        self.data_subset = data_subset

        # Get mean matching function
        if mean_match_scheme is None:
            from .builtin_mean_match_schemes import mean_match_default

            self.mean_match_scheme = mean_match_default.copy()

        else:
            assert isinstance(mean_match_scheme, MeanMatchScheme)
            self.mean_match_scheme = mean_match_scheme.copy()

        # Format and run through mean match candidate checks.
        self.mean_match_scheme._format_mean_match_candidates(data, available_candidates)

        # Ensure mmc and mms make sense:
        # mmc <= mms <= available candidates for each var
        for v in self.imputation_order:
            mmc = self.mean_match_scheme.mean_match_candidates[v]
            assert mmc <= data_subset[v], f"{v} mean_match_candidates > data_subset"
            assert (
                data_subset[v] <= available_candidates[v]
            ), f"{v} data_subset > available candidates"

        # Make sure all pandas categorical levels are used.
        rare_levels = []
        for cat in self.categorical_variables:
            cat_name = self._get_var_name_from_scalar(cat)
            cat_dat = self._get_nonmissing_values(cat)
            cat_levels, cat_count = np.unique(cat_dat, return_counts=True)
            cat_dtype = cat_dat.dtype
            if cat_dtype.name == "category":
                levels_in_data = set(cat_levels)
                levels_in_catdt = set(cat_dtype.categories)
                levels_not_in_data = levels_in_catdt - levels_in_data
                assert (
                    len(levels_not_in_data) == 0
                ), f"{cat_name} has unused categories: {','.join(levels_not_in_data)}"

            if any(cat_count / cat_count.sum() < 0.002):
                rare_levels.append(cat_name)

        if len(rare_levels) > 0:
            warn(
                f"[{','.join(rare_levels)}] have very rare categories, it is a good "
                "idea to group these, or set the min_data_in_leaf parameter to prevent "
                "lightgbm from outputting 0.0 probabilities."
            )

        # Manage randomness
        self._completely_random_kernel = random_state is None
        self._random_state = ensure_rng(random_state)

        # Set initial imputations (iteration 0).
        self._initialize_dataset(
            self, random_state=self._random_state, random_seed_array=None
        )

    def __repr__(self):
        summary_string = f'\n{" " * 14}Class: ImputationKernel\n{self._ids_info()}'
        return summary_string

    # def _mm_type_handling(self, mm, available_candidates) -> int:
    #     if isinstance(mm, float):
    #
    #         assert (mm > 0.0) and (
    #             mm <= 1.0
    #         ), "mean_matching must be < 0.0 and >= 1.0 if a float"
    #
    #         ret = int(mm * available_candidates)
    #
    #     elif isinstance(mm, int):
    #
    #         assert mm >= 0, "mean_matching must be above 0 if an int is passed."
    #         ret = mm
    #
    #     else:
    #
    #         raise ValueError(
    #             "mean_match_candidates type not recognized. "
    #             + "Any supplied values must be a 0.0 < float <= 1.0 or int >= 1"
    #         )
    #
    #     return ret

    def _initialize_random_seed_array(self, random_seed_array, expected_shape):
        """
        Formats and takes the first hash of the random_seed_array.
        """

        # Format random_seed_array if it was passed.
        if random_seed_array is not None:
            if self._completely_random_kernel:
                warn(
                    """
                    This kernel is completely random (no random_state was provided on initialization).
                    Values imputed using ThisKernel.impute_new_data() will be deterministic, however
                    the kernel itself is non-reproducible.
                    """
                )
            assert isinstance(random_seed_array, np.ndarray)
            assert (
                random_seed_array.dtype == "int32"
            ), "random_seed_array must be a np.ndarray of type int32"
            assert (
                random_seed_array.shape[0] == expected_shape
            ), "random_seed_array must be the same length as data."
            random_seed_array = hash_int32(random_seed_array)
        else:
            random_seed_array = None

        return random_seed_array

    def _iter_pairs(self, new_iterations):
        """
        Returns the absolute and relative iterations that are going to be
        run for a given function call.
        """
        current_iters = self.iteration_count()
        iter_pairs = [(current_iters + i + 1, i + 1) for i in range(new_iterations)]
        return iter_pairs

    def _initialize_dataset(self, imputed_data, random_state, random_seed_array):
        """
        Sets initial imputation values for iteration 0.
        If "random", draw values from the kernel at random.
        If "empty", keep the values missing, since missing values
        can be handled natively by lightgbm.
        """

        assert not imputed_data.initialized, "dataset has already been initialized"

        if self.initialization == "random":
            for var in imputed_data.imputation_order:
                kernel_nonmissing_ind = self._get_nonmissing_indx(var)
                candidate_values = _subset_data(
                    self.working_data, kernel_nonmissing_ind, var, return_1d=True
                )
                n_candidates = kernel_nonmissing_ind.shape[0]
                missing_ind = imputed_data.na_where[var]

                for ds in range(imputed_data.dataset_count()):
                    # Initialize using the random_state if no record seeds were passed.
                    if random_seed_array is None:
                        imputed_data[ds, var, 0] = random_state.choice(
                            candidate_values,
                            size=imputed_data.na_counts[var],
                            replace=True,
                        )
                    else:
                        assert (
                            len(random_seed_array) == imputed_data.data_shape[0]
                        ), "The random_seed_array did not match the number of rows being imputed."
                        selection_ind = random_seed_array[missing_ind] % n_candidates
                        init_imps = candidate_values[selection_ind]
                        imputed_data[ds, var, 0] = np.array(init_imps)
                        random_seed_array[missing_ind] = hash_int32(
                            random_seed_array[missing_ind]
                        )

        elif self.initialization == "empty":
            for var in imputed_data.imputation_order:
                for ds in range(imputed_data.dataset_count()):
                    # Saves space, since np.nan will be broadcast.
                    imputed_data[ds, var, 0] = np.array([np.nan])

        else:
            raise ValueError("initialization parameter not recognized.")

        imputed_data.initialized = True

    def _reconcile_parameters(self, defaults, user_supplied):
        """
        Checks in user_supplied for aliases of each parameter in defaults.
        Combines the dicts once the aliases have been reconciled.
        """
        params = defaults.copy()
        for par, val in defaults.items():
            alias_names = _ConfigAliases.get(par)
            user_supplied_aliases = [
                i for i in alias_names if i in list(user_supplied) and i != par
            ]
            if len(user_supplied_aliases) == 0:
                continue
            elif len(user_supplied_aliases) == 1:
                params[par] = user_supplied.pop(user_supplied_aliases[0])
            else:
                raise ValueError(
                    f"Supplied 2 aliases for the same parameter: {user_supplied_aliases}"
                )

        params.update(user_supplied)

        return params

    def _format_variable_parameters(
        self, variable_parameters: Optional[Dict]
    ) -> Dict[int, Any]:
        """
        Unpacking will expect an empty dict at a minimum.
        This function collects parameters if they were
        provided, and returns empty dicts if they weren't.
        """
        if variable_parameters is None:

            vsp: Dict[int, Any] = {var: {} for var in self.variable_training_order}

        else:

            for variable in list(variable_parameters):
                variable_parameters[
                    self._get_var_ind_from_scalar(variable)
                ] = variable_parameters.pop(variable)
            vsp_vars = set(variable_parameters)

            assert vsp_vars.issubset(
                self.variable_training_order
            ), "Some variable_parameters are not associated with models being trained."
            vsp = {
                var: variable_parameters[var] if var in vsp_vars else {}
                for var in self.variable_training_order
            }

        return vsp

    def _get_lgb_params(self, var, vsp, random_state, **kwlgb):
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

        if var in self.categorical_variables:
            n_c = self.category_counts[var]
            if n_c > 2:
                obj = {"objective": "multiclass", "num_class": n_c}
            else:
                obj = {"objective": "binary"}
        else:
            obj = {"objective": "regression"}

        default_lgb_params = {**default_parameters, **obj, "seed": seed}

        # Priority is [variable specific] > [global in kwargs] > [defaults]
        params = self._reconcile_parameters(default_lgb_params, kwlgb)
        params = self._reconcile_parameters(params, vsp)

        return params

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

    def _get_nonmissing_values(self, variable):
        """
        Returns the non-missing values of a column.
        """

        var_indx = self._get_var_ind_from_scalar(variable)
        nonmissing_index = self._get_nonmissing_indx(variable)
        candidate_values = _subset_data(
            self.working_data, row_ind=nonmissing_index, col_ind=var_indx
        )
        return candidate_values

    def _get_candidate_subset(self, variable, subset_count, random_seed):
        """
        Returns a reproducible subset index of the
        non-missing values for a given variable.
        """

        var_indx = self._get_var_ind_from_scalar(variable)
        nonmissing_index = self._get_nonmissing_indx(var_indx)

        # Get the subset indices
        if subset_count < len(nonmissing_index):
            candidate_values = _subset_data(
                self.working_data, row_ind=nonmissing_index, col_ind=var_indx
            )
            candidates = candidate_values.shape[0]
            groups = max(10, int(candidates / 1000))
            ss = stratified_subset(
                y=candidate_values,
                size=subset_count,
                groups=groups,
                cat=var_indx in self.categorical_variables,
                seed=random_seed,
            )
            candidate_subset = nonmissing_index[ss]

        else:
            candidate_subset = nonmissing_index

        return candidate_subset

    def _make_label(self, variable, subset_count, random_seed):
        """
        Returns a reproducible subset of the non-missing values of a variable.
        """

        var_indx = self._get_var_ind_from_scalar(variable)
        candidate_subset = self._get_candidate_subset(
            var_indx, subset_count, random_seed
        )
        label = _subset_data(
            self.working_data, row_ind=candidate_subset, col_ind=var_indx
        )

        return label

    def _make_features_label(self, variable, subset_count, random_seed):
        """
        Makes a reproducible set of features and
        target needed to train a lightgbm model.
        """

        var_indx = self._get_var_ind_from_scalar(variable)
        candidate_subset = self._get_candidate_subset(
            var_indx, subset_count, random_seed
        )
        xvars = self.variable_schema[var_indx]
        ret_cols = sorted(xvars + [var_indx])
        features = _subset_data(
            self.working_data, row_ind=candidate_subset, col_ind=ret_cols
        )

        if self.original_data_class == "pd_DataFrame":
            y_name = self._get_var_name_from_scalar(var_indx)
            label = features.pop(y_name)

        elif self.original_data_class == "np_ndarray":
            y_col = ret_cols.index(var_indx)
            label = features[:, y_col].copy()
            features = np.delete(features, y_col, axis=1)

        x_cat = [xvars.index(var) for var in self.categorical_variables if var in xvars]

        return features, label, x_cat

    def append(self, imputation_kernel):
        """
        Combine two imputation kernels together.
        For compatibility, the following attributes of each must be equal:

            - working_data
            - iteration_count
            - categorical_feature
            - mean_match_scheme
            - variable_schema
            - imputation_order
            - save_models
            - save_all_iterations

        Only cursory checks are done to ensure working_data is equal.
        Appending a kernel with different working_data could ruin this kernel.

        Parameters
        ----------
        imputation_kernel: ImputationKernel
            The kernel to merge.

        """

        _assert_dataset_equivalent(self.working_data, imputation_kernel.working_data)
        assert self.iteration_count() == imputation_kernel.iteration_count()
        assert self.variable_schema == imputation_kernel.variable_schema
        assert self.imputation_order == imputation_kernel.imputation_order
        assert self.variable_training_order == imputation_kernel.variable_training_order
        assert self.categorical_feature == imputation_kernel.categorical_feature
        assert self.save_models == imputation_kernel.save_models
        assert self.save_all_iterations == imputation_kernel.save_all_iterations
        assert (
            self.mean_match_scheme.objective_pred_dtypes
            == imputation_kernel.mean_match_scheme.objective_pred_dtypes
        )
        assert (
            self.mean_match_scheme.objective_pred_funcs
            == imputation_kernel.mean_match_scheme.objective_pred_funcs
        )
        assert (
            self.mean_match_scheme.objective_args
            == imputation_kernel.mean_match_scheme.objective_args
        )
        assert (
            self.mean_match_scheme.mean_match_candidates
            == imputation_kernel.mean_match_scheme.mean_match_candidates
        )

        current_datasets = self.dataset_count()
        new_datasets = imputation_kernel.dataset_count()

        for key, model in imputation_kernel.models.items():
            new_ds_indx = key[0] + current_datasets
            insert_key = new_ds_indx, key[1], key[2]
            self.models[insert_key] = model

        for key, cp in imputation_kernel.candidate_preds.items():
            new_ds_indx = key[0] + current_datasets
            insert_key = new_ds_indx, key[1], key[2]
            self.candidate_preds[insert_key] = cp

        for key, iv in imputation_kernel.imputation_values.items():
            new_ds_indx = key[0] + current_datasets
            self[new_ds_indx, key[1], key[2]] = iv

        # Combine dicts
        for ds in range(new_datasets):
            insert_index = current_datasets + ds
            self.optimal_parameters[
                insert_index
            ] = imputation_kernel.optimal_parameters[ds]
            self.optimal_parameter_losses[
                insert_index
            ] = imputation_kernel.optimal_parameter_losses[ds]

        # Append iterations
        self.iterations = np.append(
            self.iterations, imputation_kernel.iterations, axis=0
        )

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

        assert self.dataset_count() == 1, (
            "miceforest kernel should be initialized with datasets=1 if "
            + "being used in a sklearn pipeline."
        )
        _assert_dataset_equivalent(self.working_data, X), (
            "The dataset passed to fit() was not the same as the "
            "dataset passed to ImputationKernel()"
        )
        self.mice(**fit_params)
        return self

    def get_model(
        self, dataset: int, variable: Union[str, int], iteration: Optional[int] = None
    ):
        """
        Return the model for a specific dataset, variable, iteration.

        Parameters
        ----------
        dataset: int
            The dataset to return the model for.

        var: str
            The variable that was imputed

        iteration: int
            The model iteration to return. Keep in mind if save_models ==1,
            the model was not saved. If none is provided, the latest model
            is returned.

        Returns: lightgbm.Booster
            The model used to impute this specific variable, iteration.
        """

        var_indx = self._get_var_ind_from_scalar(variable)
        itrn = (
            self.iteration_count(datasets=dataset, variables=var_indx)
            if iteration is None
            else iteration
        )
        try:
            return self.models[dataset, var_indx, itrn]
        except Exception:
            raise ValueError("Could not find model.")

    def get_raw_prediction(
        self,
        variable: Union[int, str],
        imp_dataset: int = 0,
        imp_iteration: Optional[int] = None,
        model_dataset: Optional[int] = None,
        model_iteration: Optional[int] = None,
        dtype: Union[str, np.dtype, None] = None,
    ):
        """
        Get the raw model output for a specific variable.

        The data is pulled from the imp_dataset dataset, at the imp_iteration iteration.
        The model is pulled from model_dataset dataset, at the model_iteration iteration.

        So, for example, it is possible to get predictions using the imputed values for
        dataset 3, at iteration 2, using the model obtained from dataset 10, at iteration
        6. This is assuming desired iterations and models have been saved.

        Parameters
        ----------
        variable: int or str
            The variable to get the raw predictions for.
            Can be an index or variable name.

        imp_dataset: int
            The imputation dataset to use when creating the feature dataset.

        imp_iteration: int
            The iteration from which to draw the imputation values when
            creating the feature dataset. If None, the latest iteration
            is used.

        model_dataset: int
            The dataset from which to pull the trained model for this variable.
            If None, it is selected to be the same as imp_dataset.

        model_iteration: int
            The iteration from which to pull the trained model for this variable
            If None, it is selected to be the same as imp_iteration.

        dtype: str, np.dtype
            The datatype to cast the raw prediction as.
            Passed to MeanMatchScheme.model_predict().

        Returns
        -------
        np.ndarray of raw predictions.

        """

        var_indx = self._get_var_ind_from_scalar(variable)
        predictor_variables = self.variable_schema[var_indx]

        # Get the latest imputation iteration if imp_iteration was not specified
        if imp_iteration is None:
            imp_iteration = self.iteration_count(
                datasets=imp_dataset, variables=var_indx
            )

        # If model dataset / iteration wasn't specified, assume it is from the same
        # dataset / iteration we are pulling the imputation values from
        model_iteration = imp_iteration if model_iteration is None else model_iteration
        model_dataset = imp_dataset if model_dataset is None else model_dataset

        # Get our internal dataset ready
        self.complete_data(dataset=imp_dataset, iteration=imp_iteration, inplace=True)

        features = _subset_data(self.working_data, col_ind=predictor_variables)
        model = self.get_model(model_dataset, var_indx, iteration=model_iteration)
        preds = self.mean_match_scheme.model_predict(model, features, dtype=dtype)

        return preds

    def mice(
        self,
        iterations=2,
        verbose=False,
        variable_parameters=None,
        compile_candidates=False,
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

        __MICE_TIMED_EVENTS = ["prepare_xy", "training", "predict", "mean_matching"]
        iter_pairs = self._iter_pairs(iterations)

        # Delete models and candidate_preds if we shouldn't be saving every iteration
        if self.save_models < 2:
            self.models = {}
            self.candidate_preds = {}

        logger = Logger(
            name=f"mice {str(iter_pairs[0][0])}-{str(iter_pairs[-1][0])}",
            verbose=verbose,
        )

        vsp = self._format_variable_parameters(variable_parameters)

        for ds in range(self.dataset_count()):

            logger.log("Dataset " + str(ds))

            # set self.working_data to the most current iteration.
            self.complete_data(dataset=ds, inplace=True)
            last_iteration = False

            for iter_abs, iter_rel in iter_pairs:

                logger.log(str(iter_abs) + " ", end="")
                if iter_rel == iterations:
                    last_iteration = True
                save_model = self.save_models == 2 or (
                    last_iteration and self.save_models == 1
                )

                for variable in self.variable_training_order:

                    var_name = self._get_var_name_from_scalar(variable)
                    logger.log(" | " + var_name, end="")
                    predictor_variables = self.variable_schema[variable]
                    data_subset = self.data_subset[variable]
                    nawhere = self.na_where[variable]
                    log_context = {
                        "dataset": ds,
                        "variable_name": var_name,
                        "iteration": iter_abs,
                    }

                    # Define the lightgbm parameters
                    lgbpars = self._get_lgb_params(
                        variable, vsp[variable], self._random_state, **kwlgb
                    )
                    objective = lgbpars["objective"]

                    # These are necessary for building model in mice.
                    logger.set_start_time()
                    (
                        candidate_features,
                        candidate_values,
                        feature_cat_index,
                    ) = self._make_features_label(
                        variable=variable,
                        subset_count=data_subset,
                        random_seed=lgbpars["seed"],
                    )
                    if (
                        self.original_data_class == "pd_DataFrame"
                        or len(feature_cat_index) == 0
                    ):
                        feature_cat_index = "auto"

                    # lightgbm requires integers for label. Categories won't work.
                    if candidate_values.dtype.name == "category":
                        candidate_values = candidate_values.cat.codes

                    num_iterations = lgbpars.pop("num_iterations")
                    train_pointer = Dataset(
                        data=candidate_features,
                        label=candidate_values,
                        categorical_feature=feature_cat_index,
                    )
                    logger.record_time(timed_event="prepare_xy", **log_context)
                    logger.set_start_time()
                    current_model = train(
                        params=lgbpars,
                        train_set=train_pointer,
                        num_boost_round=num_iterations,
                        categorical_feature=feature_cat_index,
                    )
                    logger.record_time(timed_event="training", **log_context)

                    if save_model:
                        self.models[ds, variable, iter_abs] = current_model

                    # Only perform mean matching and insertion
                    # if variable is being imputed.
                    if variable in self.imputation_order:
                        mean_match_args = self.mean_match_scheme.get_mean_match_args(
                            objective
                        )

                        # Start creating kwargs for mean matching function
                        mm_kwargs = {}

                        if "lgb_booster" in mean_match_args:
                            mm_kwargs["lgb_booster"] = current_model

                        if {"bachelor_preds", "bachelor_features"}.intersection(
                            mean_match_args
                        ):
                            logger.set_start_time()
                            bachelor_features = _subset_data(
                                self.working_data,
                                row_ind=nawhere,
                                col_ind=predictor_variables,
                            )
                            logger.record_time(timed_event="prepare_xy", **log_context)
                            if "bachelor_features" in mean_match_args:
                                mm_kwargs["bachelor_features"] = bachelor_features

                            if "bachelor_preds" in mean_match_args:
                                logger.set_start_time()
                                bachelor_preds = self.mean_match_scheme.model_predict(
                                    current_model, bachelor_features
                                )
                                logger.record_time(timed_event="predict", **log_context)
                                mm_kwargs["bachelor_preds"] = bachelor_preds

                        if "candidate_values" in mean_match_args:
                            mm_kwargs["candidate_values"] = candidate_values

                        if "candidate_features" in mean_match_args:
                            mm_kwargs["candidate_features"] = candidate_features

                        # Calculate the candidate predictions if
                        # the mean matching function calls for it
                        if "candidate_preds" in mean_match_args:
                            logger.set_start_time()
                            candidate_preds = self.mean_match_scheme.model_predict(
                                current_model, candidate_features
                            )
                            logger.record_time(timed_event="predict", **log_context)
                            mm_kwargs["candidate_preds"] = candidate_preds

                            if compile_candidates and save_model:
                                self.candidate_preds[
                                    ds, variable, iter_abs
                                ] = candidate_preds

                        if "random_state" in mean_match_args:
                            mm_kwargs["random_state"] = self._random_state

                        # Hashed seeds are only to ensure record reproducibility
                        # for impute_new_data().
                        if "hashed_seeds" in mean_match_args:
                            mm_kwargs["hashed_seeds"] = None

                        logger.set_start_time()
                        imp_values = self.mean_match_scheme._mean_match(
                            variable, objective, **mm_kwargs
                        )
                        logger.record_time(timed_event="mean_matching", **log_context)

                        assert imp_values.shape == (
                            self.na_counts[variable],
                        ), f"{variable} mean matching returned malformed array"

                        # Updates our working data and saves the imputations.
                        self._insert_new_data(
                            dataset=ds, variable_index=variable, new_data=imp_values
                        )

                    self.iterations[
                        ds, self.variable_training_order.index(variable)
                    ] += 1

                logger.log("\n", end="")

        self._ampute_original_data()
        if self.save_loggers:
            self.loggers.append(logger)

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
        variables: Union[List[int], List[str], None] = None,
        variable_parameters: Optional[Dict[Any, Any]] = None,
        parameter_sampling_method: str = "random",
        nfold: int = 10,
        optimization_steps: int = 5,
        random_state: _t_random_state = None,
        verbose: bool = False,
        **kwbounds,
    ):
        """
        Perform hyperparameter tuning on models at the current iteration.

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
                int((self.data_shape[0] - len(self.na_where[var])) / 10),
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
        new_data: _t_dat,
        datasets: Optional[List[int]] = None,
        iterations: Optional[int] = None,
        save_all_iterations: bool = True,
        copy_data: bool = True,
        random_state: _t_random_state = None,
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

        datasets = list(range(self.dataset_count())) if datasets is None else datasets
        kernel_iterations = self.iteration_count()
        iterations = kernel_iterations if iterations is None else iterations
        iter_pairs = self._iter_pairs(iterations)
        __IND_TIMED_EVENTS = ["prepare_xy", "predict", "mean_matching"]
        logger = Logger(
            name=f"ind {str(iter_pairs[0][1])}-{str(iter_pairs[-1][1])}",
            verbose=verbose,
        )

        if isinstance(self.working_data, pd_DataFrame):
            assert isinstance(new_data, pd_DataFrame)
            assert set(self.working_data.columns) == set(
                new_data.columns
            ), "Different columns from original dataset."
            assert all(
                [
                    self.working_data[col].dtype == new_data[col].dtype
                    for col in self.working_data.columns
                ]
            ), "Column types are not the same as the original data. Check categorical columns."

        if self.save_models < 1:
            raise ValueError("No models were saved.")

        imputed_data = ImputedData(
            impute_data=new_data,
            datasets=len(datasets),
            variable_schema=self.variable_schema.copy(),
            imputation_order=self.variable_training_order.copy(),
            train_nonmissing=False,
            categorical_feature=self.categorical_feature,
            save_all_iterations=save_all_iterations,
            copy_data=copy_data,
        )

        ### Manage Randomness.
        if random_state is None:
            assert (
                random_seed_array is None
            ), "random_state is also required when using random_seed_array"
            random_state = self._random_state
        else:
            random_state = ensure_rng(random_state)
        # use_seed_array = random_seed_array is not None
        random_seed_array = self._initialize_random_seed_array(
            random_seed_array=random_seed_array,
            expected_shape=imputed_data.data_shape[0],
        )
        self._initialize_dataset(
            imputed_data, random_state=random_state, random_seed_array=random_seed_array
        )

        for ds_kern in datasets:

            logger.log("Dataset " + str(ds_kern))
            self.complete_data(dataset=ds_kern, inplace=True)
            ds_new = datasets.index(ds_kern)
            imputed_data.complete_data(dataset=ds_new, inplace=True)

            for iter_abs, iter_rel in iter_pairs:

                logger.log(str(iter_rel) + " ", end="")

                # Determine which model iteration to grab
                if self.save_models == 1 or iter_abs > kernel_iterations:
                    iter_model = kernel_iterations
                else:
                    iter_model = iter_abs

                for var in imputed_data.imputation_order:

                    var_name = self._get_var_name_from_scalar(var)
                    logger.log(" | " + var_name, end="")
                    log_context = {
                        "dataset": ds_kern,
                        "variable_name": var_name,
                        "iteration": iter_rel,
                    }
                    nawhere = imputed_data.na_where[var]
                    predictor_variables = self.variable_schema[var]

                    # Select our model.
                    current_model = self.get_model(
                        variable=var, dataset=ds_kern, iteration=iter_model
                    )
                    objective = current_model.params["objective"]
                    model_seed = current_model.params["seed"]

                    # Start building mean matching kwargs
                    mean_match_args = self.mean_match_scheme.get_mean_match_args(
                        objective
                    )
                    mm_kwargs = {}

                    if "lgb_booster" in mean_match_args:
                        mm_kwargs["lgb_booster"] = current_model

                    # Procure bachelor information
                    if {"bachelor_preds", "bachelor_features"}.intersection(
                        mean_match_args
                    ):
                        logger.set_start_time()
                        bachelor_features = _subset_data(
                            imputed_data.working_data,
                            row_ind=imputed_data.na_where[var],
                            col_ind=predictor_variables,
                        )
                        logger.record_time(timed_event="prepare_xy", **log_context)
                        if "bachelor_features" in mean_match_args:
                            mm_kwargs["bachelor_features"] = bachelor_features

                        if "bachelor_preds" in mean_match_args:
                            logger.set_start_time()
                            bachelor_preds = self.mean_match_scheme.model_predict(
                                current_model, bachelor_features
                            )
                            logger.record_time(timed_event="predict", **log_context)
                            mm_kwargs["bachelor_preds"] = bachelor_preds

                    # Procure candidate information
                    if {
                        "candidate_values",
                        "candidate_features",
                        "candidate_preds",
                    }.intersection(mean_match_args):

                        # Need to return candidate features if we need to calculate
                        # candidate_preds or candidate_features is needed by mean matching function
                        calculate_candidate_preds = (
                            ds_kern,
                            var,
                            iter_model,
                        ) not in self.candidate_preds.keys() and "candidate_preds" in mean_match_args
                        return_features = (
                            "candidate_features" in mean_match_args
                        ) or calculate_candidate_preds
                        # Set up like this so we only have to subset once
                        logger.set_start_time()
                        if return_features:
                            (
                                candidate_features,
                                candidate_values,
                                _,
                            ) = self._make_features_label(
                                variable=var,
                                subset_count=self.data_subset[var],
                                random_seed=model_seed,
                            )
                        else:
                            candidate_values = self._make_label(
                                variable=var,
                                subset_count=self.data_subset[var],
                                random_seed=model_seed,
                            )
                        logger.record_time(timed_event="prepare_xy", **log_context)

                        if "candidate_values" in mean_match_args:
                            # lightgbm requires integers for label. Categories won't work.
                            if candidate_values.dtype.name == "category":
                                candidate_values = candidate_values.cat.codes
                            mm_kwargs["candidate_values"] = candidate_values

                        if "candidate_features" in mean_match_args:
                            mm_kwargs["candidate_features"] = candidate_features

                        # Calculate the candidate predictions if
                        # the mean matching function calls for it
                        if "candidate_preds" in mean_match_args:
                            if calculate_candidate_preds:
                                logger.set_start_time()
                                candidate_preds = self.mean_match_scheme.model_predict(
                                    current_model, candidate_features
                                )
                                logger.record_time(timed_event="predict", **log_context)
                            else:
                                candidate_preds = self.candidate_preds[
                                    ds_kern, var, iter_model
                                ]
                            mm_kwargs["candidate_preds"] = candidate_preds

                    if "random_state" in mean_match_args:
                        mm_kwargs["random_state"] = random_state

                    if "hashed_seeds" in mean_match_args:
                        if isinstance(random_seed_array, np.ndarray):
                            seeds = random_seed_array[nawhere]
                            rehash_seeds = True

                        else:
                            seeds = None
                            rehash_seeds = False

                        mm_kwargs["hashed_seeds"] = seeds

                    else:
                        rehash_seeds = False

                    logger.set_start_time()
                    imp_values = self.mean_match_scheme._mean_match(
                        var, objective, **mm_kwargs
                    )
                    logger.record_time(timed_event="mean_matching", **log_context)
                    imputed_data._insert_new_data(
                        dataset=ds_new, variable_index=var, new_data=imp_values
                    )
                    # Refresh our seeds.
                    if rehash_seeds:
                        assert isinstance(random_seed_array, np.ndarray)
                        random_seed_array[nawhere] = hash_int32(seeds)

                    imputed_data.iterations[
                        ds_new, imputed_data.imputation_order.index(var)
                    ] += 1

                logger.log("\n", end="")

        imputed_data._ampute_original_data()
        if self.save_loggers:
            self.loggers.append(logger)

        return imputed_data

    def start_logging(self):
        """
        Start saving loggers to self.loggers
        """

        self.save_loggers = True

    def stop_logging(self):
        """
        Stop saving loggers to self.loggers
        """

        self.save_loggers = False

    def save_kernel(
        self, filepath, clevel=None, cname=None, n_threads=None, copy_while_saving=True
    ):
        """
        Compresses and saves the kernel to a file.

        Parameters
        ----------
        filepath: str
            The file to save to.

        clevel: int
            The compression level, sent to clevel argument in blosc.compress()

        cname: str
            The compression algorithm used.
            Sent to cname argument in blosc.compress.
            If None is specified, the default is lz4hc.

        n_threads: int
            The number of threads to use for compression.
            By default, all threads are used.

        copy_while_saving: boolean
            Should the kernel be copied while saving? Copying is safer, but
            may take more memory.

        """

        clevel = 9 if clevel is None else clevel
        cname = "lz4hc" if cname is None else cname
        n_threads = blosc.detect_number_of_cores() if n_threads is None else n_threads

        if copy_while_saving:
            kernel = copy(self)
        else:
            kernel = self

        # convert working data to parquet bytes object
        if kernel.original_data_class == "pd_DataFrame":
            working_data_bytes = BytesIO()
            kernel.working_data.to_parquet(working_data_bytes)
            kernel.working_data = working_data_bytes

        blosc.set_nthreads(n_threads)

        with open(filepath, "wb") as f:
            dill.dump(
                blosc.compress(
                    dill.dumps(kernel),
                    clevel=clevel,
                    typesize=8,
                    shuffle=blosc.NOSHUFFLE,
                    cname=cname,
                ),
                f,
            )

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
            iteration = self.iteration_count(datasets=dataset)

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
            **kw_plot
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
