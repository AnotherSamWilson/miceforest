from .ImputedData import ImputedData
from .utils import (
    _get_default_mmc,
    _get_default_mms,
    param_mapping,
    _ensure_iterable,
    ensure_rng,
    stratified_continuous_folds,
    stratified_categorical_folds,
    _subset_data,
    _is_int,
    hash_int32,
)
import numpy as np
from warnings import warn
from .logger import Logger
from lightgbm import train, Dataset, cv, log_evaluation, early_stopping, Booster
from .default_lightgbm_parameters import default_parameters, make_default_tuning_space
from io import BytesIO
import blosc
import dill
from copy import copy


class ImputationKernel(ImputedData):
    """Creates a kernel dataset. This dataset can perform MICE on itself,
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

    mean_match_candidates:  None or int or float or dict

        .. code-block:: text

            The number of mean matching candidates to use.
            Candidates are _always_ drawn from a kernel dataset, even
            when imputing new data.

            If float must be 0.0 < mmc <= 1.0. Interpreted as a percentage of available candidates
            If int must be mmc >= 0. Interpreted as the number of candidates.
            If dict, keys must be variable names or indexes, and values must follow two above rules.

            For more information, see:
            https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

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

    mean_match_function: Callable, default = None

        .. code-block:: text

            Must take the following parameters:
                mmc: int,
                candidate_preds: np.ndarray,
                bachelor_preds: np.ndarray,
                candidate_values: np.ndarray,
                cat_dtype: CategoricalDtype,
                random_state: np.random.RandomState,

            A default mean matching function will be used if None.
            There are multiple built-in functions available. See the miceforest.mean_matching_functions module.
            The built in behavior is as follows, for each function:

            - default_mean_match (default, if mean_match_function is None):
                This function is very fast, but may be less accurate for categorical variables.

                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, return class based on random draw weighted by
                        class probability for each sample.
                Numeric or binary:
                    If mmc = 0, the predicted value is used
                    If mmc > 0, obtain the mmc closest candidate
                        predictions and collect the associated
                        real candidate values. Choose 1 randomly.

            - mean_match_kdtree_classification
                This function is slower for categorical datatypes, but results in better imputations.

                Categorical:
                    If mmc = 0, the class with the highest probability is chosen.
                    If mmc > 0, get N nearest neighbors from class probabilities.
                        Select 1 at random.
                Numeric:
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
        data,
        datasets=5,
        variable_schema=None,
        imputation_order="ascending",
        train_nonmissing=False,
        mean_match_candidates=None,
        data_subset=None,
        mean_match_function=None,
        categorical_feature="auto",
        initialization="random",
        save_all_iterations=True,
        save_models=1,
        copy_data=True,
        save_loggers=False,
        random_state=None,
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
        self.loggers = []
        self.models = {
            ds: {var: {0: None} for var in self.variable_training_order}
            for ds in range(datasets)
        }
        self.optimal_parameters = {
            ds: {var: {} for var in self.variable_training_order}
            for ds in range(datasets)
        }
        self.optimal_parameter_losses = {
            ds: {var: np.Inf for var in self.variable_training_order}
            for ds in range(datasets)
        }

        # Format mean_match_candidates before priming datasets
        available_candidates = {
            var: (self.data_shape[0] - self.na_counts[var])
            for var in self.variable_training_order
        }
        mean_match_candidates = self._format_mm(
            mean_match_candidates, available_candidates, _get_default_mmc
        )
        data_subset = self._format_mm(
            data_subset, available_candidates, _get_default_mms
        )

        # Ensure mmc and mms make sense:
        # mmc <= mms <= available candidates for each var
        for var in self.imputation_order:

            assert (
                mean_match_candidates[var] <= data_subset[var]
            ), f"{var} mean_match_candidates > data_subset"

            assert (
                data_subset[var] <= available_candidates[var]
            ), f"{var} data_subset > available candidates"

        self.mean_match_candidates = mean_match_candidates
        self.data_subset = data_subset

        # Get mean matching function
        if mean_match_function is None:
            from .mean_matching_functions import default_mean_match

            self.mean_match_function = default_mean_match

        else:
            self.mean_match_function = mean_match_function

        # Manage randomness
        self._completely_random_kernel = random_state is None
        self._random_state = ensure_rng(random_state)

        # Set initial imputations (iteration 0).
        self._initialize_dataset(
            self, random_state=self._random_state, random_seed_array=None
        )

    def __repr__(self):
        summary_string = " " * 14 + "Class: ImputationKernel\n" + self._ids_info()
        return summary_string

    def _mm_type_handling(self, mm, available_candidates) -> int:
        if isinstance(mm, float):

            assert (mm > 0.0) and (
                mm <= 1.0
            ), "mean_matching must be < 0.0 and >= 1.0 if a float"

            ret = int(mm * available_candidates)

        elif isinstance(mm, int):

            assert mm >= 0, "mean_matching must be above 0 if an int is passed."
            ret = mm

        else:

            raise ValueError(
                "mean_match_candidates type not recognized. "
                + "Any supplied values must be a 0.0 < float <= 1.0 or int >= 1"
            )

        return ret

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
        iter_pairs = [(current_iters + 1 + i, i) for i in range(new_iterations)]
        return iter_pairs

    def _format_mm(self, mm, available_candidates, defaulting_function):
        """
        mean_match_candidates and data_subset both require similar formatting.
        The only real difference is the default value based on the number of available
        candidates, which is what the defaulting_function is for.
        """
        if mm is None:

            mm = {
                var: int(defaulting_function(available_candidates[var]))
                for var in self.variable_training_order
            }

        elif isinstance(mm, (int, float)):

            mm = {
                var: self._mm_type_handling(mm, available_candidates[var])
                for var in self.variable_training_order
            }

        elif isinstance(mm, dict):
            mm = self._get_variable_index(mm)
            mm = {
                var: self._mm_type_handling(mm[var], available_candidates[var])
                if var in mm.keys()
                else defaulting_function(available_candidates[var])
                for var in self.variable_training_order
            }
        else:
            raise ValueError(
                "one of the mean_match parameters couldn't be interpreted."
            )

        return mm

    def _insert_new_model(self, dataset, variable_index, model):
        """
        Inserts a new model if save_mdoels > 0.
        Deletes the prior one if save_models == 1.
        """
        current_variable_iteration = self.iteration_count(
            datasets=dataset, variables=variable_index
        )
        if self.save_models == 0:
            return
        else:
            self.models[dataset][variable_index][current_variable_iteration + 1] = model
        if self.save_models == 1:
            del self.models[dataset][variable_index][current_variable_iteration]

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

                kernel_nonmissing_ind = np.setdiff1d(
                    np.arange(self.data_shape[0]), self.na_where[var]
                )
                candidates = _subset_data(
                    self.working_data, kernel_nonmissing_ind, var, return_1d=True
                )
                missing_ind = imputed_data.na_where[var]

                for ds in range(imputed_data.dataset_count()):

                    # Initialize using the random_state if no record seeds were passed.
                    if random_seed_array is None:
                        imputed_data[ds, var, 0] = random_state.choice(
                            candidates, size=imputed_data.na_counts[var], replace=True
                        )
                    else:
                        assert (
                            len(random_seed_array) == imputed_data.data_shape[0]
                        ), "The random_seed_array did not match the number of rows being imputed."
                        init_imps = []
                        for i in missing_ind:
                            np.random.seed(random_seed_array[i])
                            init_imps.append(np.random.choice(candidates, size=1)[0])
                        imputed_data[ds, var, 0] = np.array(init_imps)
                        random_seed_array[missing_ind] = hash_int32(
                            random_seed_array[missing_ind]
                        )

        elif self.initialization == "empty":

            for var in imputed_data.imputation_order:

                for ds in range(imputed_data.dataset_count()):
                    # Saves space, since np.nan will be broadcast.
                    imputed_data[ds, var, 0] = np.nan

        else:
            raise ValueError("initialization parameter not recognized.")

        imputed_data.initialized = True

    def _fix_parameter_aliases(self, parameters):
        """Replaces aliases with true names in lightgbm parameters."""
        for par in list(parameters):
            if par in param_mapping.keys():
                parameters[param_mapping[par]] = parameters.pop(par)

    def _format_variable_parameters(self, variable_parameters):
        """
        Unpacking will expect an empty dict at a minimum.
        This function collects parameters if they were
        provided, and returns empty dicts if they weren't.
        """
        if variable_parameters is None:

            vsp = {var: {} for var in self.variable_training_order}

        else:

            variable_parameters = self._get_variable_index(variable_parameters)
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

        seed = random_state.randint(
            low=np.iinfo("int32").min,
            high=np.iinfo("int32").max,
            size=1,
            dtype="int32"
        )[0]

        if var in self.categorical_variables:
            n_c = self.category_counts[var]
            if n_c > 2:
                obj = {"objective": "multiclass", "num_class": n_c}
            else:
                obj = {"objective": "binary"}
        else:
            obj = {"objective": "regression"}

        default_lgb_params = {**default_parameters, **obj, "seed": seed}

        # Set any user variables to the defaults.
        user_set_lgb_params = {**kwlgb, **vsp}
        self._fix_parameter_aliases(user_set_lgb_params)

        # Priority is [variable specific] > [global in kwargs] > [defaults]
        params = {**default_lgb_params, **user_set_lgb_params}

        return params

    def _get_random_sample(self, parameters, random_state):
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

    def _draw_random_int32(self, random_state, size):
        nums = random_state.randint(
            low=0,
            high=np.iinfo("int32").max,
            size=size,
            dtype="int32"
        )
        return nums

    def _make_xy(self, variable, subset_count, return_cat=False, random_seed=None):
        """
        Make the predictor and response set used to train the model.
        Must be defined in ImputedData because this method is called
        directly in ImputationKernel.impute_new_data()
        """
        var = self._get_variable_index(variable)
        assert _is_int(var)
        xvars = self.variable_schema[var]
        non_missing_ind = self._get_working_data_nonmissing_indx(var)

        # Only get subset of indices if we need to.
        if subset_count < len(non_missing_ind):
            random_state = np.random.RandomState(seed=random_seed)
            candidate_subset = random_state.choice(
                non_missing_ind, size=subset_count
            )
        else:
            candidate_subset = non_missing_ind

        if self.original_data_class == "pd_DataFrame":
            x = _subset_data(
                self.working_data, row_ind=candidate_subset, col_ind=xvars + [var]
            ).reset_index(drop=True)
            y = x.pop(self._get_variable_name(var))

        elif self.original_data_class == "np_ndarray":
            # Don't think we can get around subsetting twice. Luckily numpy has fast indexing.
            x = _subset_data(self.working_data, row_ind=candidate_subset, col_ind=xvars)
            y = _subset_data(
                self.working_data, row_ind=candidate_subset, col_ind=var, return_1d=True
            )
        else:
            raise ValueError("Unknown data class.")

        if return_cat:
            cat = [
                xvars.index(var) for var in self.categorical_variables if var in xvars
            ]
            return x, y, cat
        else:
            return x, y

    def append(self, imputation_kernel):
        """
        Combine two imputation kernels together.
        For compatibility, the following attributes of each must be equal:
            - working_data
            - iteration_count
            - categorical_feature
            - mean_match_function
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
        assert self.working_data.shape == imputation_kernel.working_data.shape
        assert self.iteration_count() == imputation_kernel.iteration_count()
        assert (
            self.mean_match_function.__code__.co_code
            == imputation_kernel.mean_match_function.__code__.co_code
        )
        assert self.variable_schema == imputation_kernel.variable_schema
        assert self.imputation_order == imputation_kernel.imputation_order
        assert self.variable_training_order == imputation_kernel.variable_training_order
        assert self.categorical_feature == imputation_kernel.categorical_feature
        assert self.save_models == imputation_kernel.save_models
        assert self.save_all_iterations == imputation_kernel.save_all_iterations

        current_datasets = self.dataset_count()
        new_datasets = imputation_kernel.dataset_count()

        # Combine dicts
        for ds in range(new_datasets):
            insert_index = current_datasets + ds
            self.imputation_values[insert_index] = imputation_kernel.imputation_values[
                ds
            ]
            self.models[insert_index] = imputation_kernel.models[ds]
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

    def get_model(self, dataset, variable, iteration=None):
        """
        Return the model for a specific dataset, variable, iteration.

        Parameters
        ----------
        dataset: int
            The dataset to return the model for.
        var: str
            The variable that was imputed
        iteration: int
            The model iteration to return. Keep in mind if:

        Returns: lightgbm.Booster
            The model used to impute this specific variable, iteration.

        """

        var_indx = self._get_variable_index(variable)
        itrn = (
            self.iteration_count(datasets=dataset, variables=var_indx)
            if iteration is None
            else iteration
        )
        try:
            return self.models[dataset][var_indx][itrn]
        except Exception:
            raise ValueError("Could not find model.")

    def get_raw_prediction(
        self,
        variable,
        imp_dataset=0,
        imp_iteration=None,
        model_dataset=None,
        model_iteration=None,
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

        Returns
        -------
        np.ndarray of raw predictions.

        """

        var_indx = self._get_variable_index(variable)
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

        return self.get_model(
            model_dataset, var_indx, iteration=model_iteration
        ).predict(features)

    def mice(
        self,
        iterations=5,
        verbose=False,
        variable_parameters=None,
        **kwlgb,
    ):
        """
        Perform mice given dataset.

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
        kwlgb:
            Additional arguments to pass to lightgbm. Applied to all models.

        """

        __MICE_TIMED_EVENTS = ["prepare_xy", "training", "mean_matching"]
        iter_pairs = self._iter_pairs(iterations)

        logger = Logger(
            name=f"mice {str(iter_pairs[0][0])}-{str(iter_pairs[-1][0])}",
            datasets=self.dataset_count(),
            variable_names=self._get_variable_name(self.variable_training_order),
            iterations=iterations,
            timed_events=__MICE_TIMED_EVENTS,
            verbose=verbose,
        )

        vsp = self._format_variable_parameters(variable_parameters)

        for ds in range(self.dataset_count()):

            logger.log("Dataset " + str(ds))

            # set self.working_data to the most current iteration.
            self.complete_data(dataset=ds, inplace=True)

            for iter_abs, iter_rel in iter_pairs:

                logger.log(str(iter_abs) + " ", end="")

                for var in self.variable_training_order:

                    logger.log(" | " + self._get_variable_name(var), end="")
                    predictor_variables = self.variable_schema[var]
                    nawhere = self.na_where[var]
                    log_context = {
                        "dataset": ds,
                        "variable_name": self._get_variable_name(var),
                        "iteration": iter_rel
                    }

                    # These are necessary for building model in mice.
                    logger.set_start_time()
                    (
                        candidate_features,
                        candidate_values,
                        feature_cat_index,
                    ) = self._make_xy(
                        variable=var,
                        subset_count=self.data_subset[var],
                        return_cat=True,
                        random_seed=self._draw_random_int32(
                            random_state=self._random_state,
                            size=1
                        )[0]
                    )
                    feature_cat_index = (
                        "auto" if len(feature_cat_index) == 0 else feature_cat_index
                    )

                    # lightgbm requires integers for label. Categories won't work.
                    if candidate_values.dtype.name == "category":
                        candidate_values = candidate_values.cat.codes

                    lgbpars = self._get_lgb_params(
                        var, vsp[var], self._random_state, **kwlgb
                    )
                    num_iterations = lgbpars.pop("num_iterations")
                    train_pointer = Dataset(
                        data=candidate_features,
                        label=candidate_values,
                        categorical_feature=feature_cat_index,
                        # free_raw_data=True,
                    )
                    logger.record_time("prepare_xy", **log_context)
                    logger.set_start_time()
                    current_model = Booster(
                        params=lgbpars,
                        train_set=train_pointer
                    )
                    for i in range(num_iterations):
                        current_model.update()
                    logger.record_time("training", **log_context)
                    self._insert_new_model(
                        dataset=ds, variable_index=var, model=current_model
                    )
                    # Only perform mean matching and insertion if variable
                    # is being imputed.
                    if var in self.imputation_order:
                        logger.set_start_time()
                        bachelor_features = _subset_data(
                            self.working_data,
                            row_ind=nawhere,
                            col_ind=predictor_variables,
                        )
                        imp_values = np.array(
                            self.mean_match_function(
                                mmc=self.mean_match_candidates[var],
                                model=current_model,
                                bachelor_features=bachelor_features,
                                candidate_values=candidate_values,
                                random_state=self._random_state,
                                hashed_seeds=None,
                            )
                        )
                        logger.record_time("mean_matching", **log_context)
                        assert imp_values.shape == (
                            self.na_counts[var],
                        ), f"{var} mean matching returned malformed array"

                        self._insert_new_data(
                            dataset=ds, variable_index=var, new_data=imp_values
                        )

                    self.iterations[ds, self.variable_training_order.index(var)] += 1

                logger.log("\n", end="")

        self._ampute_original_data()
        if self.save_loggers:
            self.loggers.append(logger)

    def tune_parameters(
        self,
        dataset,
        variables=None,
        variable_parameters=None,
        parameter_sampling_method="random",
        nfold=10,
        optimization_steps=5,
        random_state=None,
        verbose=False,
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
            for future iterations for extremely accurate imputation.
                {variable: {parameter_name: parameter_value}}

        - optimal_parameter_losses: dict
            The average out of fold cv loss obtained directly from
            lightgbm.cv() associated with the optimal parameter set.
                {variable: loss}

        """

        if random_state is None:
            random_state = self._random_state
        else:
            random_state = ensure_rng(random_state)

        if variables is None:
            variables = self.imputation_order
        else:
            variables = self._get_variable_index(variables)

        logger = Logger(
            name=f"tune: {optimization_steps}",
            datasets=self.dataset_count(),
            variable_names=self._get_variable_name(variables),
            iterations=optimization_steps,
            timed_events=[],
            verbose=verbose,
        )

        self._fix_parameter_aliases(kwbounds)
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

                logger.log(self._get_variable_name(var) + " | ", end="")

                candidate_features, candidate_values, feature_cat_index = self._make_xy(
                    variable=var,
                    subset_count=self.data_subset[var],
                    return_cat=True,
                    random_seed=self._draw_random_int32(
                        random_state=self._random_state,
                        size=1
                    )[0]
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
                        sampling_point["num_iterations"] = best_iteration
                        self.optimal_parameters[dataset][var] = sampling_point
                        self.optimal_parameter_losses[dataset][var] = loss

                    logger.log(" - ", end="")

                logger.log("\n", end="")

            return (
                self.optimal_parameters[dataset],
                self.optimal_parameter_losses[dataset],
            )

    def impute_new_data(
        self,
        new_data,
        datasets=None,
        iterations=None,
        save_all_iterations=True,
        copy_data=True,
        random_state=None,
        random_seed_array=None,
        verbose=False,
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
            2) To save space, mice() references and manipulates self.working_data directly.
                If self.working_data is a reference to the original dataset, the original
                dataset will undergo these manipulations during the mice process.

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

        datasets = (
            range(self.dataset_count())
            if datasets is None
            else _ensure_iterable(datasets)
        )
        kernel_iterations = self.iteration_count()
        iterations = kernel_iterations if iterations is None else iterations
        iter_pairs = self._iter_pairs(iterations)
        __IND_TIMED_EVENTS = ["prepare_xy", "mean_matching"]
        logger = Logger(
            name=f"ind {str(iter_pairs[0][0])}-{str(iter_pairs[-1][0])}",
            datasets=self.dataset_count(),
            variable_names=self._get_variable_name(self.variable_training_order),
            iterations=iterations,
            timed_events=__IND_TIMED_EVENTS,
            verbose=verbose,
        )


        if self.original_data_class == "pd_DataFrame":
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
            train_nonmissing=self.train_nonmissing,
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
        use_seed_array = random_seed_array is not None
        random_seed_array = self._initialize_random_seed_array(
            random_seed_array=random_seed_array,
            expected_shape=imputed_data.data_shape[0],
        )
        # The loop below is dynamic, depending on missing data availability.
        # Therefore, we need to pre-compute random seeds, we cannot rely on
        # generating them on the fly.
        subset_seeds = self._draw_random_int32(
            random_state=random_state,
            size=(
                len(datasets),
                iterations,
                len(self.variable_training_order)
            ),
        )
        self._initialize_dataset(
            imputed_data, random_state=random_state, random_seed_array=random_seed_array
        )

        iter_range = list(range(1, iterations + 1))
        for ds in datasets:

            logger.log("Dataset " + str(ds))
            dsind = datasets.index(ds)

            for iter_abs, iter_rel in iter_pairs:

                logger.log(str(iter_abs) + " ", end="")

                # Determine which model iteration to grab
                if self.save_models == 1 or iter_abs > kernel_iterations:
                    iter_model = kernel_iterations
                else:
                    iter_model = iter_abs

                for var in imputed_data.imputation_order:

                    logger.log(" | " + self._get_variable_name(var), end="")
                    log_context = {
                        "dataset": ds,
                        "variable_name": self._get_variable_name(var),
                        "iteration": iter_rel
                    }
                    nawhere = imputed_data.na_where[var]

                    predictor_variables = self.variable_schema[var]
                    mmc = self.mean_match_candidates[var]

                    # We don't need these if imputing new data and mmc == 0
                    logger.set_start_time()
                    if mmc > 0:
                        candidate_features, candidate_values = self._make_xy(
                            variable=var,
                            subset_count=self.data_subset[var],
                            return_cat=False,
                            random_seed=subset_seeds[dsind, iter_rel, self.variable_training_order.index(var)]
                        )

                        # lightgbm requires integers for label. Categories won't work.
                        if candidate_values.dtype.name == "category":
                            candidate_values = candidate_values.cat.codes
                    else:
                        candidate_features = candidate_values = None

                    # Create copy of data bachelors
                    bachelor_features = _subset_data(
                        imputed_data.working_data,
                        row_ind=imputed_data.na_where[var],
                        col_ind=predictor_variables,
                    )
                    logger.record_time("prepare_xy", **log_context)

                    # Select our model.
                    current_model = self.get_model(
                        variable=var, dataset=ds, iteration=iter_model
                    )

                    seeds = random_seed_array[nawhere] if use_seed_array else None
                    logger.set_start_time()
                    imp_values = np.array(
                        self.mean_match_function(
                            mmc=self.mean_match_candidates[var],
                            model=current_model,
                            bachelor_features=bachelor_features,
                            candidate_values=candidate_values,
                            random_state=random_state,
                            hashed_seeds=seeds,
                        )
                    )
                    logger.record_time("mean_matching", **log_context)
                    imputed_data._insert_new_data(
                        dataset=ds, variable_index=var, new_data=imp_values
                    )
                    # Refresh our seeds.
                    if use_seed_array:
                        random_seed_array[nawhere] = hash_int32(seeds)

                    imputed_data.iterations[ds, imputed_data.imputation_order.index(var)] += 1

                logger.log("\n", end="")

        imputed_data._ampute_original_data()
        if self.save_loggers:
            self.loggers.append(logger)

        return imputed_data

    def start_logging(self):
        self.save_loggers = True

    def stop_logging(self):
        self.save_loggers = False

    def save_kernel(
            self,
            filepath,
            clevel=None,
            cname=None,
            n_threads=None,
            copy_while_saving=True
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
        working_data_bytes = BytesIO()
        if kernel.original_data_class == "pd_DataFrame":
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
                    cname=cname
                ),
                f
            )

    def get_feature_importance(self, dataset, iteration=None) -> np.ndarray:
        """
        Return a matrix of feature importance. The cells
        represent the normalized feature importance of the
        columns to impute the rows. This is calculated
        internally by RandomForestRegressor/Classifier.

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
        self, dataset, normalize: bool = True, iteration: int = None, **kw_plot
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
            self._get_variable_name(int(i)) for i in np.sort(self.imputation_order)
        ]
        predictor_var_names = [
            self._get_variable_name(int(i)) for i in np.sort(self.predictor_vars)
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
