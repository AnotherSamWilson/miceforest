from .ImputedDataSet import ImputedDataSet
from .TimeLog import TimeLog
from datetime import datetime
from .utils import (
    _get_default_mmc,
    _get_default_mms,
    param_mapping,
    MeanMatchType,
    VarSchemType,
    VarParamType,
    CatFeatType,
    ensure_rng,
    stratified_continuous_folds,
    stratified_categorical_folds,
)
import numpy as np
from typing import Union, Dict, Callable, List
from .logger import Logger
from lightgbm import train, Dataset, cv
from .compat import pd_DataFrame, pd_Series
from .default_lightgbm_parameters import default_parameters, make_default_tuning_space

_TIMED_VARIABLE_EVENTS = [
    "mice",
    "model_fit",
    "model_predict",
    "mean_match",
    "make_xy",
    "tuning",
    "impute_new_data",
]
_TIMED_GLOBAL_EVENTS = ["initialization", "other"]


class KernelDataSet(ImputedDataSet):
    """
    Creates a kernel dataset. This dataset can:
        - Perform MICE on itself
        - Impute new data from models obtained from MICE.

    Parameters
    ----------
    data: np.ndarray or pandas DataFrame.
        The data to be imputed.

    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all columns in data are used to impute the variables in the list
        If dict the values will be used to impute the keys. Either can be column
        indices or names (if data is a pd.DataFrame).

    mean_match_candidates:  None or int or float or dict
        If float must be 0.0 < mmc <= 1.0. Interpreted as a percentage of available candidates
        If int must be mmc >= 0. Interpreted as the number of candidates.
        If dict, keys must be variable names, and values must follow two above rules.

        The number of mean matching candidates to use.
        Candidates are _always_ drawn from a kernel dataset, even
        when imputing new data.

        Mean matching follows the following rules based on variable type:
            Categorical:
                If mmc = 0, the class with the highest probability is chosen.
                If mmc > 0, return class based on random draw weighted by
                    class probability for each sample.
            Numeric:
                If mmc = 0, the predicted value is used
                If mmc > 0, obtain the mmc closest candidate
                    predictions and collect the associated
                    real candidate values. Choose 1 randomly.

        For more information, see:
        https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

    mean_match_subset: None or int or float or dict.
        If float must be 0.0 < mms <= 1.0. Interpreted as a percentage of available candidates
        If int must be mms >= 0. Interpreted as the number of candidates. If 0, no subsetting is done.
        If dict, keys must be variable names, and values must follow two above rules.

        The number of candidates to search in mean matching. Set
        to a lower number to speed up mean matching. Must be greater
        than mean_match_candidates, and above 0. If a float <= 1.0 is
        passed, it is interpreted as the percentage of the candidates
        to take as a subset. If an int > 1 is passed, it is interpreted
        as the count of candidates to take as a subset.

        Mean matching can take a while on larger datasets. It is recommended to carefully
        select this value for each variable if dealing with very large data.

    mean_match_function: Callable, default = None
        Must take the following parameters:
            mmc: int,
            candidate_preds: np.ndarray,
            bachelor_preds: np.ndarray,
            candidate_values: np.ndarray,
            cat_dtype: CategoricalDtype,
            random_state: np.random.RandomState,

        A default mean matching function will be used if None.

    imputation_order: str or List[str], default = "ascending"
        The order the imputations should occur in. Must be a list or a string
        - If a list is passed, must be compatible with variable_schema. The
            variables will be imputed in the order of the list.
        - If a string is passed, must be one of the following:
            - ascending: variables are imputed from least to most missing
            - descending: most to least missing
            - roman: from left to right in the dataset
            - arabic: from right to left in the dataset.

    categorical_feature: str or list or dict
        The categorical features in the dataset. This handling depends on class of impute_data:
            pandas DataFrame:
                - "auto": categorical information is inferred from any columns with
                    datatype category or object.
                - list of column names (or indices): Useful if all categorical columns
                    have already been cast to numeric encodings of some type, otherwise you
                    should just use "auto". If columns are specified in this list, they will
                    be treated as categorical, no matter their apparent form in the data.
                - Dict of mappings: For internal purposes. This is how categorical information
                    is passed from the kernel to the new imputed dataset.
            numpy ndarray:
                - "auto": no categorical information is stored.
                - list of column indices: Specified columns are treated as categorical
                - Dict of mappings: For internal purposes. This is how categorical information
                    is passed from the kernel to the new imputed dataset.

    initialization: str
        "random" - missing values will be filled in randomly from existing values.
        "empty" - lightgbm will start MICE without initial imputation

    save_all_iterations: boolean, optional(default=True)
        Save all the imputation values from all iterations, or just
        the latest. Saving all iterations allows for additional
        plotting, but may take more memory

    save_models: int
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

    random_state: None,int, or numpy.random.RandomState
        Ensures a random state throughout the process

    """

    def __init__(
        self,
        data: Union[pd_DataFrame, np.ndarray],
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        imputation_order: Union[str, List[Union[str, int]]] = "ascending",
        categorical_feature: CatFeatType = "auto",
        initialization: str = "random",
        save_all_iterations: bool = True,
        save_models: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
    ):

        s_init = datetime.now()

        super().__init__(
            impute_data=data,
            variable_schema=variable_schema,
            imputation_order=imputation_order,
            categorical_feature=categorical_feature,
            save_all_iterations=save_all_iterations,
        )

        self.initialization = initialization
        self.save_models = save_models
        self.models: Dict[str, Dict] = {var: {0: None} for var in self.imputation_order}
        self.time_log = TimeLog(
            self.column_names, _TIMED_GLOBAL_EVENTS, _TIMED_VARIABLE_EVENTS
        )

        # Format mean_match_candidates before priming datasets
        available_candidates = {
            var: (self.data_shape[0] - self.na_counts[var])
            for var in self.imputation_order
        }
        mean_match_candidates = self._format_mm(
            mean_match_candidates, available_candidates, _get_default_mmc
        )
        mean_match_subset = self._format_mm(
            mean_match_subset, available_candidates, _get_default_mms
        )

        # Ensure mmc and mms make sense:
        # mmc <= mms <= available candidates for each var
        for var in self.imputation_order:

            assert (
                mean_match_candidates[var] <= mean_match_subset[var]
            ), f"{var} mean_match_candidates > mean_match_subset"
            assert (
                mean_match_subset[var] <= available_candidates[var]
            ), f"{var} mean_match_subset > available candidates"

        self.mean_match_candidates = mean_match_candidates
        self.mean_match_subset = mean_match_subset

        # Only import sklearn if we really need to.
        if mean_match_function is None:

            from .mean_matching_functions import default_mean_match

            self.mean_match_function = default_mean_match

        else:

            self.mean_match_function = mean_match_function

        self._random_state = ensure_rng(random_state)
        self._initialize_dataset(self)
        self.time_log.add_global_time("initialization", s_init)

    def __repr__(self):
        summary_string = " " * 14 + "Class: KernelDataSet\n" + self._ids_info()
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

    def _format_mm(
        self, mm, available_candidates, defaulting_function
    ) -> Dict[str, int]:
        """
        mean_match_candidates and mean_match_subset both require similar formatting.
        The only real difference is the default value based on the number of available
        candidates, which is what the defaulting_function is for.
        """
        if mm is None:

            mm = {
                var: int(defaulting_function(available_candidates[var]))
                for var in self.imputation_order
            }

        elif isinstance(mm, (int, float)):

            mm = {
                var: self._mm_type_handling(mm, available_candidates[var])
                for var in self.imputation_order
            }

        elif isinstance(mm, dict):
            mm = self._get_variable_index(mm)
            mm = {
                var: self._mm_type_handling(mm[var], available_candidates[var])
                if var in mm.keys()
                else defaulting_function(available_candidates[var])
                for var in self.imputation_order
            }
        else:
            raise ValueError(
                "one of the mean_match parameters couldn't be interpreted."
            )

        return mm

    def _insert_new_model(self, var, model):
        """
        Inserts a new model if save_mdoels > 0.
        Deletes the prior one if save_models == 1.
        """
        current_iter = self.iteration_count(var)
        if self.save_models == 0:
            return
        else:
            self.models[var][current_iter + 1] = model
        if self.save_models == 1:
            del self.models[var][current_iter]

    def _initialize_dataset(self, dataset: ImputedDataSet):
        assert not dataset.initialized, "dataset has already been initialized"
        if self.initialization == "random":
            for var in list(dataset.variable_schema):
                ind = self.na_where[var]
                dataset.imputation_values[var][0] = self._random_state.choice(
                    np.delete(self.data[:, var], ind), size=dataset.na_counts[var]
                )
        elif self.initialization == "empty":
            for var in list(dataset.variable_schema):
                dataset.imputation_values[var][0] = np.array(np.nan).repeat(
                    dataset.na_counts[var]
                )
        else:
            raise ValueError("initialization parameter not recognized.")
        dataset.initialized = True

    def _fix_parameter_aliases(self, parameters):
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

            vsp = {var: {} for var in self.imputation_order}

        else:

            variable_parameters = self._get_variable_index(variable_parameters)
            vsp_vars = set(variable_parameters)

            assert vsp_vars.issubset(
                self.imputation_order
            ), "Some variable_parameters are not being imputed."
            vsp = {
                var: variable_parameters[var] if var in vsp_vars else {}
                for var in self.imputation_order
            }

        return vsp

    def _get_lgb_params(self, var, vsp, **kwlgb):

        seed = self._random_state.randint(1000000, size=1)[0]

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

    def _get_random_sample(self, parameters: dict):
        parameters = parameters.copy()
        for p, v in parameters.items():
            if hasattr(v, "__iter__"):
                if isinstance(v, list):
                    parameters[p] = self._random_state.choice(v)
                elif isinstance(v, tuple):
                    parameters[p] = self._random_state.uniform(v[0], v[1], size=1)[0]
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
            early_stopping_rounds=10,
            verbose_eval=False,
            return_cvbooster=True,
        )
        best_iteration = lgbcv["cvbooster"].best_iteration
        loss_metric_key = list(lgbcv)[0]
        loss = np.min(lgbcv[loss_metric_key])
        return loss, best_iteration

    def get_model(self, var: Union[int, str], iteration=None):
        """
        Return the model for a specific variable, iteration.

        Parameters
        ----------
        var: str
            The variable that was imputed
        iteration: int
            The model iteration to return. Keep in mind if:
                - save_models == 0, no models are saved
                - save_models == 1, only the last model iteration is saved
                - save_models == 2, all model iterations are saved

        Returns: RandomForestRegressor or RandomForestClassifier
            The model used to impute this specific variable, iteration.

        """

        iteration = self._default_iteration(iteration)
        var = self._get_variable_index(var)
        try:
            return self.models[var][iteration]
        except Exception:
            raise ValueError("Model was not saved for this iteration.")

    def get_raw_prediction(self, var: Union[str, int], iteration=None):
        """
        Gets the raw prediction for every row in the kernel dataset.

        Parameters
        ----------
        var: The variable for which to get predictions
        iteration: The iteration of model to get. If None,
        the latest iteration is grabbed.

        Returns
        -------
        np.ndarray

        """
        var = self._get_variable_index(var)
        x, y = self._make_xy(var, iteration=iteration, return_cat=False)
        return self.get_model(var, iteration=iteration).predict(x)

    # Models are updated here, and only here.
    def mice(
        self,
        iterations: int = 5,
        verbose: bool = False,
        variable_parameters: VarParamType = None,
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
        https://github.com/AnotherSamWilson/miceforest#The-MICE-Algorithm

        Parameters
        ----------
        iterations: int
            The number of iterations to run.
        verbose: bool
            Should information about the process be printed?
        variable_parameters: None or dict
            Model parameters can be specified by variable here. Keys should
            be variable names or indices, and values should be a dict of
            parameter which should apply to that variable only. For full
            examples, see:
            https://github.com/AnotherSamWilson/miceforest#Controlling-Tree-Growth
        kwlgb:
            Additional arguments to pass to lightgbm. Applied to all models.

        """

        logger = Logger(verbose)

        iterations_at_start = self.iteration_count()
        iter_range = range(
            iterations_at_start + 1, iterations_at_start + iterations + 1
        )
        vsp = self._format_variable_parameters(variable_parameters)

        for iteration in iter_range:
            logger.log(str(iteration) + " ", end="")
            for var in self.imputation_order:
                logger.log(" | " + self._get_column_name(var), end="")
                mice_s = datetime.now()

                x, y, cat = self._make_xy(var=var, return_cat=True)
                self.time_log.add_variable_time(var, "make_xy", mice_s)
                non_missing_ind = np.setdiff1d(
                    range(self.data_shape[0]), self.na_where[var]
                )
                candidate_features = x[non_missing_ind, :]
                candidate_target = y[non_missing_ind]

                lgbpars = self._get_lgb_params(var, vsp[var], **kwlgb)
                num_iterations = lgbpars.pop("num_iterations")
                train_pointer = Dataset(
                    data=candidate_features,
                    label=candidate_target,
                    categorical_feature=cat,
                )
                fit_s = datetime.now()
                current_model = train(
                    params=lgbpars,
                    train_set=train_pointer,
                    num_boost_round=num_iterations,
                    categorical_feature=cat,
                    verbose_eval=False,
                )
                self.time_log.add_variable_time(var, "model_fit", fit_s)
                self._insert_new_model(var=var, model=current_model)

                bachelor_features = x[self.na_where[var], :]

                meanmatch_s = datetime.now()
                imp_values = np.array(
                    self.mean_match_function(
                        mmc=self.mean_match_candidates[var],
                        mms=self.mean_match_subset[var],
                        model=current_model,
                        candidate_features=candidate_features,
                        bachelor_features=bachelor_features,
                        candidate_values=candidate_target,
                        random_state=self._random_state,
                    )
                )
                self.time_log.add_variable_time(var, "mean_match", meanmatch_s)
                assert imp_values.shape == (
                    self.na_counts[var],
                ), f"{var} mean matching returned malformed array"
                self._insert_new_data(var, imp_values)
                self.time_log.add_variable_time(var, "mice", mice_s)

            logger.log("\n", end="")

    def tune_parameters(
        self,
        variables: List[Union[str, int]] = None,
        variable_parameters: VarParamType = None,
        parameter_sampling_method: str = "random",
        nfold: int = 10,
        optimization_steps: int = 5,
        verbose: bool = False,
        **kwbounds,
    ):
        """
        Perform hyperparameter tuning on models at the current iteration.
        A few notes:
            - Underlying models will now be gradient boosted trees by default (or any
                other boosting type compatible with lightgbm.cv).
            - The parameters are tuned on the data that would currently be returned by
                complete_data(). It is usually a good idea to run at least 1 iteration
                of mice with the default parameters to get a more accurate idea of the
                real optimal parameters, since Missing At Random (MAR) data imputations
                tend to converge over time.
            - num_iterations is treated as the maximum number of boosting rounds to run
                in lightgbm.cv. It is NEVER optimized. The num_iterations that is returned
                is the best_iteration returned by lightgbm.cv. num_iterations can be passed to
                limit the boosting rounds, but the returned value will always be obtained
                from best_iteration.
            - Parameters are chosen in the following order of priority:
                1) Anything specified in variable_parameters
                2) Parameters specified globally in **kwbounds
                3) Default tuning space (miceforest.default_lightgbm_parameters.make_default_tuning_space)
                4) Default parameters (miceforest.default_lightgbm_parameters.default_parameters)
            - See examples for a detailed run-through. See
                https://github.com/AnotherSamWilson/miceforest#Tuning-Parameters
                for even more detailed examples.


        Parameters
        ----------

        variables: None or list
            - If None, default hyper-parameter spaces are selected based on kernel data, and
            all variables with missing values are tuned.
            - If list, must either be indexes or variable names corresponding to the variables
            that are to be tuned.

        variable_parameters: None or dict
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
            If 'random', parameters are randomly selected.
            Other methods will be added in future releases.

        nfold: int
            The number of folds to perform cross validation with. More folds takes longer, but
            Gives a more accurate distribution of the error metric.

        optimization_steps:
            How many steps to run the process for.

        kwbounds:
            Any additional arguments that you want to apply globally to every variable.
            For example, if you want to limit the number of iterations, you could pass
            num_iterations = x to this functions, and it would apply globally. Custom
            bounds can also be passed.

        Returns
        -------
        2 dicts: optimal_parameters, optimal_parameter_losses

            optimal_parameters - a dict of the optimal parameters found for each variable.
                This can be passed directly to the variable_parameters parameter in mice()
                for future iterations for extremely accurate imputation.

                    {variable: {parameter_name: parameter_value}}

            optimal_parameter_losses - the average out of fold cv loss obtained directly from
                lightgbm.cv() associated with the optimal parameter set.

                    {variable: loss}

        """

        s_init = datetime.now()
        logger = Logger(verbose=verbose)
        self._fix_parameter_aliases(kwbounds)

        if variables is None:
            variables = self.imputation_order
        else:
            variables = self._get_variable_index(variables)

        vsp = self._format_variable_parameters(variable_parameters)

        variable_parameter_space = {}
        for var in variables:

            default_tuning_space = make_default_tuning_space(
                len(self.categorical_mapping[var])
                if var in self.categorical_variables
                else 1,
                int((self.data.shape[0] - len(self.na_where[var])) / 3),
            )

            variable_parameter_space[var] = self._get_lgb_params(
                var=var,
                vsp={**kwbounds, **vsp[var]},
                **default_tuning_space,
            )

        optimal_parameters = {var: {} for var in variables}
        optimal_parameter_losses = {var: np.Inf for var in variables}
        self.time_log.add_global_time("other", s_init)

        if parameter_sampling_method == "random":

            for var, parameters in variable_parameter_space.items():

                s_tune = datetime.now()
                logger.log(self._get_column_name(var) + " | ", end="")

                x, y, cat = self._make_xy(var=var, return_cat=True)
                non_missing_ind = np.setdiff1d(
                    range(self.data_shape[0]), self.na_where[var]
                )
                candidate_features = x[non_missing_ind, :]
                candidate_target = y[non_missing_ind]
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
                            label=candidate_target,
                            categorical_feature=cat,
                        )
                        if is_categorical:
                            folds = stratified_categorical_folds(
                                candidate_target, nfold
                            )
                        else:
                            folds = stratified_continuous_folds(candidate_target, nfold)
                        sampling_point = self._get_random_sample(parameters=parameters)
                        try:
                            loss, best_iteration = self._get_oof_performance(
                                parameters=sampling_point,
                                folds=folds,
                                train_pointer=train_pointer,
                                categorical_feature=cat,
                            )
                        except:
                            loss, best_iteration = np.Inf, 0

                        if best_iteration > 1:
                            break
                        else:
                            non_learners += 1

                    if loss < optimal_parameter_losses[var]:
                        sampling_point["num_iterations"] = best_iteration
                        optimal_parameters[var] = sampling_point
                        optimal_parameter_losses[var] = loss

                    logger.log(" - ", end="")

                logger.log("\n", end="")
                self.time_log.add_variable_time(var, "tuning", s_tune)

            return optimal_parameters, optimal_parameter_losses

    def impute_new_data(
        self,
        new_data: Union[pd_DataFrame, np.ndarray, pd_Series],
        iterations: int = None,
        save_all_iterations: bool = True,
        verbose: bool = False,
    ) -> ImputedDataSet:
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

        Parameters
        ----------
        new_data: pandas DataFrame / Series or np.ndarray
            The new data to impute. Needs to be compatible with
            the original kernel data.
        iterations: None, int
            The iterations to run. If None, the same number
            of iterations in the kernel are used. If iterations
            is greater than the number of MICE iterations, the
            latest model will be used for any additional iterations.
        save_all_iterations: bool
            Whether to save about all of the imputed values
            in every iteration, or just the last iteration.
        verbose: bool
            Print progress

        Returns
        -------
        ImputedDataSet

        """

        s_ind = datetime.now()
        logger = Logger(verbose)
        new_data = new_data.copy()

        if self.save_models < 1:
            raise ValueError("No models were saved.")

        if isinstance(new_data, pd_DataFrame):
            assert (
                self.original_data_class == "pd_DataFrame"
            ), "Cannot impute a new pd.DataFrame unless kernel dataset was also pd.DataFrame"
            if new_data.shape[1] != self.data.shape[1]:
                raise ValueError("Columns are not the same as kernel data")
            self._enforce_pandas_types(new_data)
        elif isinstance(new_data, pd_Series):
            assert (
                self.original_data_class == "pd_DataFrame"
            ), "Cannot impute a new pd.Series unless kernel dataset was a pd.DataFrame"
            assert (
                new_data.index.tolist() == self.column_names
            ), "Series index not consistent with original column names."
            new_data = new_data.to_frame().T
            self._enforce_pandas_types(new_data)

        imputed_data_set = ImputedDataSet(
            impute_data=new_data,
            variable_schema=self.variable_schema.copy(),
            imputation_order=self.imputation_order.copy(),
            categorical_feature=self.categorical_mapping,
            save_all_iterations=save_all_iterations,
        )
        self._initialize_dataset(imputed_data_set)

        curr_iters = self.iteration_count()
        iterations = self._default_iteration(iterations)
        iter_range = range(1, iterations + 1)
        iter_vars = imputed_data_set.imputation_order
        self.time_log.add_global_time("other", s_ind)

        for iteration in iter_range:
            logger.log(str(iteration) + " ", end="")

            # Determine which model iteration to grab
            if self.save_models == 1 or iteration > curr_iters:
                itergrab = curr_iters
            else:
                itergrab = iteration

            for var in iter_vars:
                logger.log(" | " + self._get_column_name(var), end="")
                impute_new_data_s = datetime.now()

                current_model = self.get_model(var, itergrab)

                # Collect bachelor information
                x, y = imputed_data_set._make_xy(var, return_cat=False)
                bachelor_features = x[imputed_data_set.na_where[var], :]
                candidate_features, candidate_target = self._make_xy(
                    var, return_cat=False
                )

                meanmatch_s = datetime.now()
                imp_values = np.array(
                    self.mean_match_function(
                        mmc=self.mean_match_candidates[var],
                        mms=self.mean_match_subset[var],
                        model=current_model,
                        candidate_features=candidate_features,
                        bachelor_features=bachelor_features,
                        candidate_values=candidate_target,
                        random_state=self._random_state,
                    )
                )
                self.time_log.add_variable_time(var, "mean_match", meanmatch_s)
                imputed_data_set._insert_new_data(var, imp_values)
                self.time_log.add_variable_time(
                    var, "impute_new_data", impute_new_data_s
                )

            logger.log("\n", end="")

        return imputed_data_set

    def get_feature_importance(self, iteration: int = None) -> np.ndarray:
        """
        Return a matrix of feature importance. The cells
        represent the normalized feature importance of the
        columns to impute the rows. This is calculated
        internally by RandomForestRegressor/Classifier.

        Parameters
        ----------
        iteration: int
            The iteration to return the feature importance for.
            Right now, the model must be saved to return importance

        Returns
        -------
        np.ndarray of importance values. Rows are imputed variables, and
        columns are predictor variables.

        """
        # Should change this to save importance as models are updated, so
        # we can still get feature importance even if models are not saved.

        iteration = self._default_iteration(iteration)

        importance_matrix = np.full(
            shape=(len(self.imputation_order), len(self.predictor_vars)),
            fill_value=np.NaN,
        )

        for ivar in self.imputation_order:
            importance_dict = dict(
                zip(
                    self.variable_schema[ivar],
                    self.get_model(ivar, iteration).feature_importance(),
                )
            )
            for pvar in importance_dict:
                importance_matrix[
                    np.sort(self.imputation_order).tolist().index(ivar),
                    np.sort(self.predictor_vars).tolist().index(pvar),
                ] = importance_dict[pvar]

        return importance_matrix

    def plot_feature_importance(
        self, normalize: bool = True, iteration: int = None, **kw_plot
    ):
        """
        Plot the feature importance. See get_feature_importance()
        for more details.

        Parameters
        ----------
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

        importance_matrix = self.get_feature_importance(iteration=iteration)
        if normalize:
            importance_matrix = (
                importance_matrix / np.nansum(importance_matrix, 1).reshape(-1, 1)
            ).round(2)

        imputed_var_names = [
            self._get_column_name(int(i)) for i in np.sort(self.imputation_order)
        ]
        predictor_var_names = [
            self._get_column_name(int(i)) for i in np.sort(self.predictor_vars)
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
