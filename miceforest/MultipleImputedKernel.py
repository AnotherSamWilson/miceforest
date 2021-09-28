from .ImputedDataSet import ImputedDataSet
from .MultipleImputedDataSet import MultipleImputedDataSet
from .KernelDataSet import KernelDataSet
from typing import Union, List, Callable
from .utils import ensure_rng, MeanMatchType, VarSchemType, VarParamType
from .logger import Logger
import numpy as np
from .compat import pd_DataFrame, pd_Series


class MultipleImputedKernel(MultipleImputedDataSet):
    """
    Multiple Imputed Kernel

    Creates and stores a collection of KernelDataSet instances.
    Has methods that allow for easy access to datasets, as well
    as comparison and plotting methods.

    More details on usage can be found on the GitHub:
    https://github.com/AnotherSamWilson/miceforest

    Parameters
    ----------
    data: pandas DataFrame or np.ndarray
        The data to impute.

    datasets: int, optional(default=5)
        The number of kernel datasets to create.

    <<The parameters below are all sent to the underlying KernelDataSets>>

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
        datasets: int = 5,
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        imputation_order: Union[str, List[str], List[int]] = "ascending",
        categorical_feature: Union[str, List[str], List[int]] = "auto",
        initialization: str = "random",
        save_all_iterations: bool = True,
        save_models: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
    ):

        self.random_state = ensure_rng(random_state)

        super().__init__(
            impute_data=data,
            variable_schema=variable_schema,
            imputation_order=imputation_order,
            categorical_feature=categorical_feature,
            save_all_iterations=save_all_iterations,
        )

        # Prime with the required number of kernel datasets
        while self.dataset_count() < datasets:
            self.append(
                KernelDataSet(
                    data=data,
                    variable_schema=variable_schema,
                    mean_match_candidates=mean_match_candidates,
                    mean_match_subset=mean_match_subset,
                    mean_match_function=mean_match_function,
                    imputation_order=imputation_order,
                    categorical_feature=categorical_feature,
                    initialization=initialization,
                    save_all_iterations=save_all_iterations,
                    save_models=save_models,
                    random_state=self.random_state,
                )
            )

        self.save_models = save_models
        self.mean_match_candidates = getattr(self[0], "mean_match_candidates")
        self.mean_match_subset = getattr(self[0], "mean_match_subset")

    def __repr__(self):
        mss = {0: "None", 1: "Last Iteration", 2: "All Iterations"}
        summary_string = (
            f"""\
              Class: MultipleImputedKernel
       Models Saved: {mss[self.save_models]}\n"""
            + self._mids_info()
        )
        return summary_string

    def mice(
        self,
        iterations: int = 5,
        verbose: bool = False,
        variable_parameters: VarParamType = None,
        **kwlgb,
    ):
        """
        Calls mice() on all datasets stored in this instance.

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
            Should information about the process
            be printed?
        kw_fit:
            Additional arguments to pass to
            sklearn.RandomForestRegressor and
            sklearn.RandomForestClassifier
        """
        logger = Logger(verbose)
        for dataset in list(self.keys()):
            logger.log("Dataset " + str(dataset))
            self[dataset].mice(
                iterations=iterations,
                verbose=verbose,
                variable_parameters=variable_parameters,
                **kwlgb,
            )

    def impute_new_data(
        self,
        new_data: Union[pd_DataFrame, np.ndarray, pd_Series],
        datasets: List[int] = None,
        iterations: int = None,
        save_all_iterations: bool = True,
        verbose: bool = False,
    ) -> Union[ImputedDataSet, MultipleImputedDataSet]:
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
        datasets: list of ints
            The datasets to use to impute the new data.
            If None, all kernels are used.
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

        logger = Logger(verbose)

        if datasets is None:
            datasets = list(self.keys())
        elif isinstance(datasets, int):
            datasets = [datasets]

        if len(datasets) == 1:
            logger.log("Dataset " + str(datasets[0]))
            imputed_data_set = self[datasets[0]].impute_new_data(
                new_data=new_data,
                iterations=iterations,
                save_all_iterations=save_all_iterations,
                verbose=verbose,
            )
            return imputed_data_set
        else:
            multiple_imputed_dataset = MultipleImputedDataSet(
                impute_data=new_data,
                variable_schema=self.variable_schema.copy(),
                imputation_order=self.imputation_order.copy(),
                categorical_feature=self.categorical_mapping,
                save_all_iterations=save_all_iterations,
            )
            while len(datasets) > 0:
                ds = datasets.pop(0)
                logger.log("Dataset " + str(ds))
                multiple_imputed_dataset.append(
                    self[ds].impute_new_data(
                        new_data=new_data,
                        iterations=iterations,
                        save_all_iterations=save_all_iterations,
                        verbose=verbose,
                    )
                )
            return multiple_imputed_dataset

    def get_feature_importance(self, dataset: int = 0) -> np.ndarray:
        """
        Return an array of feature importance values.
        The values represent the scaled importance of the
        column variables in imputing the row variables.

        Parameters
        ----------
        dataset: int
            The dataset to get the feature importance of.

        Returns
        -------

        """
        return self[dataset].get_feature_importance()

    def plot_feature_importance(self, dataset: int = 0, **kw_plot):
        """
        Plot the feature importance of a specific stored dataset.

        Parameters
        ----------
        dataset: int
            The dataset to plot the feature importance of.
        kw_plot
            Other arguments passed to seaborn.heatmap()

        Returns
        -------

        """
        self[dataset].plot_feature_importance(**kw_plot)

    def tune_parameters(
        self,
        dataset,
        variables=None,
        variable_parameters=None,
        parameter_sampling_method="random",
        nfold=10,
        optimization_steps=5,
        verbose=False,
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
        dataset: int
            Which kernel dataset to use to tune parameters.

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

        return self[dataset].tune_parameters(
            variables=variables,
            variable_parameters=variable_parameters,
            parameter_sampling_method=parameter_sampling_method,
            nfold=nfold,
            optimization_steps=optimization_steps,
            verbose=verbose,
            **kwbounds,
        )
