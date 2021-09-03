from .ImputedDataSet import ImputedDataSet
from .MultipleImputedDataSet import MultipleImputedDataSet
from .KernelDataSet import KernelDataSet
from pandas import DataFrame
from typing import Union, List, Dict, Callable
from .utils import ensure_rng, MeanMatchType, VarSchemType
from .logger import Logger


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
    data: DataFrame
        A pandas DataFrame to impute.
    datasets: int, optional(default=5)
        The number of kernel datasets to create.

    The following parameters are all sent to the underlying KernelDataSets

    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all variables are used to impute the variables in the list
        If dict the values will be used to impute the keys.

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

    mean_match_function: Calable, default = None
        Must take the following parameters:
            mmc: int,
            candidate_preds: np.ndarray,
            bachelor_preds: np.ndarray,
            candidate_values: np.ndarray,
            cat_dtype: CategoricalDtype,
            random_state: np.random.RandomState,

        A default mean matching function will be used if None.

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
        data: DataFrame,
        datasets: int = 5,
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        save_all_iterations: bool = False,
        save_models: int = 1,
        random_state=None,
    ):

        random_state = ensure_rng(random_state)

        super().__init__(
            initial_dataset=KernelDataSet(
                data=data,
                variable_schema=variable_schema,
                mean_match_candidates=mean_match_candidates,
                mean_match_subset=mean_match_subset,
                mean_match_function=mean_match_function,
                save_all_iterations=save_all_iterations,
                save_models=save_models,
                random_state=random_state,
            )
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
                    save_all_iterations=save_all_iterations,
                    save_models=save_models,
                    random_state=random_state,
                )
            )

        self.save_models = save_models
        self.random_state = random_state

    def __repr__(self):
        mss = {0: "None", 1: "Last Iteration", 2: "All Iterations"}
        summary_string = (
            f"""\
              Class: MultipleImputedKernel
       Models Saved: {mss[self.save_models]}\n"""
            + self._mids_info()
        )
        return summary_string

    def mice(self, iterations: int = 5, verbose: bool = False, **kw_fit):
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
            self[dataset].mice(iterations=iterations, verbose=verbose, **kw_fit)

    def impute_new_data(
        self,
        new_data: DataFrame,
        datasets: List[int] = None,
        iterations: int = None,
        save_all_iterations: bool = False,
        verbose: bool = False,
    ) -> Union[ImputedDataSet, MultipleImputedDataSet]:
        """
        Call impute_new_data on multiple kernel kernel datasets,
        returning a MultipleImputedDataset.
        If len(datasets) == 1, an ImputedDataSet is returned.

        Parameters
        ----------
        new_data: pandas DataFrame
            The new data to impute
        datasets: None, List[int]
            The datasets and corresponding models from
            the kernel to use for imputation. If None,
            all datasets are used.
        iterations: None, int
            The iterations to run. If None, the same number
            of iterations in the kernel are used.
        save_all_iterations: bool
            Whether to save about all of the imputed values
            in every iteration, or just the last iteration.
        verbose: bool
            Print progress

        Returns ImputedDataSet or MultipleImputedDataSet
        -------
            If 1 dataset is selected, an ImputedDataSet is returned
            If more than 1 dataset is selected, a MultipleImputedDataset
            is returned.

        """
        logger = Logger(verbose)
        if datasets is None:
            datasets = list(self.keys())

        # Create an ImputedDataSet on the first dataset in the list
        logger.log("Dataset " + str(datasets[0]))
        imputed_data_set = self[datasets.pop(0)].impute_new_data(
            new_data=new_data,
            iterations=iterations,
            save_all_iterations=save_all_iterations,
            verbose=verbose,
        )
        if len(datasets) > 0:
            multiple_imputed_set = MultipleImputedDataSet(
                initial_dataset=imputed_data_set
            )
            while len(datasets) > 0:
                logger.log("Dataset " + str(datasets[0]))
                multiple_imputed_set.append(
                    self[datasets.pop(0)].impute_new_data(
                        new_data=new_data,
                        iterations=iterations,
                        save_all_iterations=save_all_iterations,
                        verbose=verbose,
                    )
                )
            return multiple_imputed_set
        else:
            return imputed_data_set

    def get_feature_importance(self, dataset: int = 0) -> DataFrame:
        """
        Return a dataframe of feature importance values.
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
