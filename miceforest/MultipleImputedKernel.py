from .ImputedDataSet import ImputedDataSet
from .MultipleImputedDataSet import MultipleImputedDataSet
from .KernelDataSet import KernelDataSet
from pandas import DataFrame
from typing import Union, List, Dict
from .utils import ensure_rng
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
    variable_schema: None or list or Dict[str, List[str]]
        If None all variables are used to impute all variables which have
        missing values.
        If list all variables are used to impute the variables in the list
        If dict the values will be used to impute the keys.
    save_all_iterations: boolean, optional(default=False)
        Save all iterations that have been imputed, or just the latest.
        Saving all iterations allows for additional plotting,
        but may take more memory

    """

    def __init__(
        self,
        data: DataFrame,
        datasets: int = 5,
        variable_schema: Union[List[str], Dict[str, List[str]]] = None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
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
                    mean_match_candidates=self.mean_match_candidates,
                    save_all_iterations=save_all_iterations,
                    save_models=save_models,
                    random_state=random_state,
                )
            )

        self.save_models = save_models
        self.iteration_time_seconds = 0

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
