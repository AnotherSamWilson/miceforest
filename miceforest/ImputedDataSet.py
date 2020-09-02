from .ImputationSchema import _ImputationSchema
import numpy as np
from pandas import DataFrame
from itertools import combinations
from .utils import (
    ensure_rng,
    _distinct_from_list,
    _copy_and_remove,
    _list_union
)
from typing import Optional, Union, List, Dict


class ImputedDataSet(_ImputationSchema):
    """
    Imputed Data Set

    This class should not be instantiated directly.
    Instead, use derived method MultipleImputedKernel.

    Parameters
    ----------
    data: DataFrame
        A pandas DataFrame to impute.
    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all variables are used to impute the variables in the list
        If dict the values will be used to impute the keys.
    save_all_iterations: boolean, optional(default=False)
        Save all iterations that have been imputed, or just the latest.
        Saving all iterations allows for additional plotting,
        but may take more memory
    verbose: boolean, optional(default=False)
        Print warnings and imputation progress?
    random_state: None,int, or numpy.random.RandomState
        Ensures a random state throughout the process

    Methods
    -------
    get_iterations()
        Return iterations for the entire process, or a specific
        dataset, variable.
    get_imps()
        Return imputations for specified dataset, variable, iteration.
    complete_data()
        Replace missing values with imputed values.
    get_correlations()
        Return the correlations between datasets for
        the specified variables.
    """

    def __init__(
        self,
        data: DataFrame,
        variable_schema: Union[List[str], Dict[str, List[str]]] = None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
        save_all_iterations: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
    ):

        super().__init__(
            variable_schema=variable_schema,
            mean_match_candidates=mean_match_candidates,
            validation_data=data,
        )

        self._random_state = ensure_rng(random_state)

        self.data = data
        self.save_all_iterations = save_all_iterations
        self.categorical_variables = list(
            self.data_dtypes[self.data_dtypes == "category"].keys()
        )
        self.imputation_values: Dict[str, Dict] = {var: dict() for var in self.response_vars}

        # Right now variables are filled in randomly.
        # Add options in the future.
        for var in list(self.imputation_values):
            self.imputation_values[var] = {
                0: self._random_state.choice(
                    data[var].dropna(), size=self.na_counts[var]
                )
            }

    def _ids_info(self) -> str:
        summary_string = f"""\
         Iterations: {self.get_iterations()}
  Imputed Variables: {self.n_imputed_vars}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

    def __repr__(self):
        summary_string = "              Class: ImputedDataSet\n" + self._ids_info()
        return summary_string

    def get_iterations(self, var: str = None) -> int:
        """
        Return iterations for the entire dataset, or a specific variable

        Parameters
        ----------
        dataset: None,int
            The dataset to return the iterations for. If not None,
            var must also be supplied, which returns the specific
            iterations for a dataset, variable.
            If dataset and var are None, the meta iteration count
            for the entire process is returned.
        var: None,str
            The variable to get the number of iterations for. See
            dataset parameter for specifics.

        Returns
        -------
        int
            The iterations run so far.
        """

        # If var is None, we want the meta iteration level. This must fail
        # if called inside iteration updates, which would mean certain variables
        # have different numbers of iterations.
        if var is None:
            var_iterations = {var:np.max(list(iter)) for var,iter in self.imputation_values.items()}
            distinct_iterations = _distinct_from_list(list(var_iterations.values()))
            if len(distinct_iterations) > 1:
                raise ValueError(
                    "Inconsistent state - cannot get meta iteration count."
                )
            else:
                return next(iter(distinct_iterations))
        else:
            # Extract the number of iterations so far for a specific dataset, variable
            return np.max(list(self.imputation_values[var]))

    def get_imps(self, var: str, iteration: int = None) -> np.ndarray:
        """
        Return imputations for specified dataset, variable, iteration.

        Parameters
        ----------
        var: str
            The variable to return the imputations for
        iteration: int
            The iteration to return.
            If not None, save_all_iterations must be True

        Returns
        -------
            An array of imputed values.
        """

        current_iteration = self.get_iterations(var=var)
        if iteration is None:
            iteration = current_iteration
        elif iteration != current_iteration:
            if iteration not in self.imputation_values[var]:
                raise ValueError('This iterations imputed values were not saved.')
            else:
                return self.imputation_values[var][iteration]

    # Should return all rows of data
    def _make_xy(self, var: str):

        xvars = self.variable_schema[var]
        completed_data = self.complete_data(all_vars=True)
        x = completed_data[xvars].copy()
        y = completed_data[var].copy()
        to_convert = _list_union(self.categorical_variables,xvars)
        for ctc in to_convert:
            x[ctc] = x[ctc].cat.codes
        return x, y

    def _insert_new_data(self, var: str, new_data: np.ndarray):

        current_iter = self.get_iterations(var)
        if not self.save_all_iterations:
            del self.imputation_values[var][current_iter]

        self.imputation_values[var][current_iter + 1] = new_data

    def complete_data(
        self, iteration: int = None, all_vars: bool = False
    ) -> DataFrame:
        """
        Replace missing values with imputed values.

        Parameters
        ----------
        dataset: int
            The dataset to return
        iteration: int
            The iteration to return.
            If not None, save_all_iterations must be True
        all_vars: bool
            Should all variables in the imputation schema be
            imputed, or just the ones specified to be imputed?

        Returns
        -------
        pandas DataFrame
            The completed data

        """

        imputed_dataset = self.data.copy()

        # Need to impute all variables used in variable_schema if we are running model
        # Just impute specified variables if the user wants it.
        ret_vars = self.all_vars if all_vars else self.response_vars

        for var in ret_vars:
            imputed_dataset.loc[self.na_where[var], var] = self.get_imps(
                var, iteration
            )
        return imputed_dataset

    def _cross_check_numeric(self, variables: Optional[List[str]]) -> List[str]:

        numeric_imputed_vars = _copy_and_remove(variables,self.categorical_variables)

        if variables is None:
            variables = numeric_imputed_vars
        else:
            if any([var not in numeric_imputed_vars for var in variables]):
                raise ValueError(
                    "Specified variable is not in imputed numeric variables."
                )

        return variables

    def get_correlations(
        self, variables: List[str] = None
    ) -> Dict[str, Dict[int, List[float]]]:
        """
        Return the correlations between datasets for
        the specified variables.

        Parameters
        ----------
        variables: None,str
            The variables to return the correlations for.

        Returns
        -------
        dict
            The correlations at each iteration for the specified
            variables.

        """

        if len(self.dataset_list) < 3:
            raise ValueError(
                "Not enough datasets to calculate correlations between them"
            )

        variables = self._cross_check_numeric(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        correlation_dict = {}
        if self.save_all_iterations:
            iter_range = list(range(self.get_iterations() + 1))
        else:
            # Make this iterable for code tidyness
            iter_range = [self.get_iterations()]

        for var in variables:

            # Get a dict of variables and imputations for all datasets for this iteration
            iteration_level_imputations = {
                iteration: {
                    dataset: self.get_imps(dataset, var, iteration=iteration)
                    for dataset in self.dataset_list
                }
                for iteration in iter_range
            }

            combination_correlations = {
                iteration: np.array(
                    [
                        round(np.corrcoef(impcomb)[0, 1], 3)
                        for impcomb in list(combinations(varimps.values(), 2))
                    ]
                )
                for iteration, varimps in iteration_level_imputations.items()
            }

            correlation_dict[var] = combination_correlations

        return correlation_dict

    def plot_correlations(self, variables: List[str] = None, **adj_args):
        """
        Plot the correlations between datasets.
        See get_correlations for more details.

        Parameters
        ----------
        variables: None,list
            The variables to plot.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """

        if len(self.dataset_list) < 4:
            raise ValueError("Not enough datasets to make box plot")

        variables = self._cross_check_numeric(variables)
        correlation_dict = self.get_correlations(variables=variables)

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        plots = len(correlation_dict)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = gs[v].get_rows_columns()[2], gs[v].get_rows_columns()[5]
            var = list(correlation_dict)[v]
            ax[axr, axc].boxplot(
                list(correlation_dict[var].values()),
                labels=range(len(correlation_dict[var])),
            )
            ax[axr, axc].set_title(var)
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("Correlations")
            ax[axr, axc].set_ylim([-1, 1])
        plt.subplots_adjust(**adj_args)

    def plot_imputed_distributions(
        self, variables: List[str] = None, iteration: int = None, **adj_args
    ):
        """
        Plot the imputed value distribution for all datasets.

        Parameters
        ----------
        variables: None,list
            The variables to plot.
        iteration: None,int
            The iteration to plot the distribution for.
            If None, the latest iteration is plotted.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        variables = self._cross_check_numeric(variables)

        if iteration is None:
            iteration = self.get_iterations()

        plots = len(variables)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = list(variables)[v]
            axr, axc = gs[v].get_rows_columns()[2], gs[v].get_rows_columns()[5]

            iteration_level_imputations = {
                dataset: self.get_imps(dataset, var, iteration=iteration)
                for dataset in self.dataset_list
            }
            plt.sca(ax[axr, axc])
            ax[axr, axc] = sns.distplot(
                self.data[var].dropna(),
                hist=False,
                kde=True,
                kde_kws={"linewidth": 2, "color": "red"},
            )
            for imparray in iteration_level_imputations.values():
                # Subset to the airline

                # Draw the density plot
                ax[axr, axc] = sns.distplot(
                    imparray,
                    hist=False,
                    kde=True,
                    kde_kws={"linewidth": 1, "color": "black"},
                )

        plt.subplots_adjust(**adj_args)


class MultipleImputedDataSet:
    def __init__(self,imputed_data_sets: Dict[int,ImputedDataSet] = {}):
        self.imputed_data_sets = imputed_data_sets

    def append(self,imputed_data_set: ImputedDataSet):
        curr_count = self.get_dataset_count()
        self.imputed_data_sets[curr_count] = imputed_data_set

    def remove(self,datasets: Union[int,List[int]]):
        """
        Remove an ImputedDataSet by key

        Parameters
        ----------
        datasets: int or list of int
            The dataset(s) to remove.

        Returns
        -------
            A dict with datasets removed.
            Renames the keys to be sequential.
        """
        if datasets._class__.__name__ == 'int':
            datasets = list(datasets)

        for d in datasets:
            del self.imputed_data_sets[d]

        # Rename the keys. Supporting 3.6 means we can't use zip,
        # or they'll be unordered.
        curr_keys = sorted(list(self.imputed_data_sets))
        for i in curr_keys:
            ind = curr_keys.index(i)
            self.imputed_data_sets[ind] = self.imputed_data_sets.pop(i)

    def get_dataset_count(self):
        return len(self.imputed_data_sets)

    def get_imputed_dataset(self,dataset):
        return self.imputed_data_sets[dataset]

    def get_iterations(self,dataset: int = None,var: str = None):
        if dataset is not None:
            self.imputed_data_sets[dataset].get_iterations(var=var)
        else:
            # Get iterations for all imputed data sets.
            # If the iterations differ, fail.
            ids_iterations = [
                ids.get_iterations(var=var)
                for ids in self.imputed_data_sets.values()
            ]
            unique_iterations = _distinct_from_list(ids_iterations)
            if len(unique_iterations) > 1:
                raise ValueError('Iterations are not consistent across provided datasets, var.')
            else:
                return next(iter(unique_iterations))

    def get_correlations(
        self, variables: List[str] = None
    ) -> Dict[str, Dict[int, List[float]]]:
        """
        Return the correlations between datasets for
        the specified variables.

        Parameters
        ----------
        variables: None,str
            The variables to return the correlations for.

        Returns
        -------
        dict
            The correlations at each iteration for the specified
            variables.

        """

        if self.get_dataset_count() < 3:
            raise ValueError(
                "Not enough datasets to calculate correlations between them"
            )

        variables = self._cross_check_numeric(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        correlation_dict = {}
        if self.save_all_iterations:
            iter_range = list(range(self.get_iterations() + 1))
        else:
            # Make this iterable for code tidyness
            iter_range = [self.get_iterations()]

        for var in variables:

            # Get a dict of variables and imputations for all datasets for this iteration
            iteration_level_imputations = {
                iteration: {
                    dataset: self.get_imps(dataset, var, iteration=iteration)
                    for dataset in self.dataset_list
                }
                for iteration in iter_range
            }

            combination_correlations = {
                iteration: np.array(
                    [
                        round(np.corrcoef(impcomb)[0, 1], 3)
                        for impcomb in list(combinations(varimps.values(), 2))
                    ]
                )
                for iteration, varimps in iteration_level_imputations.items()
            }

            correlation_dict[var] = combination_correlations

        return correlation_dict

    def plot_correlations(self, variables: List[str] = None, **adj_args):
        """
        Plot the correlations between datasets.
        See get_correlations for more details.

        Parameters
        ----------
        variables: None,list
            The variables to plot.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """

        if len(self.dataset_list) < 4:
            raise ValueError("Not enough datasets to make box plot")

        variables = self._cross_check_numeric(variables)
        correlation_dict = self.get_correlations(variables=variables)

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        plots = len(correlation_dict)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = gs[v].get_rows_columns()[2], gs[v].get_rows_columns()[5]
            var = list(correlation_dict)[v]
            ax[axr, axc].boxplot(
                list(correlation_dict[var].values()),
                labels=range(len(correlation_dict[var])),
            )
            ax[axr, axc].set_title(var)
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("Correlations")
            ax[axr, axc].set_ylim([-1, 1])
        plt.subplots_adjust(**adj_args)


