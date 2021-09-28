from .ImputedDataSet import ImputedDataSet
import numpy as np
from itertools import combinations
from .utils import VarSchemType, CatFeatType
from typing import Union, List, Dict
from .compat import pd_DataFrame, pd_Series


class MultipleImputedDataSet(ImputedDataSet):
    """
    A collection of ImputedDataSets with similar schemas.
    Includes methods allowing for easy access and comparisons.
    Can be treated as a subscriptable list of ImputedDataSets.
    Stored datasets can be accessed by key in the range of:
    range(# Datasets).

    This class should not be instantiated by the user,
    but may be returned by a MultipleImputedKernel.impute_new_data()

    Parameters
    ----------
    initial_dataset: ImputedDataSet

    """

    def __init__(
        self,
        impute_data: Union[pd_DataFrame, pd_Series, np.ndarray],
        variable_schema: VarSchemType = None,
        imputation_order: Union[str, List[Union[str, int]]] = "ascending",
        categorical_feature: CatFeatType = "auto",
        save_all_iterations: bool = True,
    ):

        self.imputed_data_sets: Dict[int, ImputedDataSet] = {}

        super().__init__(
            impute_data,
            variable_schema,
            imputation_order,
            categorical_feature,
            save_all_iterations,
        )

    def __getitem__(self, key):
        return self.imputed_data_sets[key]

    def __setitem__(self, key, newitem):
        self.imputed_data_sets[key] = newitem

    def __delitem__(self, key):
        del self.imputed_data_sets[key]

    def __len__(self):
        return len(self.imputed_data_sets)

    def items(self):
        return self.imputed_data_sets.items()

    def values(self):
        return self.imputed_data_sets.values()

    def keys(self):
        return self.imputed_data_sets.keys()

    def __repr__(self):
        summary_string = (
            f"""\
              Class: MultipleImputedDataSet\n"""
            + self._mids_info()
        )
        return summary_string

    def _mids_info(self) -> str:
        summary_string = (
            " " * 11 + f"Datasets: {self.dataset_count()}" + "\n" + self._ids_info()
        )
        return summary_string

    def append(self, imputed_data_set: ImputedDataSet):
        """
        Appends an ImputedDataSet

        Parameters
        ----------
        imputed_data_set: ImputedDataSet
            The dataset to add

        """
        self._check_appendable(imputed_data_set)
        if isinstance(imputed_data_set, ImputedDataSet):
            self[self.dataset_count()] = imputed_data_set
        elif isinstance(imputed_data_set, MultipleImputedDataSet):
            for ds in range(imputed_data_set.dataset_count()):
                self[self.dataset_count()] = imputed_data_set[ds]

    def remove(self, datasets: Union[int, List[int]]):
        """
        Remove an ImputedDataSet by key. Renames keys
        in remaining datasets to be sequential.

        Parameters
        ----------
        datasets: int or list of int
            The dataset(s) to remove.

        """
        if isinstance(datasets, int):
            datasets = [datasets]

        for dataset in datasets:
            del self[dataset]

        # Rename the keys. Supporting 3.6 means we can't use zip,
        # or they'll be unordered.
        curr_keys = list(self.keys())
        for key in curr_keys:
            ind = curr_keys.index(key)
            self.imputed_data_sets[ind] = self.imputed_data_sets.pop(key)

    def dataset_count(self) -> int:
        """
        Returns the number of datasets being stored
        """
        return len(self)

    def get_correlations(
        self, variables: List[str]
    ) -> Dict[str, Dict[int, List[float]]]:
        """
        Return the correlations between datasets for
        the specified variables.

        Parameters
        ----------
        variables: None, List[str]
            The variables to return the correlations for.

        Returns
        -------
        dict
            The correlations at each iteration for the specified
            variables.

        """

        if self.dataset_count() < 3:
            raise ValueError(
                "Not enough datasets to calculate correlations between them"
            )
        curr_iteration = self.iteration_count()

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        correlation_dict = {}
        if self.save_all_iterations:
            iter_range = list(range(curr_iteration + 1))
        else:
            # Make this iterable for code tidyness
            iter_range = [curr_iteration]

        for var in variables:
            # Get a dict of variables and imputations for all datasets for this iteration
            iteration_level_imputations = {
                iteration: {
                    key: dataset[var, iteration] for key, dataset in self.items()
                }
                for iteration in iter_range
            }

            combination_correlations = {
                iteration: [
                    round(np.corrcoef(impcomb)[0, 1], 3)
                    for impcomb in list(combinations(varimps.values(), 2))
                ]
                for iteration, varimps in iteration_level_imputations.items()
            }

            correlation_dict[var] = combination_correlations

        return correlation_dict

    # Helper methods that reach into self.imputed_data_sets
    def iteration_count(self, dataset: int = None, var: str = None):  # type: ignore
        """
        Return iterations for the entire dataset, or a specific variable

        Parameters
        ----------
        dataset: int
            The dataset to return the iterations for. If None,
            the iteration for all datasets is returned. Will fail
            if iterations are not consistent between datasets.
        var: None,str
            If None, the iteration of all variables is returned.
            Will fail if iterations are not consistent.

        Returns
        -------
        int
            The iterations run so far.
        """

        if dataset is not None:
            self[dataset].iteration_count(var=var)
        else:

            # Return 0 if this dataset contains no imputed data sets.
            if len(self) == 0:
                return 0

            ids_iterations = np.unique(
                [ids.iteration_count(var=var) for key, ids in self.items()]
            )
            if len(ids_iterations) > 1:
                raise ValueError("Iterations are not consistent.")
            else:
                return next(iter(ids_iterations))

    def complete_data(
        self, dataset: int, iteration: int = None, cast: str = "original"
    ) -> Union[pd_DataFrame, np.ndarray]:  # type: ignore
        """
        Calls complete_data() from the specified stored dataset. See
        ImputedDataSet.complete_data().

        Parameters
        ----------
        dataset: int
            The dataset to return
        iteration:
            Iteration to return. If None, the latest iteration is
            returned. Iteration must have been saved if iteration
            is not None.
        all_vars: bool
            Should all variables used in the process be imputed,
            or just the ones specified as response variables?

        Returns
        -------
        The completed data.
        """
        compdat = self[dataset].complete_data(iteration=iteration, cast=cast)
        return compdat

    def plot_mean_convergence(self, variables: List[str] = None, **adj_args):
        """
        Plots the average value of imputations over each iteration.

        Parameters
        ----------
        variables: List[str]
            The variables to plot. Must be numeric.
        adj_args
            Passed to matplotlib.pyplot.subplots_adjust()

        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        if self.iteration_count() < 2:
            raise ValueError("There is only one iteration.")

        num_vars = self._get_num_vars(variables)

        mean_dict = {key: ds.get_means(variables=num_vars) for key, ds in self.items()}
        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            var = num_vars[v]
            for d in mean_dict.values():
                ax[axr, axc].plot(list(d[var].values()), color="black")
            ax[axr, axc].set_title(var)
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("mean")
        plt.subplots_adjust(**adj_args)

    def plot_imputed_distributions(
        self, variables: List[str] = None, iteration: int = None, **adj_args
    ):
        """
        Plot the imputed value distribution for all datasets.
        Red lines are the distribution of original data.
        Black lines are the distribution of the imputed values
        for each dataset.

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

        iteration = self._default_iteration(iteration)
        num_vars = self._get_num_vars(variables)

        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = num_vars[v]
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            iteration_level_imputations = {
                key: dataset[var, iteration] for key, dataset in self.items()
            }
            plt.sca(ax[axr, axc])
            dat = np.delete(self.data[:, var], self.na_where[var])
            ax[axr, axc] = sns.kdeplot(dat, color="red", linewidth=2)
            for imparray in iteration_level_imputations.values():
                ax[axr, axc] = sns.kdeplot(imparray, color="black", linewidth=1)
            ax[axr, axc].set(xlabel=self._get_column_name(var))

        plt.subplots_adjust(**adj_args)

    def plot_correlations(self, variables: List[str] = None, **adj_args):
        """
        Plot the correlations between datasets.
        See get_correlations() for more details.

        Parameters
        ----------
        variables: None,list
            The variables to plot.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """

        # Move this to .compat at some point.
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise ImportError("matplotlib must be installed to plot importance")

        if self.dataset_count() < 4:
            raise ValueError("Not enough datasets to make box plot")

        num_vars = self._get_num_vars(variables)
        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        correlation_dict = self.get_correlations(variables=num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
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
