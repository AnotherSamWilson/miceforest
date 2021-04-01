from .ImputedDataSet import ImputedDataSet
from .ImputationSchema import _ImputationSchema
import numpy as np
from pandas import DataFrame
from itertools import combinations
from .utils import _var_comparison
from typing import Union, List, Dict


class MultipleImputedDataSet(_ImputationSchema):
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

    def __init__(self, initial_dataset: ImputedDataSet):

        # Inherit schema and other attributes from initial_dataset
        super().__init__(
            validation_data=initial_dataset.data,
            variable_schema=initial_dataset.variable_schema,
            mean_match_candidates=initial_dataset.mean_match_candidates,
        )
        self.data = getattr(initial_dataset, "data")
        self.save_all_iterations = getattr(initial_dataset, "save_all_iterations")
        self.categorical_variables = getattr(initial_dataset, "categorical_variables")
        self._varfilter = getattr(initial_dataset, "_varfilter")
        self._prep_multi_plot = getattr(initial_dataset, "_prep_multi_plot")
        self._default_iteration = getattr(initial_dataset, "_default_iteration")

        self.imputed_data_sets = {0: initial_dataset}

    def __getitem__(self, key):
        return self.imputed_data_sets[key]

    def __setitem__(self, key, newitem):
        self.imputed_data_sets[key] = newitem

    def __delitem__(self, key):
        del self.imputed_data_sets[key]

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
        summary_string = f"""\
           Datasets: {self.dataset_count()}
         Iterations: {self.iteration_count()}
  Imputed Variables: {self.n_imputed_vars}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

    def _ensure_dataset_fidelity(self, new_set: ImputedDataSet):
        """
        To be consistent with the original, an imputed dataset must
        have the same:
            1) schema
            2) data
            3) number of iterations
            4) save_all_iterations

        Datasets can be updated internally, but cannot be
        added to the dict unless they are consistent.
        """

        assert self.equal_schemas(new_set)
        assert self.data.equals(new_set.data)
        assert new_set.iteration_count() == self.iteration_count()
        assert new_set.save_all_iterations == self.save_all_iterations

    def append(self, imputed_data_set: ImputedDataSet):
        """
        Appends an ImputedDataSet

        Parameters
        ----------
        imputed_data_set: ImputedDataSet
            The dataset to add

        """
        self._ensure_dataset_fidelity(imputed_data_set)
        curr_count = self.dataset_count()
        self[curr_count] = imputed_data_set

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
        return len(self.imputed_data_sets)

    def _get_all_vars(self) -> List[str]:
        all_vars = np.unique([ids.all_vars for key, ids in self.items()])
        return all_vars

    def _get_cat_vars(self, response=True, predictor=False) -> List[str]:
        cat_vars = np.unique([ids.categorical_variables for key, ids in self.items()])
        cat_vars = self._varfilter(cat_vars, response, predictor)
        return cat_vars

    def _get_num_vars(self, response=True, predictor=False) -> List[str]:
        all_vars = self._get_all_vars()
        cat_vars = self._get_cat_vars()
        num_vars = [i for i in all_vars if i not in cat_vars]
        num_vars = self._varfilter(num_vars, response, predictor)
        return num_vars

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
    def iteration_count(self, dataset: int = None, var: str = None):
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
            # Get iterations for all imputed data sets.
            ids_iterations = np.unique(
                [ids.iteration_count(var=var) for key, ids in self.items()]
            )
            if len(ids_iterations) > 1:
                raise ValueError("Iterations are not consistent.")
            else:
                return next(iter(ids_iterations))

    def complete_data(
        self, dataset: int, iteration: int = None, all_vars: bool = False
    ) -> DataFrame:
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
        pandas DataFrame
            The completed data
        """
        compdat = self[dataset].complete_data(iteration=iteration, all_vars=all_vars)
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

        num_vars = self._get_num_vars()
        variables = _var_comparison(variables, num_vars)

        mean_dict = {key: ds.get_means(variables=variables) for key, ds in self.items()}
        plots, plotrows, plotcols = self._prep_multi_plot(variables)
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
        num_vars = self._get_num_vars()
        variables = _var_comparison(variables, num_vars)

        plots, plotrows, plotcols = self._prep_multi_plot(variables)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = variables[v]
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            iteration_level_imputations = {
                key: dataset[var, iteration] for key, dataset in self.items()
            }
            plt.sca(ax[axr, axc])
            ax[axr, axc] = sns.kdeplot(
                self.data[var].dropna(), color="red", linewidth=2
            )
            for imparray in iteration_level_imputations.values():
                ax[axr, axc] = sns.kdeplot(imparray, color="black", linewidth=1)

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
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        if self.dataset_count() < 4:
            raise ValueError("Not enough datasets to make box plot")

        num_vars = self._get_num_vars()
        variables = _var_comparison(variables, num_vars)
        plots, plotrows, plotcols = self._prep_multi_plot(variables)
        correlation_dict = self.get_correlations(variables=variables)
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
