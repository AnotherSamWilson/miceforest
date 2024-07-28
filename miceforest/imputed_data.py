from io import BytesIO
from itertools import combinations
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import numpy as np
from pandas import DataFrame, MultiIndex, RangeIndex, Series, concat, read_parquet

from .utils import get_best_int_downcast, hash_numpy_int_array


class ImputedData:
    def __init__(
        self,
        impute_data: DataFrame,
        # num_datasets: int = 5,
        datasets: List[int],
        variable_schema: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        save_all_iterations_data: bool = True,
        copy_data: bool = True,
        random_seed_array: Optional[np.ndarray] = None,
    ):
        # All references to the data should be through self.
        self.working_data = impute_data.copy() if copy_data else impute_data
        self.shape = self.working_data.shape
        self.save_all_iterations_data = save_all_iterations_data
        self.datasets = datasets

        assert isinstance(
            self.working_data.index, RangeIndex
        ), "Please reset the index on the dataframe"

        column_names = []
        pd_dtypes_orig = {}
        for col, series in self.working_data.items():
            assert isinstance(col, str), "column names must be strings"
            assert (
                series.dtype.name != "object"
            ), "convert object dtypes to something else"
            column_names.append(col)
            pd_dtypes_orig[col] = series.dtype.name

        self.column_names = column_names
        pd_dtypes_orig = self.working_data.dtypes

        # Collect info about what data is missing.
        na_where = {}
        for col in column_names:
            nas = np.where(self.working_data[col].isnull())[0]
            if len(nas) == 0:
                best_downcast = "uint8"
            else:
                best_downcast = get_best_int_downcast(int(nas.max()))
            na_where[col] = nas.astype(best_downcast)
        na_counts = {col: len(nw) for col, nw in na_where.items()}
        self.vars_with_any_missing = [
            col for col, count in na_counts.items() if count > 0
        ]

        # If variable_schema was passed, use that as the
        # list of variables that should have models trained.
        # Otherwise, only train models on variables that have
        # missing values.
        if variable_schema is None:
            modeled_variables = self.vars_with_any_missing.copy()
            variable_schema = {
                target: [
                    regressor for regressor in self.column_names if regressor != target
                ]
                for target in modeled_variables
            }
        elif isinstance(variable_schema, list):
            variable_schema = {
                target: [
                    regressor for regressor in self.column_names if regressor != target
                ]
                for target in variable_schema
            }
        elif isinstance(variable_schema, dict):
            # Don't alter the original dict out of scope
            variable_schema = variable_schema.copy()
            for target, regressors in variable_schema.items():
                if target in regressors:
                    raise ValueError(f"{target} being used to impute itself")

        self.variable_schema = variable_schema

        self.modeled_variables = list(self.variable_schema)
        self.imputed_variables = [
            col for col in self.modeled_variables if col in self.vars_with_any_missing
        ]

        if random_seed_array is not None:
            assert isinstance(random_seed_array, np.ndarray)
            assert (
                random_seed_array.shape[0] == self.shape[0]
            ), "random_seed_array must be the same length as data."
            # Our hashing scheme doesn't work for specifically the value 0.
            # Set any values == 0 to the value 1.
            random_seed_array = random_seed_array.copy()
            zero_value_seeds = random_seed_array == 0
            random_seed_array[zero_value_seeds] = 1
            hash_numpy_int_array(random_seed_array)
            self.random_seed_array: Optional[np.ndarray] = random_seed_array
        else:
            self.random_seed_array = None

        self.na_counts = na_counts
        self.na_where = na_where
        self.num_datasets = len(datasets)
        self.initialized = False
        self.imputed_variable_count = len(self.imputed_variables)
        self.modeled_variable_count = len(self.modeled_variables)
        # self.iterations = np.zeros(
        #     shape=(self.num_datasets, self.modeled_variable_count)
        # ).astype(int)

        # Create a multiindexed dataframe to store our imputation values
        iv_multiindex = MultiIndex.from_product(
            [[0], datasets], names=("iteration", "dataset")
        )
        self.imputation_values = {
            var: DataFrame(index=na_where[var], columns=iv_multiindex).astype(
                pd_dtypes_orig[var]
            )
            for var in self.imputed_variables
        }

        # Create an iteration counter
        self.iteration_tab = {}
        for variable in self.modeled_variables:
            for dataset in datasets:
                self.iteration_tab[variable, dataset] = 0

    # Subsetting allows us to get to the imputation values:
    def __getitem__(self, tup):
        variable, iteration, dataset = tup
        return self.imputation_values[variable].loc[:, (iteration, dataset)]

    def __setitem__(self, tup, newitem):
        variable, iteration, dataset = tup
        imputation_iteration = self.iteration_count(dataset=dataset, variable=variable)

        # Don't throw this warning on initialization
        if (iteration <= imputation_iteration) and (iteration > 0):
            warn(
                f"Overwriting Variable: {variable} Dataset: {dataset} Iteration: iteration"
            )

        self.imputation_values[variable].loc[:, (iteration, dataset)] = newitem

    def __delitem__(self, tup):
        variable, iteration, dataset = tup
        self.imputation_values[variable].drop(
            [(iteration, dataset)], axis=1, inplace=True
        )

    def __getstate__(self):
        """
        For pickling
        """
        # Copy the entire object, minus the big stuff
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["imputation_values"]
        }.copy()

        state["imputation_values"] = {}

        for col, df in self.imputation_values.items():
            byte_stream = BytesIO()
            df.to_parquet(byte_stream)
            state["imputation_values"][col] = byte_stream

        return state

    def __setstate__(self, state):
        """
        For unpickling
        """
        self.__dict__ = state

        for col, bytes in self.imputation_values.items():
            self.imputation_values[col] = read_parquet(bytes)

    def __repr__(self):
        summary_string = f'\n{" " * 14}Class: ImputedData\n{self._ids_info()}'
        return summary_string

    def _ids_info(self):
        summary_string = f"""\
            Datasets: {self.num_datasets}
          Iterations: {self.iteration_count()}
        Data Samples: {self.shape[0]}
        Data Columns: {self.shape[1]}
   Imputed Variables: {self.imputed_variable_count}
   Modeled Variables: {self.modeled_variable_count}
All Iterations Saved: {self.save_all_iterations_data}
        """
        return summary_string

    def _get_nonmissing_index(self, variable: str):
        na_where = self.na_where[variable]
        dtype = na_where.dtype
        non_missing_ind = np.setdiff1d(
            np.arange(self.shape[0], dtype=dtype), na_where, assume_unique=True
        )
        return non_missing_ind

    def _get_nonmissing_values(self, variable: str):
        ind = self._get_nonmissing_index(variable)
        return self.working_data.loc[ind, variable]

    def _ampute_original_data(self):
        """Need to put self.working_data back in its original form"""
        for variable in self.imputed_variables:
            na_where = self.na_where[variable]
            self.working_data.loc[na_where, variable] = np.nan

    def _get_hashed_seeds(self, variable: str):
        if self.random_seed_array is not None:
            na_where = self.na_where[variable]
            hashed_seeds = self.random_seed_array[na_where].copy()
            hash_numpy_int_array(self.random_seed_array, ind=na_where)
            return hashed_seeds
        else:
            return None

    def _get_bachelor_features(self, variable):
        na_where = self.na_where[variable]
        predictors = self.variable_schema[variable]
        bachelor_features = self.working_data.loc[na_where, predictors]
        return bachelor_features

    def iteration_count(
        self,
        dataset: Union[slice, int] = slice(None),
        variable: Union[slice, str] = slice(None),
    ):
        """
        Grabs the iteration count for specified variables, datasets.
        If the iteration count is not consistent across the provided
        datasets/variables, an error will be thrown. Providing None
        will use all datasets/variables.

        This is to ensure the process is in a consistent state when
        the iteration count is needed.

        Parameters
        ----------
        datasets: int or None
            The datasets to check the iteration count for.
            If None, all datasets are assumed (and assured)
            to have the same iteration count, otherwise error.
        variables: str or None
            The variable to check the iteration count for.
            If None, all variables are assumed (and assured)
            to have the same iteration count, otherwise error.

        Returns
        -------
        An integer representing the iteration count.
        """

        iteration_tab = Series(self.iteration_tab)
        iteration_tab.index.names = ["variable", "dataset"]

        iterations = np.unique(iteration_tab.loc[variable, dataset])
        if iterations.shape[0] > 1:
            raise ValueError("Multiple iteration counts found")
        else:
            return iterations[0]

    def complete_data(
        self,
        dataset: int = 0,
        iteration: int = -1,
        inplace: bool = False,
        variables: Optional[List[str]] = None,
    ):
        """
        Return dataset with missing values imputed.

        Parameters
        ----------
        dataset: int
            The dataset to complete.
        iteration: int
            Impute data with values obtained at this iteration.
            If -1, returns the most up-to-date iterations,
            even if different between variables. If not -1,
            iteration must have been saved in imputed values.
        inplace: bool
            Should the data be completed in place? If True,
            self.working_data is imputed,and nothing is returned.
            This is useful if the dataset is very large. If
            False, a copy of the data is returned, with missing
            values imputed.

        Returns
        -------
        The completed data, with values imputed for specified variables.

        """

        # Return a copy if not inplace.
        impute_data = self.working_data if inplace else self.working_data.copy()

        # Figure out which variables we need to impute.
        # Never impute variables that are not in imputed_variables.
        imp_vars = self.imputed_variables if variables is None else variables
        assert set(imp_vars).issubset(
            set(self.imputed_variables)
        ), "Not all variables specified were imputed."

        for variable in imp_vars:
            if iteration == -1:
                iteration = self.iteration_count(dataset=dataset, variable=variable)
            na_where = self.na_where[variable]
            impute_data.loc[na_where, variable] = self[variable, iteration, dataset]

        if not inplace:
            return impute_data

    # def get_means(self, datasets, variables=None):
    #     """
    #     Return a dict containing the average imputation value
    #     for specified variables at each iteration.
    #     """
    #     num_vars = self._get_num_vars(variables)

    #     # For every variable, get the correlations between every dataset combination
    #     # at each iteration
    #     curr_iteration = self.iteration_count(datasets=datasets)
    #     if self.save_all_iterations:
    #         iter_range = list(range(curr_iteration + 1))
    #     else:
    #         iter_range = [curr_iteration]
    #     mean_dict = {
    #         ds: {
    #             var: {itr: np.mean(self[ds, var, itr]) for itr in iter_range}
    #             for var in num_vars
    #         }
    #         for ds in datasets
    #     }

    #     return mean_dict

    # def plot_mean_convergence(self, datasets=None, variables=None, **adj_args):
    #     """
    #     Plots the average value of imputations over each iteration.

    #     Parameters
    #     ----------
    #     variables: None or list
    #         The variables to plot. Must be numeric.
    #     adj_args
    #         Passed to matplotlib.pyplot.subplots_adjust()

    #     """

    #     try:
    #         import matplotlib.pyplot as plt
    #         from matplotlib import gridspec
    #     except ImportError:
    #         raise ImportError("matplotlib must be installed to plot mean convergence")

    #     if self.iteration_count() < 2 or not self.save_all_iterations:
    #         raise ValueError("There is only one iteration.")

    #     if datasets is None:
    #         datasets = list(range(self.dataset_count()))
    #     else:
    #         datasets = _ensure_iterable(datasets)
    #     num_vars = self._get_num_vars(variables)
    #     mean_dict = self.get_means(datasets=datasets, variables=variables)
    #     plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
    #     gs = gridspec.GridSpec(plotrows, plotcols)
    #     fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

    #     for v in range(plots):
    #         axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
    #         var = num_vars[v]
    #         for d in mean_dict.values():
    #             ax[axr, axc].plot(list(d[var].values()), color="black")
    #         ax[axr, axc].set_title(var)
    #         ax[axr, axc].set_xlabel("Iteration")
    #         ax[axr, axc].set_ylabel("mean")
    #     plt.subplots_adjust(**adj_args)

    def plot_imputed_distributions(
        self, variables: Optional[List[str]] = None, iteration: int = -1
    ):
        """
        Plot the imputed value distributions.
        Red lines are the distribution of original data
        Black lines are the distribution of the imputed values.

        Parameters
        ----------
        datasets: None, int, list[int]
        variables: None, list[str]
            The variables to plot. If None, all numeric variables
            are plotted.
        iteration: int
            The iteration to plot the distribution for.
            If None, the latest iteration is plotted.
            save_all_iterations must be True if specifying
            an iteration.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """

        try:
            from plotnine import (
                aes,
                facet_wrap,
                geom_density,
                ggplot,
                ggtitle,
                scale_color_manual,
                theme,
                xlab,
            )
        except ImportError:
            raise ImportError("plotnine must be installed to plot distributions.")

        if iteration == -1:
            iteration = self.iteration_count()

        colors = {str(i): "black" for i in range(self.num_datasets)}
        colors["-1"] = "red"

        num_vars = self.working_data.select_dtypes("number").columns.to_list()

        if variables is None:
            variables = [var for var in self.imputed_variables if var in num_vars]
        else:
            variables = [var for var in variables if var in num_vars]

        dat = DataFrame()
        for variable in variables:

            imps = self.imputation_values[variable].loc[:, iteration].melt()
            imps["variable"] = variable
            ind = self._get_nonmissing_index(variable)
            orig = self.working_data.loc[ind, variable].rename("value").to_frame()
            orig["dataset"] = -1
            orig["variable"] = variable
            dat = concat([dat, imps, orig], axis=0)

        dat["dataset"] = dat["dataset"].astype("string")

        fig = (
            ggplot()
            + geom_density(
                data=dat, mapping=aes(x="value", group="dataset", color="dataset")
            )
            + facet_wrap("variable", scales="free")
            + scale_color_manual(values=colors)
            + ggtitle("Distribution Plots")
            + xlab("")
            + theme(legend_position="none")
        )

        return fig

    # def get_correlations(
    #     self, datasets: List[int], variables: Union[List[int], List[str]]
    # ):
    #     """
    #     Return the correlations between datasets for
    #     the specified variables.

    #     Parameters
    #     ----------
    #     variables: list[str], list[int]
    #         The variables to return the correlations for.

    #     Returns
    #     -------
    #     dict
    #         The correlations at each iteration for the specified
    #         variables.

    #     """

    #     if self.dataset_count() < 3:
    #         raise ValueError(
    #             "Not enough datasets to calculate correlations between them"
    #         )
    #     curr_iteration = self.iteration_count()
    #     var_indx = self._get_var_ind_from_list(variables)

    #     # For every variable, get the correlations between every dataset combination
    #     # at each iteration
    #     correlation_dict = {}
    #     if self.save_all_iterations:
    #         iter_range = list(range(1, curr_iteration + 1))
    #     else:
    #         # Make this iterable for code tidyness
    #         iter_range = [curr_iteration]

    #     for var in var_indx:
    #         # Get a dict of variables and imputations for all datasets for this iteration
    #         iteration_level_imputations = {
    #             iteration: {ds: self[ds, var, iteration] for ds in datasets}
    #             for iteration in iter_range
    #         }

    #         combination_correlations = {
    #             iteration: [
    #                 round(np.corrcoef(impcomb)[0, 1], 3)
    #                 for impcomb in list(combinations(varimps.values(), 2))
    #             ]
    #             for iteration, varimps in iteration_level_imputations.items()
    #         }

    #         correlation_dict[var] = combination_correlations

    #     return correlation_dict

    # def plot_correlations(self, datasets=None, variables=None, **adj_args):
    #     """
    #     Plot the correlations between datasets.
    #     See get_correlations() for more details.

    #     Parameters
    #     ----------
    #     datasets: None or list[int]
    #         The datasets to plot.
    #     variables: None,list
    #         The variables to plot.
    #     adj_args
    #         Additional arguments passed to plt.subplots_adjust()

    #     """

    #     try:
    #         import matplotlib.pyplot as plt
    #         from matplotlib import gridspec
    #     except ImportError:
    #         raise ImportError("matplotlib must be installed to plot importance")

    #     if self.dataset_count() < 4:
    #         raise ValueError("Not enough datasets to make box plot")
    #     if datasets is None:
    #         datasets = list(range(self.dataset_count()))
    #     else:
    #         datasets = _ensure_iterable(datasets)
    #     var_indx = self._get_var_ind_from_list(variables)
    #     num_vars = self._get_num_vars(var_indx)
    #     plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
    #     correlation_dict = self.get_correlations(datasets=datasets, variables=num_vars)
    #     gs = gridspec.GridSpec(plotrows, plotcols)
    #     fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

    #     for v in range(plots):
    #         axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
    #         var = list(correlation_dict)[v]
    #         ax[axr, axc].boxplot(
    #             list(correlation_dict[var].values()),
    #             labels=range(len(correlation_dict[var])),
    #         )
    #         ax[axr, axc].set_title(self._get_var_name_from_scalar(var))
    #         ax[axr, axc].set_xlabel("Iteration")
    #         ax[axr, axc].set_ylabel("Correlations")
    #         ax[axr, axc].set_ylim([-1, 1])
    #     plt.subplots_adjust(**adj_args)
