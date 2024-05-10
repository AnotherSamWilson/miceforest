import numpy as np
from pandas import DataFrame, MultiIndex, RangeIndex, read_parquet
from .utils import (
    get_best_int_downcast,
    hash_numpy_int_array,
)
from io import BytesIO
from itertools import combinations
from typing import Dict, List, Union, Any, Optional
from warnings import warn


class ImputedData:
    def __init__(
        self,
        impute_data: DataFrame,
        num_datasets: int = 5,
        variable_schema: Union[List[str], Dict[str, str]] = None,
        save_all_iterations_data: bool = True,
        copy_data: bool = True,
        random_seed_array: Optional[np.ndarray] = None,
    ):
        # All references to the data should be through self.
        self.working_data = impute_data.copy() if copy_data else impute_data
        self.shape = self.working_data.shape
        self.save_all_iterations_data = save_all_iterations_data

        assert isinstance(self.working_data.index, RangeIndex), (
            'Please reset the index on the dataframe'
        )

        column_names = []
        pd_dtypes_orig = {}
        for col, series in self.working_data.items():
            assert isinstance(col, str), 'column names must be strings'
            assert series.dtype.name != 'object', 'convert object dtypes to something else'
            column_names.append(col)
            pd_dtypes_orig[col] = series.dtype.name

        column_names: List[str] = [str(x) for x in self.working_data.columns]
        self.column_names = column_names
        pd_dtypes_orig = self.working_data.dtypes

        # Collect info about what data is missing.
        na_where = {}
        for col in column_names:
            nas = np.where(self.working_data[col].isnull())[0]
            if len(nas) == 0:
                best_downcast = 'uint8'
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
                    regressor
                    for regressor in self.column_names
                    if regressor != target
                ]
                for target in modeled_variables
            }
        elif isinstance(variable_schema, list):
            variable_schema = {
                target: [
                    regressor
                    for regressor in self.column_names
                    if regressor != target
                ]
                for target in variable_schema
            }
        elif isinstance(variable_schema, dict):
            # Don't alter the original dict out of scope
            variable_schema = variable_schema.copy()
            for target, regressors in variable_schema.items():
                if target in regressors:
                    raise ValueError(f'{target} being used to impute itself')
                
        self.variable_schema = variable_schema

        self.modeled_variables = list(self.variable_schema)
        self.imputed_variables = [
            col for col in self.modeled_variables
            if col in self.vars_with_any_missing
        ]

        self.using_random_seed_array = not random_seed_array is None
        if self.using_random_seed_array:
            assert isinstance(random_seed_array, np.ndarray)
            assert (
                random_seed_array.shape[0] == self.shape[0]
            ), "random_seed_array must be the same length as data."
            self.random_seed_array = hash_numpy_int_array(random_seed_array + 1)
        else:
            self.random_seed_array = None

        self.na_counts = na_counts
        self.na_where = na_where
        self.num_datasets = num_datasets
        self.initialized = False
        self.imputed_variable_count = len(self.imputed_variables)
        self.modeled_variable_count = len(self.modeled_variables)
        self.iterations = np.zeros(
            shape=(num_datasets, self.modeled_variable_count)
        ).astype(int)

        iv_multiindex = MultiIndex.from_product([[0], np.arange(num_datasets)], names=('iteration', 'dataset'))
        self.imputation_values = {
            var: DataFrame(index=na_where[var], columns=iv_multiindex).astype(pd_dtypes_orig[var])
            for var in self.imputed_variables
        }

    # Subsetting allows us to get to the imputation values:
    def __getitem__(self, tup):
        variable, iteration, dataset = tup
        return self.imputation_values[variable].loc[:, (iteration, dataset)]

    def __setitem__(self, tup, newitem):
        variable, iteration, dataset = tup
        imputation_iteration = self.iteration_count(dataset=dataset, variable=variable)

        # Don't throw this warning on initialization
        if (iteration <= imputation_iteration) and (iteration > 0):
            warn(f'Overwriting Variable: {variable} Dataset: {dataset} Iteration: iteration')

        self.imputation_values[variable].loc[:, (iteration, dataset)] = newitem

    def __delitem__(self, tup):
        variable, iteration, dataset = tup
        self.imputation_values[variable].drop([(iteration, dataset)], axis=1, inplace=True)

    def __getstate__(self):
        """
        For pickling
        """
        # Copy the entire object, minus the big stuff
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ['imputation_values']
        }.copy()

        state['imputation_values'] = {}

        for col, df in self.imputation_values.items():
            byte_stream = BytesIO()
            df.to_parquet(byte_stream)
            state['imputation_values'][col] = byte_stream

        return state
    
    def __setstate__(self, state):
        """
        For unpickling
        """
        self.__dict__ = state

        for col, bytes in self.imputation_values.items():
            self.imputation_values[col] = read_parquet(bytes)

    def __repr__(self):
        summary_string = f'\n{" " * 14}Class: ImputedData\n{self.__ids_info()}'
        return summary_string

    def __ids_info(self):
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

    def _get_nonmissing_index(self, variable):
        na_where = self.na_where[variable]
        dtype = na_where.dtype
        non_missing_ind = np.setdiff1d(
            np.arange(self.shape[0], dtype=dtype), 
            na_where,
            assume_unique=True
        )
        return non_missing_ind
    
    def _get_nonmissing_values(self, variable):
        ind = self._get_nonmissing_index(variable)
        return self.working_data.loc[ind, variable]
    
    def get_bachelor_features(self, variable):
        na_where = self.na_where[variable]
        predictors = self.variable_schema[variable]
        bachelor_features = self.working_data.loc[na_where, predictors]
        return bachelor_features

    def _ampute_original_data(self):
        """Need to put self.working_data back in its original form"""
        for variable in self.imputed_variables:
            na_where = self.na_where[variable]
            self.working_data.loc[na_where, variable] = np.nan

    def _prep_multi_plot(
        self,
        variables,
    ):
        plots = len(variables)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        return plots, plotrows, plotcols

    def iteration_count(
            self, 
            dataset: Optional[int] = None, 
            variable: Optional[str] = None
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

        ds_slice = slice(None) if dataset is None else dataset
        # Check all variables if None specified
        check_vars = self.imputed_variables if variable is None else [variable]
        assert len(check_vars) > 0, 'No variables to get iteration count for.'
        variable_dataset_iterations = {}
        for var in check_vars:
            var_ds_iter = (
                self.imputation_values[var]
                .columns
                .to_frame()
                .loc[(slice(None), ds_slice), :]
                .reset_index(drop=True)
                .groupby('dataset')
                .iteration
                .max()
            )
            assert var_ds_iter.nunique() == 1, (
                f'{var} has different iteration counts between datasets:\n'
                f'{var_ds_iter}'
            )
            variable_dataset_iterations[var] = var_ds_iter.iloc[0]

        distinct_variable_iteration_counts = set(variable_dataset_iterations.values())
        assert len(distinct_variable_iteration_counts) == 1, (
            'Variables have different iteration counts:\n'
            f'{variable_dataset_iterations}'
        )

        return distinct_variable_iteration_counts.pop()
        

    def complete_data(
        self,
        dataset: int = 0,
        iteration: Optional[int] = None,
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
            If None, returns the most up-to-date iterations,
            even if different between variables. If not none,
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
        assert set(imp_vars).issubset(set(self.imputed_variables)), (
            'Not all variables specified were imputed.'
        )

        for variable in imp_vars:
            if iteration is None:
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

    #     # Move this to .compat at some point.
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

    # def plot_imputed_distributions(
    #     self, datasets=None, variables=None, iteration=None, **adj_args
    # ):
    #     """
    #     Plot the imputed value distributions.
    #     Red lines are the distribution of original data
    #     Black lines are the distribution of the imputed values.

    #     Parameters
    #     ----------
    #     datasets: None, int, list[int]
    #     variables: None, str, int, list[str], or list[int]
    #         The variables to plot. If None, all numeric variables
    #         are plotted.
    #     iteration: None, int
    #         The iteration to plot the distribution for.
    #         If None, the latest iteration is plotted.
    #         save_all_iterations must be True if specifying
    #         an iteration.
    #     adj_args
    #         Additional arguments passed to plt.subplots_adjust()

    #     """
    #     # Move this to .compat at some point.
    #     try:
    #         import seaborn as sns
    #         import matplotlib.pyplot as plt
    #         from matplotlib import gridspec
    #     except ImportError:
    #         raise ImportError(
    #             "matplotlib and seaborn must be installed to plot distributions."
    #         )

    #     if datasets is None:
    #         datasets = list(range(self.dataset_count()))
    #     else:
    #         datasets = _ensure_iterable(datasets)
    #     if iteration is None:
    #         iteration = self.iteration_count(datasets=datasets, variables=variables)
    #     num_vars = self._get_num_vars(variables)
    #     plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
    #     gs = gridspec.GridSpec(plotrows, plotcols)
    #     fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

    #     for v in range(plots):
    #         var = num_vars[v]
    #         axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
    #         iteration_level_imputations = {
    #             ds: self[ds, var, iteration] for ds in datasets
    #         }
    #         plt.sca(ax[axr, axc])
    #         non_missing_ind = self._get_nonmissing_index(var)
    #         nonmissing_values = _subset_data(
    #             self.working_data, row_ind=non_missing_ind, col_ind=var, return_1d=True
    #         )
    #         ax[axr, axc] = sns.kdeplot(nonmissing_values, color="red", linewidth=2)
    #         for imparray in iteration_level_imputations.values():
    #             ax[axr, axc] = sns.kdeplot(
    #                 imparray, color="black", linewidth=1, warn_singular=False
    #             )
    #         ax[axr, axc].set(xlabel=self._get_var_name_from_scalar(var))

    #     plt.subplots_adjust(**adj_args)

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

    #     # Move this to .compat at some point.
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
