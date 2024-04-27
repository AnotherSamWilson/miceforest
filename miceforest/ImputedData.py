import numpy as np
from .compat import pd_DataFrame
from pandas import DataFrame, MultiIndex
from .utils import (
    get_best_int_downcast,
    _t_dat,
    _t_var_list,
    _t_var_dict,
    _ensure_iterable,
)
from itertools import combinations
from typing import Dict, List, Union, Any, Optional
from warnings import warn


class ImputedPandasDataFrame:
    def __init__(
        self,
        impute_data: pd_DataFrame,
        num_datasets: int = 5,
        variable_schema: Union[List[str], Dict[str, str]] = None,
        imputation_order: str = "ascending",
        copy_data: bool = True,
    ):
        # All references to the data should be through self.
        self.working_data = impute_data.copy() if copy_data else impute_data
        data_shape = self.working_data.shape

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
            best_downcast = get_best_int_downcast(nas.max())
            na_where[col] = nas.astype(best_downcast)
        na_counts = {col: len(nw) for col, nw in na_where.items()}
        vars_with_any_missing = [
            col for col, ind in na_where.items() if len(ind > 0)
        ]

        # If variable_schema was passed, use that as the 
        # list of variables that should have models trained.
        # Otherwise, only train models on variables that have
        # missing values, unless train_nonmissing, in which
        # case we build imputation models for all variables.
        if variable_schema is None:
            modeled_variables = vars_with_any_missing
            variable_schema = {
                target: [
                    regressor
                    for regressor in column_names
                    if regressor != target
                ]
                for target in modeled_variables
            }
        else:
            if isinstance(variable_schema, list):
                variable_schema = {
                    target: [
                        regressor
                        for regressor in column_names
                        if regressor != target
                    ]
                    for target in variable_schema
                }
            elif isinstance(variable_schema, dict):
                for target, regressors in variable_schema.items():
                    if target in regressors:
                        raise ValueError(f'{target} being used to impute itself')

        # variable schema at this point should only 
        # contain the variables that are to have models trained.
        modeled_variables = list(variable_schema)
        imputed_variables = [
            col for col in modeled_variables
            if col in vars_with_any_missing
        ]
        modeled_but_not_imputed_variables = [
            col for col in modeled_variables
            if col not in imputed_variables
        ]

        # Model Training Order:
            # Variables with missing data are always trained 
            # first, according to imputation_order. Afterwards, 
            # variables with no missing values have models trained.
        if imputation_order in ["ascending", "descending"]:
            na_counts_of_imputed_variables = {
                key: value 
                for key, value in self.na_counts.items()
                if key in imputed_variables
            }
            self.imputation_order = list(sorted(
                na_counts_of_imputed_variables.items(), 
                key=lambda item: item[1]
            ))
            if imputation_order == "decending":
                self.imputation_order.reverse()
        elif imputation_order == "roman":
            self.imputation_order = imputed_variables.copy()
        elif imputation_order == "arabic":
            self.imputation_order = imputed_variables.copy()
            self.imputation_order.reverse()
        else:
            raise ValueError("imputation_order not recognized.")
        
        model_training_order = self.imputation_order + modeled_but_not_imputed_variables

        self.variable_schema = variable_schema
        self.model_training_order = model_training_order
        self.data_shape = data_shape
        self.na_counts = na_counts
        self.na_where = na_where
        self.vars_with_any_missing = vars_with_any_missing
        self.imputed_variable_count = len(self.imputation_order)
        self.modeled_variable_count = len(self.model_training_order)
        self.iterations = np.zeros(
            shape=(num_datasets, self.modeled_variable_count)
        ).astype(int)

        iv_multiindex = MultiIndex.from_arrays([np.arange(num_datasets), [0]], names=('dataset', 'iteration'))
        self.imputation_values = {
            var: DataFrame(index=na_where[var], columns=iv_multiindex).astype(pd_dtypes_orig[var])
            for var in self.imputation_order
        }

    # Subsetting allows us to get to the imputation values:
    def __getitem__(self, tup):
        var, ds, iter = tup
        return self.imputation_values[var].loc[:, (ds, iter)]

    def __setitem__(self, tup, newitem):
        var, ds, iter = tup
        self.imputation_values[var].loc[:, (ds, iter)] = newitem

    def __delitem__(self, tup):
        var, ds, iter = tup
        del self.imputation_values[var][ds, iter]

    def __repr__(self):
        summary_string = f'\n{" " * 14}Class: ImputedData\n{self._ids_info()}'
        return summary_string

    def _ids_info(self):
        summary_string = f"""\
           Datasets: {self.dataset_count()}
         Iterations: {self.iteration_count()}
       Data Samples: {self.data_shape[0]}
       Data Columns: {self.data_shape[1]}
  Imputed Variables: {self.imputed_variable_count}
  Modeled Variables: {self.modeled_variable_count}
        """
        return summary_string

    def _get_nonmissing_index(self, column):
        na_where = self.na_where[column]
        dtype = na_where.dtype
        non_missing_ind = np.setdiff1d(np.arange(self.data_shape[0], dtype=dtype), na_where)
        return non_missing_ind
    
    def _get_nonmissing_values(self, column):
        ind = self._get_nonmissing_index(column)
        return self.working_data.loc[ind, column]

    def _add_imputed_values(self, dataset, variable_index, new_data):
        current_iter = self.iteration_count(datasets=dataset, variables=variable_index)

    def _ampute_original_data(self):
        """Need to put self.working_data back in its original form"""
        for c in self.imputation_order:
            _assign_col_values_without_copy(
                dat=self.working_data,
                row_ind=self.na_where[c],
                col_ind=c,
                val=np.array([np.nan]),
            )

    def _get_numeric_columns(
            self,
            imputed: bool = True,
            modeled: bool = True,
        ):
        """Returns the non-categorical imputed variable indexes."""

        num_vars = [
            v for v in self.imputation_order if v not in self.categorical_variables
        ]

        if subset is not None:
            subset = self._get_var_ind_from_list(subset)
            num_vars = [v for v in num_vars if v in subset]

        return num_vars

    def _prep_multi_plot(
        self,
        variables,
    ):
        plots = len(variables)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        return plots, plotrows, plotcols

    def iteration_count(self, datasets=None, variables=None):
        """
        Grabs the iteration count for specified variables, datasets.
        If the iteration count is not consistent across the provided
        datasets/variables, an error will be thrown. Providing None
        will use all datasets/variables.

        This is to ensure the process is in a consistent state when
        the iteration count is needed.

        Parameters
        ----------
        datasets: int or list[int]
            The datasets to check the iteration count for.
        variables: int, str, list[int] or list[str]:
            The variables to check the iteration count for.
            Variables can be specified by their names or indexes.

        Returns
        -------
        An integer representing the iteration count.
        """

        ds = (
            list(range(self.dataset_count()))
            if datasets is None
            else _ensure_iterable(datasets)
        )

        if variables is None:
            var = self.variable_training_order
        else:
            variables = _ensure_iterable(variables)
            var = self._get_var_ind_from_list(variables)

        assert set(var).issubset(self.variable_training_order)

        iter_indx = [self.variable_training_order.index(v) for v in var]
        ds_uniq = np.unique(self.iterations[np.ix_(ds, iter_indx)])
        if len(ds_uniq) == 0:
            return -1
        if len(ds_uniq) > 1:
            raise ValueError(
                "iterations were not consistent across provided datasets, variables."
            )

        return ds_uniq[0]

    def complete_data(
        self,
        dataset: int = 0,
        iteration: Optional[int] = None,
        inplace: bool = False,
        variables: Optional[_t_var_list] = None,
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
        # Never impute variables that are not in imputation_order.
        imp_vars = self.imputation_order if variables is None else variables
        imp_vars = self._get_var_ind_from_list(imp_vars)
        imp_vars = [v for v in imp_vars if v in self.imputation_order]

        for var in imp_vars:
            if iteration is None:
                iteration = self.iteration_count(datasets=dataset, variables=var)
            _assign_col_values_without_copy(
                dat=impute_data,
                row_ind=self.na_where[var],
                col_ind=var,
                val=self[dataset, var, iteration],
            )

        if not inplace:
            return impute_data

    def get_means(self, datasets, variables=None):
        """
        Return a dict containing the average imputation value
        for specified variables at each iteration.
        """
        num_vars = self._get_num_vars(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        curr_iteration = self.iteration_count(datasets=datasets)
        if self.save_all_iterations:
            iter_range = list(range(curr_iteration + 1))
        else:
            iter_range = [curr_iteration]
        mean_dict = {
            ds: {
                var: {itr: np.mean(self[ds, var, itr]) for itr in iter_range}
                for var in num_vars
            }
            for ds in datasets
        }

        return mean_dict

    def plot_mean_convergence(self, datasets=None, variables=None, **adj_args):
        """
        Plots the average value of imputations over each iteration.

        Parameters
        ----------
        variables: None or list
            The variables to plot. Must be numeric.
        adj_args
            Passed to matplotlib.pyplot.subplots_adjust()

        """

        # Move this to .compat at some point.
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise ImportError("matplotlib must be installed to plot mean convergence")

        if self.iteration_count() < 2 or not self.save_all_iterations:
            raise ValueError("There is only one iteration.")

        if datasets is None:
            datasets = list(range(self.dataset_count()))
        else:
            datasets = _ensure_iterable(datasets)
        num_vars = self._get_num_vars(variables)
        mean_dict = self.get_means(datasets=datasets, variables=variables)
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
        self, datasets=None, variables=None, iteration=None, **adj_args
    ):
        """
        Plot the imputed value distributions.
        Red lines are the distribution of original data
        Black lines are the distribution of the imputed values.

        Parameters
        ----------
        datasets: None, int, list[int]
        variables: None, str, int, list[str], or list[int]
            The variables to plot. If None, all numeric variables
            are plotted.
        iteration: None, int
            The iteration to plot the distribution for.
            If None, the latest iteration is plotted.
            save_all_iterations must be True if specifying
            an iteration.
        adj_args
            Additional arguments passed to plt.subplots_adjust()

        """
        # Move this to .compat at some point.
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn must be installed to plot distributions."
            )

        if datasets is None:
            datasets = list(range(self.dataset_count()))
        else:
            datasets = _ensure_iterable(datasets)
        if iteration is None:
            iteration = self.iteration_count(datasets=datasets, variables=variables)
        num_vars = self._get_num_vars(variables)
        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = num_vars[v]
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            iteration_level_imputations = {
                ds: self[ds, var, iteration] for ds in datasets
            }
            plt.sca(ax[axr, axc])
            non_missing_ind = self._get_nonmissing_index(var)
            nonmissing_values = _subset_data(
                self.working_data, row_ind=non_missing_ind, col_ind=var, return_1d=True
            )
            ax[axr, axc] = sns.kdeplot(nonmissing_values, color="red", linewidth=2)
            for imparray in iteration_level_imputations.values():
                ax[axr, axc] = sns.kdeplot(
                    imparray, color="black", linewidth=1, warn_singular=False
                )
            ax[axr, axc].set(xlabel=self._get_var_name_from_scalar(var))

        plt.subplots_adjust(**adj_args)

    def get_correlations(
        self, datasets: List[int], variables: Union[List[int], List[str]]
    ):
        """
        Return the correlations between datasets for
        the specified variables.

        Parameters
        ----------
        variables: list[str], list[int]
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
        var_indx = self._get_var_ind_from_list(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        correlation_dict = {}
        if self.save_all_iterations:
            iter_range = list(range(1, curr_iteration + 1))
        else:
            # Make this iterable for code tidyness
            iter_range = [curr_iteration]

        for var in var_indx:
            # Get a dict of variables and imputations for all datasets for this iteration
            iteration_level_imputations = {
                iteration: {ds: self[ds, var, iteration] for ds in datasets}
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

    def plot_correlations(self, datasets=None, variables=None, **adj_args):
        """
        Plot the correlations between datasets.
        See get_correlations() for more details.

        Parameters
        ----------
        datasets: None or list[int]
            The datasets to plot.
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
        if datasets is None:
            datasets = list(range(self.dataset_count()))
        else:
            datasets = _ensure_iterable(datasets)
        var_indx = self._get_var_ind_from_list(variables)
        num_vars = self._get_num_vars(var_indx)
        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        correlation_dict = self.get_correlations(datasets=datasets, variables=num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            var = list(correlation_dict)[v]
            ax[axr, axc].boxplot(
                list(correlation_dict[var].values()),
                labels=range(len(correlation_dict[var])),
            )
            ax[axr, axc].set_title(self._get_var_name_from_scalar(var))
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("Correlations")
            ax[axr, axc].set_ylim([-1, 1])
        plt.subplots_adjust(**adj_args)
