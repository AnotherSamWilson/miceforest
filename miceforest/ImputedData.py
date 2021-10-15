import numpy as np
from .compat import pd_DataFrame
from .utils import (
    _ensure_iterable,
    _is_int,
    _assign_col_values_without_copy,
    _slice,
    _subset_data,
)
from itertools import combinations


class ImputedData:
    """
    Imputed Data

    This class should not be instantiated directly.
    Instead, it is returned when ImputationKernel.impute_new_data() is called.

    Parameters
    ----------
    impute_data: pandas DataFrame or np.ndarray
        The data used by the kernel to impute impute_data.

    variable_schema: None or ist or dict, default=None
        If None all variables are used to impute all variables which have
        missing values.
        If list all columns in data are used to impute the variables in the list
        If dict the values will be used to impute the keys. Either can be column
        indices or names (if data is a pd.DataFrame).

    imputation_order: str, list[str], list[int], default="ascending"
        The order the imputations should occur in.
            ascending: variables are imputed from least to most missing
            descending: most to least missing
            roman: from left to right in the dataset
            arabic: from right to left in the dataset.

        If a list is provided, the variables will be imputed in that order.

    categorical_feature: str or list, default="auto"
        The categorical features in the dataset. This handling depends on class of impute_data:

            pandas DataFrame:
                - "auto": categorical information is inferred from any columns with
                    datatype category or object.
                - list of column names (or indices): Useful if all categorical columns
                    have already been cast to numeric encodings of some type, otherwise you
                    should just use "auto". Will throw an error if a list is provided AND
                    categorical dtypes exist in data. If a list is provided, values in the
                    columns must be consecutive integers starting at 0, as required by lightgbm.

            numpy ndarray:
                - "auto": no categorical information is stored.
                - list of column indices: Specified columns are treated as categorical. Column
                    values must be consecutive integers starting at 0, as required by lightgbm.

    save_all_iterations: boolean, default=True
        Save all the imputation values from all iterations, or just
        the latest. Saving all iterations allows for additional
        plotting, but may take more memory

    copy_data: boolean, default=True
        Should the dataset be referenced directly? If False, this will cause the dataset to be
        altered in place. If a copy is created, it is saved in self.working_data. There are different
        ways in which the dataset can be altered:

        1) complete_data() will fill in missing values
        2) To save space, mice() references and manipulates self.working_data directly.
            If self.working_data is a reference to the original dataset, the original
            dataset will undergo these manipulations during the mice process.
            At the end of the mice process, missing will be set back to np.NaN where
            they were originally missing.

    """

    def __init__(
        self,
        impute_data,
        datasets=5,
        variable_schema=None,
        imputation_order="ascending",
        categorical_feature="auto",
        save_all_iterations=True,
        copy_data=True,
    ):

        # All references to the data should be through self.
        self.working_data = impute_data.copy() if copy_data else impute_data

        # Collect metadata and format data
        if isinstance(self.working_data, pd_DataFrame):

            if len(self.working_data.shape) != 2 or self.working_data.shape[0] < 1:
                raise ValueError("Input data must be 2 dimensional and non empty.")

            original_data_class = "pd_DataFrame"
            column_names = self.working_data.columns.tolist()
            pd_dtypes_orig = self.working_data.dtypes
            data_shape = self.working_data.shape

            if any([x.name == "object" for x in pd_dtypes_orig]):
                raise ValueError(
                    "Please convert object columns to categorical or some numeric type."
                )

            # Assume categories are set dtypes.
            if categorical_feature == "auto":
                categorical_variables = [
                    column_names.index(var)
                    for var in pd_dtypes_orig.index
                    if pd_dtypes_orig[var].name in ["category"]
                ]

            elif isinstance(categorical_feature, list):
                if any([x.name == "category" for x in pd_dtypes_orig]):
                    raise ValueError(
                        "If categories are already encoded as such, set categorical_feature = auto"
                    )
                if isinstance(categorical_feature[0], str):
                    categorical_variables = [
                        column_names.index(var) for var in categorical_feature
                    ]

                else:
                    categorical_variables = categorical_feature.copy()

            else:
                raise ValueError("Unknown categorical_feature")

            # Collect category counts.
            category_counts = {}
            for var in categorical_variables:
                uniq = self.working_data.iloc[:, var].dropna().unique()
                category_counts[var] = len(uniq)

            # self.cast_dtype = self.working_data.dtypes
            na_where = {
                col: np.where(self.working_data.iloc[:, col].isnull())[0]
                for col in range(data_shape[1])
            }
            na_counts = {col: len(nw) for col, nw in na_where.items()}
            vars_with_any_missing = [
                int(col) for col, ind in na_where.items() if len(ind > 0)
            ]

        elif isinstance(self.working_data, np.ndarray):

            if len(self.working_data.shape) != 2 or self.working_data.shape[0] < 1:
                raise ValueError("Input data must be 2 dimensional and non empty.")

            original_data_class = "np_ndarray"

            # DATASET ALTERATION
            if (
                self.working_data.dtype != np.float32
                and self.working_data.dtype != np.float64
            ):
                self.working_data = self.working_data.astype(np.float32)

            data_shape = self.working_data.shape
            column_names = list(range(self.working_data.shape[1]))
            # self.cast_dtype = self.working_data.dtype
            na_where = {
                col: np.where(np.isnan(self.working_data[:, col]))[0]
                for col in range(data_shape[1])
            }
            na_counts = {col: len(nw) for col, nw in na_where.items()}
            vars_with_any_missing = [
                int(col) for col, ind in na_where.items() if len(ind > 0)
            ]

            if categorical_feature == "auto":
                categorical_variables = []
            elif isinstance(categorical_feature, list):
                assert (
                    max(categorical_feature) < self.working_data.shape[1]
                ), "categorical_feature not in dataset"
                categorical_variables = categorical_feature
            else:
                raise ValueError("categorical_feature not recognized")

            # Collect category counts.
            category_counts = {}
            for var in categorical_variables:
                uniq = np.unique(self.working_data[:, var])
                uniq = uniq[~np.isnan(uniq)]
                category_counts[var] = len(uniq)

        else:

            raise ValueError("impute_data not recognized.")

        # Formatting of variable_schema.
        if variable_schema is None:
            variable_schema = vars_with_any_missing
            variable_schema = {
                var: list(np.setdiff1d(range(data_shape[1]), [var]))
                for var in variable_schema
            }
        else:
            if isinstance(variable_schema, list):
                if isinstance(variable_schema[0], str):
                    variable_schema = [
                        column_names.index(var) for var in variable_schema
                    ]
                variable_schema = {
                    var: list(np.setdiff1d(range(data_shape[1]), [var]))
                    for var in variable_schema
                }

            elif isinstance(variable_schema, dict):
                if isinstance(list(variable_schema)[0], str):
                    variable_schema = {
                        column_names.index(var): [
                            column_names.index(prd) for prd in variable_schema[var]
                        ]
                        for var in variable_schema
                    }
                self_impute_attempt = [
                    var for var, prd in variable_schema.items() if var in prd
                ]
                if len(self_impute_attempt) > 0:
                    raise ValueError(
                        ",".join(self_impute_attempt)
                        + " variables cannot be used to impute itself."
                    )

            not_imputable = [
                var for var in list(variable_schema) if var not in vars_with_any_missing
            ]
            for rnm in not_imputable:
                del variable_schema[rnm]

        # Format imputation order
        if isinstance(imputation_order, list):
            # subset because response vars can be removed if imputing new data with no missing values.
            assert set(variable_schema).issubset(
                imputation_order
            ), "imputation_order does not include all variables to be imputed."
            imputation_order = self._get_variable_index(
                [i for i in imputation_order if i in list(variable_schema)]
            )
        elif isinstance(imputation_order, str):
            if imputation_order in ["ascending", "descending"]:
                imputation_order = (
                    np.argsort(list(na_counts.values()))
                    if imputation_order == "ascending"
                    else np.argsort(na_counts)[::-1]
                )
                imputation_order = [
                    int(i) for i in imputation_order if i in list(variable_schema)
                ]
            elif imputation_order == "roman":
                imputation_order = list(variable_schema)
            elif imputation_order == "arabic":
                imputation_order = list(variable_schema)
                imputation_order.reverse()
            else:
                raise ValueError("imputation_order not recognized.")

        # Get distinct predictor variables
        predictor_vars = [prd for prd in variable_schema.values()]
        self.predictor_vars = list(
            dict.fromkeys([item for sublist in predictor_vars for item in sublist])
        )

        self.column_names = column_names
        self.categorical_variables = categorical_variables
        self.category_counts = category_counts
        self.original_data_class = original_data_class
        self.save_all_iterations = save_all_iterations
        self.variable_schema = variable_schema
        self.imputation_order = imputation_order
        self.data_shape = data_shape
        self.na_counts = na_counts
        self.na_where = na_where
        self.vars_with_any_missing = vars_with_any_missing
        self.imputed_variable_count = len(imputation_order)
        self.iterations = np.zeros(
            shape=(datasets, self.imputed_variable_count)
        ).astype(int)

        # Create structure to store imputation values.
        # These will be initialized by an ImputationKernel.
        self.imputation_values = {
            ds: {var: dict() for var in list(variable_schema)} for ds in range(datasets)
        }
        self.initialized = False

        # Sanity checks
        if self.imputed_variable_count == 0:
            raise ValueError("Something went wrong. No variables to impute.")

    # Subsetting allows us to get to the imputation values:
    def __getitem__(self, tup):
        ds, var, iter = tup
        return self.imputation_values[ds][var][iter]

    def __setitem__(self, tup, newitem):
        ds, var, iter = tup
        self.imputation_values[ds][var][iter] = newitem

    def __delitem__(self, tup):
        ds, var, iter = tup
        del self.imputation_values[ds][var][iter]

    def __repr__(self):
        summary_string = " " * 14 + "Class: ImputedData\n" + self._ids_info()
        return summary_string

    def _ids_info(self):
        summary_string = f"""\
           Datasets: {self.dataset_count()}
         Iterations: {self.iteration_count()}
  Imputed Variables: {len(self.imputation_order)}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

    def dataset_count(self):
        """
        Return the number of datasets.
        Datasets are defined by how many different sets of imputation
        values we have accumulated.
        """
        return len(self.imputation_values)

    def _get_variable_name(self, ind):
        """Gets the variable name from an index."""
        return str(self.column_names[ind])

    def _get_variable_index(self, variables):
        """
        Variables can commonly be specified by their names.
        If names are passed, indexes are returned in the same structure.
        If indexes are passed, the object is returned.
        """
        if variables is None:
            indx = list(range(self.data_shape[1]))
        elif isinstance(variables, str):
            indx = self.column_names.index(variables)
        elif _is_int(variables):
            indx = variables
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                indx = [self.column_names.index(v) for v in variables]
            elif _is_int(variables[0]):
                indx = variables
            else:
                raise ValueError("Variable type not recognized")
        elif isinstance(variables, dict):
            if isinstance(list(variables)[0], str):
                var = variables.copy()
                indx = {self.column_names.index(v): var.pop(v) for v in list(var)}
            elif _is_int(list(variables)[0]):
                indx = variables
            else:
                raise ValueError("Variable type not recognized")
        else:
            raise ValueError("Variable type not recognized")
        return indx

    def _get_working_data_nonmissing_indx(self, var):
        non_missing_ind = np.setdiff1d(range(self.data_shape[0]), self.na_where[var])
        return non_missing_ind

    def _insert_new_data(self, dataset, variable_index, new_data):

        current_iter = self.iteration_count(datasets=dataset, variables=variable_index)

        # We need to insert the categories if the raw data is stored as a category.
        # Otherwise, pandas won't let us insert.
        view = _slice(self.working_data, col_slice=variable_index)
        if view.dtype.name == "category":
            new_data = np.array(view.cat.categories)[new_data]

        _assign_col_values_without_copy(
            dat=self.working_data,
            row_ind=self.na_where[variable_index],
            col_ind=variable_index,
            val=new_data,
        )
        self[dataset, variable_index, current_iter + 1] = new_data
        if not self.save_all_iterations:
            del self[dataset, variable_index, current_iter]

    def _ampute_original_data(self):
        """Need to put self.working_data back in its original form"""
        for c in self.imputation_order:
            _assign_col_values_without_copy(
                dat=self.working_data, row_ind=self.na_where[c], col_ind=c, val=np.NaN
            )

    def _get_num_vars(self, subset=None):
        """Returns the non-categorical imputed variable indexes."""

        num_vars = [
            v for v in self.imputation_order if v not in self.categorical_variables
        ]

        if subset is not None:
            subset = self._get_variable_index(subset)
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
            var = self.imputation_order
        else:
            variables = _ensure_iterable(variables)
            var = self._get_variable_index(variables)

        assert set(var).issubset(self.imputation_order)

        iter_indx = [self.imputation_order.index(v) for v in var]
        ds_uniq = np.unique(self.iterations[np.ix_(ds, iter_indx)])
        if len(ds_uniq) > 1:
            raise ValueError(
                "iterations were not consistent across provided datasets, variables."
            )

        return ds_uniq[0]

    def complete_data(self, dataset, iteration=None, inplace=False, variables=None):
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
        imp_vars = self._get_variable_index(imp_vars)
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
            non_missing_ind = self._get_working_data_nonmissing_indx(var)
            nonmissing_values = _subset_data(
                self.working_data, row_ind=non_missing_ind, col_ind=var, return_1d=True
            )
            ax[axr, axc] = sns.kdeplot(nonmissing_values, color="red", linewidth=2)
            for imparray in iteration_level_imputations.values():
                ax[axr, axc] = sns.kdeplot(imparray, color="black", linewidth=1)
            ax[axr, axc].set(xlabel=self._get_variable_name(var))

        plt.subplots_adjust(**adj_args)

    def get_correlations(self, datasets, variables):
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
        var_indx = self._get_variable_index(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        correlation_dict = {}
        if self.save_all_iterations:
            iter_range = list(range(curr_iteration + 1))
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
        var_indx = self._get_variable_index(variables)
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
            ax[axr, axc].set_title(var)
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("Correlations")
            ax[axr, axc].set_ylim([-1, 1])
        plt.subplots_adjust(**adj_args)
