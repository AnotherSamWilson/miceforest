import numpy as np
from .utils import VarSchemType, _setequal, _get_missing_stats
from typing import Optional, Union, List, Dict
from .compat import pd_DataFrame, pd_Series


class ImputedDataSet:
    """
    Imputed Data Set

    This class should not be instantiated directly.
    Instead, use derived method MultipleImputedKernel.

    Parameters
    ----------
    impute_data: pandas DataFrame or np.ndarray
        The data used by the kernel to impute impute_data.

    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all columns in data are used to impute the variables in the list
        If dict the values will be used to impute the keys. Either can be column
        indices or names (if data is a pd.DataFrame).

    imputation_order: str or List[str], default = "ascending"
        The order the imputations should occur in.
            ascending: variables are imputed from least to most missing
            descending: most to least missing
            roman: from left to right in the dataset
            arabic: from right to left in the dataset.

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

    save_all_iterations: boolean, optional(default=True)
        Save all the imputation values from all iterations, or just
        the latest. Saving all iterations allows for additional
        plotting, but may take more memory

    """

    def __init__(
        self,
        impute_data: Union[pd_DataFrame, pd_Series, np.ndarray],
        variable_schema: VarSchemType = None,
        imputation_order: Union[str, List[Union[str, int]]] = "ascending",
        categorical_feature: Union[str, List[str], List[int], Dict[int, dict]] = "auto",
        save_all_iterations: bool = True,
    ):

        # Collect metadata and format data
        if isinstance(impute_data, pd_DataFrame):
            if len(impute_data.shape) != 2 or impute_data.shape[0] < 1:
                raise ValueError("Input data must be 2 dimensional and non empty.")

            impute_data = impute_data.copy()
            pd_dtypes = impute_data.dtypes

            # List so we can use .index()
            column_names = impute_data.columns.tolist()
            original_data_class = "pd_DataFrame"

            # If we need to gather info about categorical variables:
            if not isinstance(categorical_feature, dict):

                convert_objects = [
                    var for var in column_names if pd_dtypes[var].name == "object"
                ]
                impute_data[convert_objects] = impute_data[convert_objects].astype(
                    "category"
                )

                # Assume categories are set dtypes.
                if categorical_feature == "auto":
                    categorical_variables = [
                        column_names.index(var)
                        for var in pd_dtypes.index
                        if pd_dtypes[var].name in ["category", "object"]
                    ]

                # Assume categories aren't encoded as such.
                # Convert to categories
                elif isinstance(categorical_feature, list):
                    if isinstance(categorical_feature[0], str):
                        categorical_variables = [
                            column_names.index(var) for var in categorical_feature
                        ]
                    else:
                        categorical_variables = categorical_feature.copy()

                    # Convert to categorical columns.
                    impute_data.iloc[:, categorical_variables] = impute_data.iloc[
                        :, categorical_variables
                    ].astype("category")

                else:
                    raise ValueError("Unknown categorical_feature")

                # All categorical columns should be encoded as such.
                categorical_mapping = {
                    var: dict(enumerate(impute_data.iloc[:, var].cat.categories))
                    for var in categorical_variables
                }
                category_counts = {
                    var: len(categorical_mapping[var]) for var in categorical_variables
                }

            else:
                categorical_variables = list(categorical_feature)
                categorical_mapping = categorical_feature.copy()
                category_counts = {
                    var: len(categorical_feature[var]) for var in categorical_feature
                }

            # Assume new data to be imputed still has its categorical dtypes.
            self.cast_dtype = impute_data.dtypes
            impute_data.iloc[:, categorical_variables] = (
                impute_data.iloc[:, categorical_variables]
                .apply(lambda x: x.cat.codes)
                .replace({-1: np.nan})
            )
            impute_data = impute_data.values
            na_where, data_shape, na_counts, vars_with_any_missing = _get_missing_stats(
                impute_data
            )
            if impute_data.dtype != np.float32 and impute_data.dtype != np.float64:
                impute_data = impute_data.astype(np.float32)
            self.data = impute_data

        elif isinstance(impute_data, np.ndarray):
            if len(impute_data.shape) != 2 or impute_data.shape[0] < 1:
                raise ValueError("Input data must be 2 dimensional and non empty.")

            column_names = list(range(impute_data.shape[1]))
            original_data_class = "np_ndarray"
            if impute_data.dtype != np.float32 and impute_data.dtype != np.float64:
                impute_data = impute_data.astype(np.float32)
            self.cast_dtype = impute_data.dtype
            na_where, data_shape, na_counts, vars_with_any_missing = _get_missing_stats(
                impute_data
            )

            if not isinstance(categorical_feature, dict):
                if categorical_feature == "auto":
                    categorical_variables = []
                elif isinstance(categorical_feature, list):
                    assert (
                        max(categorical_feature) < impute_data.shape[1]
                    ), "categorical_feature not in dataset"
                    categorical_variables = categorical_feature
                else:
                    raise ValueError("categorical_feature not recognized")
                categorical_mapping = {
                    var: dict(
                        enumerate(
                            np.sort(
                                np.unique(np.delete(impute_data[:, var], na_where[var]))
                            )
                        )
                    )
                    for var in categorical_variables
                }
                category_counts = {
                    var: len(categorical_mapping[var]) for var in categorical_variables
                }
            else:
                categorical_variables = list(categorical_feature)
                categorical_mapping = categorical_feature.copy()
                category_counts = {
                    var: len(categorical_feature[var]) for var in categorical_feature
                }

            self.data = impute_data

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
                    var: list(np.setdiff1d(range(data_shape[1]),[var]))
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
                    np.argsort(na_counts)
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

        if len(imputation_order) == 0:
            raise ValueError("Something went wrong. No variables to impute.")

        # Get distinct predictor variables
        predictor_vars = [prd for prd in variable_schema.values()]
        self.predictor_vars = list(
            dict.fromkeys([item for sublist in predictor_vars for item in sublist])
        )

        self.column_names = column_names
        self.categorical_variables = categorical_variables
        self.categorical_mapping = categorical_mapping
        self.category_counts = category_counts
        self.original_data_class = original_data_class
        self.save_all_iterations = save_all_iterations
        self.initialized = False
        self.imputation_values: Dict[str, Dict] = {
            var: dict() for var in list(variable_schema)
        }
        self.variable_schema = variable_schema
        self.imputation_order = imputation_order
        self.data_shape = data_shape
        self.na_counts = na_counts
        self.na_where = na_where
        self.vars_with_any_missing = vars_with_any_missing

    # Subsetting allows us to get to the imputation values:
    def __getitem__(self, tup):
        var, iteration = tup
        return self.imputation_values[var][iteration]

    def __setitem__(self, tup, newitem):
        var, iteration = tup
        self.imputation_values[var][iteration] = newitem

    def __delitem__(self, tup):
        var, iteration = tup
        del self.imputation_values[var][iteration]

    def __repr__(self):
        summary_string = " " * 14 + "Class: ImputedDataSet\n" + self._ids_info()
        return summary_string

    def _ids_info(self) -> str:
        summary_string = f"""\
         Iterations: {self.iteration_count()}
  Imputed Variables: {len(self.imputation_order)}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

    def _get_column_name(self, ind: int) -> str:
        return str(self.column_names[ind])

    def _check_appendable(self, imputed_dataset, fail=True):
        """
        Checks if two imputation schemas are similar enough
        """
        checks = {
            "variable_schema": (
                self.variable_schema == imputed_dataset.variable_schema
            ),
            "imputation_order": _setequal(
                self.imputation_order, imputed_dataset.imputation_order
            ),
            "na_where": all(
                [
                    np.array_equal(self.na_where[key], value)
                    for key, value in imputed_dataset.na_where.items()
                ]
            ),
            "data": np.array_equal(self.data, imputed_dataset.data, equal_nan=True),
            "save_all_iterations": (
                self.save_all_iterations == imputed_dataset.save_all_iterations
            ),
        }
        failed_checks = [key for key, value in checks.items() if not value]
        if len(failed_checks) > 0:
            if fail:
                raise ValueError(
                    "Inconsistency in schemas in regards to " + ",".join(failed_checks)
                )
            else:
                return False
        else:
            return True

    def _get_variable_index(self, variables) -> List[int]:
        """
        Variables can commonly be specified by their names
        This finds their index.
        """
        if isinstance(variables, str):
            variables = self.column_names.index(variables)
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                variables = [self.column_names.index(v) for v in variables]
            elif isinstance(variables[0], int):
                pass
            else:
                raise ValueError("Variable type not recognized")
        elif isinstance(variables, dict):
            if isinstance(list(variables)[0], str):
                var = variables.copy()
                variables = {self.column_names.index(v): var.pop(v) for v in list(var)}
            elif isinstance(list(variables)[0], int):
                pass
            else:
                raise ValueError("Variable type not recognized")

        return variables

    def _default_iteration(self, iteration: Optional[int], **kwargs) -> int:
        """
        If iteration is not specified it is assumed to
        be the last iteration run in many cases.
        """
        if iteration is None:
            return self.iteration_count(**kwargs)
        else:
            return iteration

    def _make_xy(self, var: str, iteration: int = None, return_cat: bool = False):
        """
        Make the predictor and response set used to train the model.
        Must be defined in ImputedDataSet because this method is called
        directly in KernelDataSet.impute_new_data()

        If iteration is None, it returns the most up-to-date imputations
        for each variable.
        """
        xvars = self.variable_schema[var]
        completed_data = self.complete_data(iteration=iteration, cast="raw")
        x = completed_data[:, xvars]
        y = completed_data[:, var]
        if return_cat:
            cat = [
                xvars.index(var) for var in self.categorical_variables if var in xvars
            ]
            return x, y, cat
        else:
            return x, y

    def _insert_new_data(self, var: str, new_data: np.ndarray):
        current_iter = self.iteration_count(var)
        if not self.save_all_iterations:
            del self[var, current_iter]
        self[var, current_iter + 1] = new_data

    def _enforce_pandas_types(self, data: Union[pd_DataFrame, pd_Series]):
        # Cast to original datatypes if needed.
        cast_vars = -(self.cast_dtype == data.dtypes)
        for var in [var for var, cast in cast_vars.items() if cast]:
            data[var] = data[var].astype(self.cast_dtype[var])

    def _get_num_vars(self, subset: List[Union[str, int]] = None):

        num_vars = [
            v for v in self.imputation_order if v not in self.categorical_variables
        ]

        if subset is not None:
            subset = self._get_variable_index(subset)
            num_vars = [v for v in num_vars if v in subset]

        return num_vars

    def _prep_multi_plot(
        self,
        variables: List[str],
    ):
        plots = len(variables)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        return plots, plotrows, plotcols

    def iteration_count(self, var: str = None) -> int:
        """
        Return iterations for the entire dataset, or a specific variable

        Parameters
        ----------
        var: None,str
            If None, the meta iteration is returned.

        Returns
        -------
        int
            The iterations run so far.
        """

        if not self.initialized:
            return 0

        # If var is None, we want the meta iteration level. This must fail
        # if called inside iteration updates, which would mean certain variables
        # have different numbers of iterations.
        if var is None:
            var_iterations = [
                np.max(list(itr))
                for var, itr in self.imputation_values.items()
                if var in self.imputation_order
            ]
            distinct_iterations = np.unique(var_iterations)
            if len(distinct_iterations) > 1:
                raise ValueError(
                    "Inconsistent state - cannot get meta iteration count."
                )
            else:
                return next(iter(distinct_iterations))
        else:
            # Extract the number of iterations so far for a specific dataset, variable
            return np.max(list(self.imputation_values[var]))

    def complete_data(
        self, iteration: int = None, cast: str = "original"
    ) -> Union[pd_DataFrame, np.ndarray]:  # type:ignore
        """
        Replace missing values with imputed values.

        Parameters
        ----------
        iteration: int
            The iteration to return.complete_data
            If None, returns the most up-to-date iterations,
            even if different between variables. If not none,
            iteration must have been saved in imputed values.
        cast: str
            If "original", the data is cast to the original
            data class and types that were provided. If "raw",
            a numpy array is returned. "raw" is used internally
            for efficiency.

        Returns
        -------
        The completed data, with values imputed for specified variables.

        """
        imputed_data = self.data.copy()

        for var in self.imputation_order:
            itrn = self._default_iteration(iteration=iteration, var=var)
            imputed_data[self.na_where[var], var] = self[var, itrn]

        if cast == "raw":
            return imputed_data

        elif cast == "original":
            if self.original_data_class == "pd_DataFrame":
                imputed_data = pd_DataFrame(imputed_data)
                imputed_data.columns = self.column_names
                for var, mapping in self.categorical_mapping.items():
                    imputed_data.iloc[:, var] = imputed_data.iloc[:, var].map(mapping)
                self._enforce_pandas_types(imputed_data)
                return imputed_data
            elif self.original_data_class == "np_ndarray":
                return imputed_data.astype(self.cast_dtype)
            else:
                raise ValueError("Unknown original class")

    def get_means(self, variables: List[str] = None):
        """
        Return a dict containing the average imputation value
        for specified variables at each iteration.
        """
        num_vars = self._get_num_vars(variables)

        # For every variable, get the correlations between every dataset combination
        # at each iteration
        curr_iteration = self.iteration_count()
        if self.save_all_iterations:

            iter_range = list(range(curr_iteration + 1))

        else:

            # Make this iterable for code tidyness
            iter_range = [curr_iteration]

        mean_dict = {
            var: {itr: np.mean(self[var, itr]) for itr in iter_range}
            for var in num_vars
        }

        return mean_dict

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

        # Move this to .compat at some point.
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise ImportError("matplotlib must be installed to plot mean convergence")

        if self.iteration_count() < 2 or not self.save_all_iterations:
            raise ValueError("There is only one iteration.")

        num_vars = self._get_num_vars(variables)
        mean_dict = self.get_means(variables=num_vars)

        plots = len(mean_dict)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            var = list(mean_dict)[v]
            ax[axr, axc].plot(list(mean_dict[var].values()))
            ax[axr, axc].set_title(var)
            ax[axr, axc].set_xlabel("Iteration")
            ax[axr, axc].set_ylabel("mean")
        plt.subplots_adjust(**adj_args)

    def plot_imputed_distributions(
        self, variables: List[str] = None, iteration: int = None, **adj_args
    ):
        """
        Plot the imputed value distributions.
        Red lines are the distribution of original data
        Black lines are the distribution of the imputed values.

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
        # Move this to .compat at some point.
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn must be installed to plot distributions."
            )

        iteration = self._default_iteration(iteration)
        num_vars = self._get_num_vars(variables)

        plots, plotrows, plotcols = self._prep_multi_plot(num_vars)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = num_vars[v]
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            plt.sca(ax[axr, axc])
            dat = np.delete(self.data[:, var], self.na_where[var])
            ax[axr, axc] = sns.kdeplot(dat, color="red", linewidth=2)
            ax[axr, axc] = sns.kdeplot(self[var, iteration], color="black", linewidth=1)
            ax[axr, axc].set(xlabel=self._get_column_name(var))

        plt.subplots_adjust(**adj_args)
