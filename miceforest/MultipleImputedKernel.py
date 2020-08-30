from sklearn.neighbors import NearestNeighbors
import numpy as np
from pandas import DataFrame
from itertools import combinations
from datetime import datetime
from .utils import (
    ensure_rng,
    _default_rf_classifier,
    _default_rf_regressor,
    _distinct_from_list,
)
from .ImputationSchema import ImputationSchema
from typing import Optional, Union, Any, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ImputedDataSet:
    """
    Imputed Data Set

    This class should not be instantiated directly.
    Instead, use derived method MultipleImputedKernel.

    Parameters
    ----------
    data: DataFrame
        A pandas DataFrame to impute.
    datasets: int or list, optional(default=5)
        If int, the number of datasets to create. If list, the returned
        imputation_values dict will have keys as this list.
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
        datasets: Union[int, List[int]] = 5,
        variable_schema: Union[List[str], Dict[str, List[str]]] = None,
        save_all_iterations: bool = True,
        verbose: bool = False,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:

        self._random_state = ensure_rng(random_state)
        assert isinstance(data, DataFrame)

        data_dtypes = data.dtypes
        if any(data.dtypes == "object"):
            raise ValueError("Object columns detected - please convert to category.")

        # Need to be able to specify specific datasets. If a constant
        # was passed, make it into a range. If a list was passed, the user
        # wants to interact with those datasets specifically.
        if isinstance(datasets, int):
            dataset_list = list(range(datasets))
        else:
            dataset_list = datasets
        self.dataset_list = dataset_list

        self.data = data
        self.data_shape = data.shape
        self.save_all_iterations = save_all_iterations
        self.categorical_features = list(data_dtypes[data_dtypes == "category"].keys())

        # Find a good home for these
        na_where = data.isnull()
        self.na_where = na_where
        na_counts = na_where.sum()
        self.na_counts = na_counts

        imputation_schema = ImputationSchema(
            variable_schema=variable_schema, validation_data=data, verbose=verbose
        )
        self.imputation_schema = imputation_schema

        imputation_values: Dict[int, Dict] = {i: {} for i in dataset_list}
        for ds in list(imputation_values):
            # all_vars are primed at iteration = with random sampling,
            # since predictors cannot be missing in RandomForestClassifier()
            for var in self.imputation_schema.all_vars:
                imputation_values[ds][var] = {
                    0: self._random_state.choice(
                        data[var].dropna(), size=na_counts[var]
                    )
                }
        self.imputation_values = imputation_values

    def _ids_info(self) -> str:
        summary_string = f"""\
           Datasets: {len(self.dataset_list)}
         Iterations: {self.get_iterations()}
  Imputed Variables: {self.imputation_schema.n_imputed_vars}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

    def __repr__(self):
        summary_string = "              Class: ImputedDataSet\n" + self._ids_info()
        return summary_string

    def get_iterations(self, dataset: int = None, var: str = None) -> int:
        """
        Return iterations for the entire process, or a specific
        dataset, variable.

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

        # If dataset and var are None, we want the meta iteration level. This must fail
        # if called inside iteration updates, which would mean certain dataset/iterations
        # have different numbers of iterations.
        if dataset is None and var is None:
            dataset_iterations = [
                [
                    np.max(list(iterd))
                    for vark, iterd in vard.items()
                    if vark in self.imputation_schema.response_vars
                ]
                for key, vard in self.imputation_values.items()
            ]
            distinct_iterations = _distinct_from_list(
                list(np.array(dataset_iterations).flat)
            )
            if len(distinct_iterations) > 1:
                raise ValueError(
                    "Inconsistent state - cannot get meta iteration count."
                )
            else:
                return next(iter(distinct_iterations))
        elif dataset is not None and var is not None:
            # Extract the number of iterations so far for a specific dataset, variable
            return np.max(list(self.imputation_values[dataset][var]))
        else:
            raise ValueError("Provide both or neither datasets,var parameters")

    def get_imps(self, dataset: int, var: str, iteration: int = None) -> np.ndarray:
        """
        Return imputations for specified dataset, variable, iteration.

        Parameters
        ----------
        dataset: int
            The dataset to return
        var: str
            The variable to return the imputations for
        iteration: int
            The iteration to return.
            If not None, save_all_iterations must be True

        Returns
        -------
            An array of imputed values.
        """

        # Return the imputation values for a specific iteration and variable.
        # If no iteration is specified, return the latest.
        # If iteration is specified, but it has not been saved, then throw error
        current_iteration = self.get_iterations(dataset=dataset, var=var)
        if iteration is None:
            iteration = current_iteration
        if iteration != current_iteration and not self.save_all_iterations:
            raise ValueError(
                "This iteration was not saved because save_all_iterations == False"
            )
        if iteration is None or not self.save_all_iterations:
            iteration = self.get_iterations(dataset, var)
        return self.imputation_values[dataset][var][iteration]

    # Should return all rows of data
    def _make_xy(self, dataset: int, var: str):

        xvars = self.imputation_schema.get_var_pred_list(var)
        completed_data = self.complete_data(dataset=dataset, all_vars=True)
        x = completed_data[xvars].copy()
        y = completed_data[var].copy()
        to_convert = self.imputation_schema.get_var_cat_preds(var)
        for ctc in to_convert:
            x[ctc] = x[ctc].cat.codes
        return x, y

    def _insert_new_data(self, dataset: int, var: str, new_data: np.ndarray):

        current_iter = self.get_iterations(dataset, var)
        if self.save_all_iterations:
            self.imputation_values[dataset][var][current_iter + 1] = new_data
        else:
            del self.imputation_values[dataset][var][current_iter]
            self.imputation_values[dataset][var][current_iter + 1] = new_data

    def complete_data(
        self, dataset: int = 0, iteration: int = None, all_vars: bool = False
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
        if all_vars:
            ret_vars = self.imputation_schema.all_vars
        else:
            ret_vars = self.imputation_schema.response_vars

        for var in ret_vars:
            imputed_dataset.loc[self.na_where[var], var] = self.get_imps(
                dataset, var, iteration
            )
        return imputed_dataset

    def _cross_check_numeric(self, variables: Optional[List[str]]) -> List[str]:

        numeric_imputed_vars = list(
            set(self.imputation_schema.response_vars) - set(self.categorical_features)
        )

        if variables is None:
            variables = numeric_imputed_vars
            variables.sort()
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


# MultipleImputedKernel inherits from ImputedDataSet because:
# MultipleImputedKernel needs all the functionality of ImputedDataSet + some
# A ImputedDataSet will be returned if impute_new_data is called in MultipleImputedKernel
class MultipleImputedKernel(ImputedDataSet):
    """
    Multiple Imputed Kernel

    Creates a multiple imputed kernel, which is an ImputedDataSet
    with additional methods. Can perform mice, as well as impute
    new data.

    Parameters
    ----------
    data: DataFrame
        A pandas DataFrame to impute.
    datasets: int or list, optional(default=5)
        If int, the number of datasets to create. If list, the returned
        imputation_values dict will have keys as this list.
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

    Methods
    -------
    get_iterations()
        Return the iterations that have been performed so far.
    get_imps()
        Return the imputations of a specific dataset, variable, iteration.
    complete_data()
        Return the completed data, with the original missing values replaced
        by the modeled imputation values.
    mice()
        Runs multiple imputation iterations on the kernel ImputedDataSet.
    impute_new_data()
        Uses models obtained in process of imputing kernel
        to impute a new dataset. Returns instance of ImputedDataSet
    get_feature_importance()
        Return the feature importance returned by random forest

    """

    def __init__(
        self,
        data,
        datasets=5,
        variable_schema=None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
        save_all_iterations=False,
        verbose=False,
        random_state=None,
    ):
        super().__init__(
            data=data,
            datasets=datasets,
            variable_schema=variable_schema,
            save_all_iterations=save_all_iterations,
            verbose=verbose,
            random_state=random_state,
        )

        self.iteration_time_seconds = -1

        # If mean_match_candidates is None, use max(5, 0.1% of the available candidates)
        if mean_match_candidates is None:
            mean_match_candidates = {
                key: (max([5, int((self.data_shape[0] - self.na_counts[key]) * 0.001)]))
                for key in self.imputation_schema.response_vars
            }
            mmc_inadequate = [
                var
                for var, mmc in mean_match_candidates.items()
                if (mmc >= (self.data_shape[0] - self.na_counts[var]))
            ]
            if len(mmc_inadequate) > 0:
                raise ValueError(
                    "Custom mean_match_candidates dict required due to lack of candidates."
                )

        elif isinstance(mean_match_candidates, int):
            mean_match_candidates = {
                key: mean_match_candidates
                for key in self.imputation_schema.response_vars
            }
        elif isinstance(mean_match_candidates, dict):
            if not set(mean_match_candidates) == set(variable_schema):
                raise ValueError(
                    "mean_match_candidates not consistent with variable_schema. "
                    + "Do all variables in variable_schema have missing values?."
                )

        # Make sure mean_match_candidates can be pulled when it is time.
        for var in self.imputation_schema.response_vars:
            candidate_count = self.data_shape[0] - self.na_counts[var]
            if mean_match_candidates[var] > candidate_count:
                raise ValueError(
                    "mean_match_candidates is higher than available candidates"
                    + "for the variable <"
                    + var
                    + ">"
                )
            elif (mean_match_candidates[var] / candidate_count > 0.1) & verbose:
                print(
                    "WARNING: mean_match_candidates is a very high percentage of the"
                    + "available candidates in "
                    + var
                )
        self.mean_match_candidates = mean_match_candidates

        # We save the models in the kernel
        self.models: Dict[int, Dict[str, Any]] = {
            i: {var: None for var in self.imputation_schema.response_vars}
            for i in self.dataset_list
        }
        self.models_saved = False

    def __repr__(self):
        summary_string = (
            f"""\
              Class: MultipleImputedKernel
       Models Saved: {self.models_saved}\n"""
            + self._ids_info()
        )
        return summary_string

    # Either returns value (if mean_matching_candidates == 0) or the result from mean matching
    # Candidate data is always drawn from self.data.
    def _calculate_imputation_values(
        self,
        model: Union["RandomForestClassifier", "RandomForestRegressor"],
        var: str,
        bachelor_features: List[str],
        candidate_features: List[str],
        candidate_values: np.ndarray,
    ):

        mmc = self.mean_match_candidates[var]
        if mmc == 0:
            bachelor_preds = model.predict(bachelor_features)
            return bachelor_preds
        else:
            if var in self.imputation_schema.categorical_variables:
                # Select category according to their probability for each sample
                bachelor_preds = model.predict_proba(bachelor_features)
                return [
                    self._random_state.choice(model.classes_, p=p, size=1)[0]
                    for p in bachelor_preds
                ]
            else:
                # Collect the candidates, and the predictions for the candidates and bachelors
                bachelor_preds = np.array(model.predict(bachelor_features))
                candidate_preds = np.array(model.predict(candidate_features))
                candidate_values = np.array(candidate_values)

                # Determine the nearest neighbors of the bachelor predictions in the candidate predictions
                knn = NearestNeighbors(n_neighbors=mmc, algorithm="ball_tree")
                knn.fit(candidate_preds.reshape(-1, 1))
                knn_indices = knn.kneighbors(
                    bachelor_preds.reshape(-1, 1), return_distance=False
                )
                index_choice = [self._random_state.choice(i) for i in knn_indices]

                # Use the real candidate values
                return candidate_values[index_choice]

    # This function will _always_ be performed on self.kernel_mids.
    # Models are updated here, and only here.
    def mice(
        self,
        iterations: int = 5,
        save_models: bool = True,
        verbose: bool = False,
        **kw_fit,
    ):
        """Perform mice on kernel dataset.

        Parameters
        ----------
        iterations: int
            The number of iterations to run.
        save_models: bool
            Should the random forests be saved?
            Set to true to use additional methods.
        verbose: bool
            Should information about the process
            be printed?
        kw_fit:
            Additional arguments to pass to
            sklearn.RandomForestRegressor and
            sklearn.RandomForestClassifier

        """

        self.models_saved = save_models

        # This seems goofy. Maybe a bad idea to have initial imputation as iteration 0?
        iterations_at_start = self.get_iterations()
        iter_range = range(
            iterations_at_start + 1, iterations_at_start + iterations + 1
        )
        iter_vars = self.imputation_schema.response_vars

        iteration_start = datetime.now()
        # Classic triple nested for loop.
        for dataset in self.dataset_list:
            if verbose:
                print("Dataset " + str(dataset))
            for iteration in iter_range:
                if verbose:
                    print("Iteration " + str(iteration), end="")
                for var in iter_vars:
                    if verbose:
                        if var == iter_vars[-1]:
                            endcap = "\n"
                        else:
                            endcap = ""
                        print(" | " + var, end=endcap)

                    x, y = self._make_xy(dataset, var)

                    if var in self.categorical_features:
                        current_model = _default_rf_classifier(
                            random_state=self._random_state, **kw_fit
                        )
                    else:
                        current_model = _default_rf_regressor(
                            random_state=self._random_state, **kw_fit
                        )

                    current_model.fit(X=x, y=y)

                    imp_values = self._calculate_imputation_values(
                        model=current_model,
                        var=var,
                        bachelor_features=x[self.na_where[var]],
                        candidate_features=x[self.na_where[var] == False],
                        candidate_values=y[self.na_where[var] == False],
                    )
                    self._insert_new_data(dataset, var, imp_values)

                    if iteration == iter_range[-1] and save_models:
                        self.models[dataset][var] = current_model

                if iteration == iter_range[0] and dataset == self.dataset_list[0]:
                    iteration_finish = datetime.now()
                    iteration_time_delta = iteration_finish - iteration_start
                    if iteration_time_delta.seconds > 30 and verbose:
                        expected_completion_time = (
                            iteration_finish
                            + iteration_time_delta * (iterations * 5 - 1)
                        )
                        print(
                            "\nExpected Time of Completion: \n"
                            + expected_completion_time.strftime("%Y-%m-%d - %H:%M")
                            + "\n"
                        )
                        help(datetime.strftime)

        self.iteration_time_seconds = iteration_time_delta.seconds

    def impute_new_data(
        self,
        new_data: DataFrame,
        datasets: List[int] = None,
        iterations: int = None,
        save_all_iterations: bool = False,
        verbose: bool = False,
    ) -> ImputedDataSet:
        """Impute a dataset using a preexisting kernel

        Parameters
        ----------
        new_data: pandas DataFrame
            The new data to impute
        datasets: None, list
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

        Returns
        -------
        ImputedDataSet

        """

        if set(new_data.columns) != set(list(self.data)):
            raise ValueError("Columns are not the same as kernel data")

        if datasets is None:
            dataset_list = self.dataset_list
        elif isinstance(datasets, list):
            if (
                np.max(datasets) > self.dataset_list
                or np.min(datasets) < self.dataset_list
            ):
                raise ValueError("Dataset does not exist")
        else:
            raise ValueError(
                "datasets not recognized. Please provide a list, or leave None to return all."
            )

        if iterations is None:
            iterations = self.get_iterations()

        imputed_data_set = ImputedDataSet(
            new_data,
            datasets=dataset_list,
            variable_schema=self.imputation_schema.variable_schema.copy(),
            save_all_iterations=save_all_iterations,
            verbose=verbose,
        )

        iter_range = range(1, iterations + 1)
        iter_vars = list(imputed_data_set.imputation_schema.response_vars)

        for dataset in dataset_list:
            if verbose:
                print("Dataset " + str(dataset))
            for iteration in iter_range:
                if verbose:
                    print("Iteration " + str(iteration), end="")
                for var in iter_vars:
                    if verbose:
                        if var == iter_vars[-1]:
                            endcap = "\n"
                        else:
                            endcap = ""
                        print(" | " + var, end=endcap)

                    x, y = imputed_data_set._make_xy(dataset, var)
                    kernelx, kernely = self._make_xy(dataset, var)

                    imp_values = self._calculate_imputation_values(
                        model=self.models[dataset][var],
                        var=var,
                        bachelor_features=x[imputed_data_set.na_where[var]],
                        candidate_features=kernelx[self.na_where[var] == False],
                        candidate_values=kernely[self.na_where[var] == False],
                    )
                    imputed_data_set._insert_new_data(dataset, var, imp_values)

        return imputed_data_set

    def get_feature_importance(self, dataset: int = 0) -> DataFrame:
        """
        Return a matrix of feature importance. The cells
        represent the normalized feature importance of the
        columns to impute the rows. This is calculated
        internally by RandomForestRegressor/Classifier.


        Parameters
        ----------
        dataset: int
            The dataset to return the feature importance for.

        Returns
        -------
        pandas DataFrame
            A pandas DataFrame with variable column names and
            indexes.

        """
        # Should change this to save importance as models are updated, so
        # we can still get feature importance even if models are not saved.
        assert self.models_saved
        predictor_names = self.imputation_schema.predictor_vars
        predictor_names.sort()
        imputed_var_names = self.imputation_schema.response_vars
        imputed_var_names.sort()
        importance_matrix = DataFrame(
            columns=predictor_names, index=imputed_var_names, dtype=np.double
        )

        for ivar in imputed_var_names:
            importance_dict = dict(
                zip(
                    self.imputation_schema.variable_schema[ivar],
                    self.models[dataset][ivar].feature_importances_,
                )
            )
            for pvar in importance_dict:
                importance_matrix.loc[ivar, pvar] = np.round(importance_dict[pvar], 3)

        return importance_matrix

    def plot_feature_importance(self, dataset: int = 0, **kw_plot):
        """
        Plot the feature importance. See get_feature_importance
        for more details.

        Parameters
        ----------
        dataset: int
            The dataset to plot the feature importance of
        kw_plot
            Additional arguments sent to sns.heatmap()

        """
        import seaborn as sns

        importance_matrix = self.get_feature_importance(dataset=dataset)
        print(sns.heatmap(importance_matrix, **kw_plot))
