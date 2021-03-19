from .ImputedDataSet import ImputedDataSet
from sklearn.neighbors import NearestNeighbors
from .TimeLog import TimeLog
from datetime import datetime
from .utils import _get_default_mmc, _default_rf_classifier, _default_rf_regressor
from pandas import DataFrame
import numpy as np
from typing import Union, List, Dict, Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_TIMED_EVENTS = ["mice", "model_fit", "model_predict", "mean_match", "impute_new_data"]


class KernelDataSet(ImputedDataSet):
    """
    Creates a kernel dataset. This dataset can:
        - Perform MICE on itself
        - Impute new data from models obtained from MICE.

    Parameters
    ----------
    data: DataFrame
        A pandas DataFrame to impute.

    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all variables are used to impute the variables in the list
        If dict the values will be used to impute the keys.

    mean_match_candidates:  None or int or dict
        The number of mean matching candidates to use.
        Candidates are _always_ drawn from a kernel dataset, even
        when imputing new data.

        Mean matching follows the following rules based on variable type:
            Categorical:
                If mmc = 0, the predicted class is used
                If mmc > 0, return class based on random draw weighted by
                    class probability for each sample.
            Numeric:
                If mmc = 0, the predicted value is used
                If mmc > 0, obtain the mmc closest candidate
                    predictions and collect the associated
                    real candidate values. Choose 1 randomly.

        For more information, see:
        https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

    save_all_iterations: boolean, optional(default=True)
        Save all the imputation values from all iterations, or just
        the latest. Saving all iterations allows for additional
        plotting, but may take more memory

    save_models: int
        Which models should be saved:
            = 0: no models are saved. Cannot get feature importance or
                impute new data.
            = 1: only the last model iteration is saved. Can only get
                feature importance of last iteration. New data is
                imputed using the last model for all specified iterations.
                This is only an issue if data is heavily Missing At Random.
            = 2: all model iterations are saved. Can get feature importance
                for any iteration. When imputing new data, each iteration is
                imputed using the model obtained at that iteration in mice.
                This allows for imputations that most closely resemble those
                that would have been obtained in mice.

    random_state: None,int, or numpy.random.RandomState
        Ensures a random state throughout the process

    """

    def __init__(
        self,
        data: DataFrame,
        variable_schema: Union[List[str], Dict[str, List[str]]] = None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
        save_all_iterations: bool = True,
        save_models: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        initial_imputation: Union[str, dict, Callable] = None, 
    ):
        super().__init__(
            data=data,
            variable_schema=variable_schema,
            mean_match_candidates=mean_match_candidates,
            save_all_iterations=save_all_iterations,
            random_state=random_state,
            initial_imputation=initial_imputation,
        )

        self.save_models = save_models

        # Format mean_match_candidates before priming datasets
        available_candidates = {
            var: self.data_shape[0] - self.data[var].isna().sum()
            for var in self.response_vars
        }

        if self.mean_match_candidates is None:
            self.mean_match_candidates = {
                var: _get_default_mmc(available_candidates[var])
                for var in self.response_vars
            }

        elif isinstance(self.mean_match_candidates, int):
            self.mean_match_candidates = {
                key: self.mean_match_candidates for key in self.response_vars
            }
        elif isinstance(mean_match_candidates, dict):
            if not set(mean_match_candidates) == set(self.response_vars):
                raise ValueError(
                    "mean_match_candidates not consistent with variable_schema. "
                    + "Do all variables in variable_schema have missing values?."
                )

        mmc_inadequate = [
            var
            for var, mmc in self.mean_match_candidates.items()
            if (mmc >= available_candidates[var])
        ]
        if len(mmc_inadequate) > 0:
            raise ValueError(
                "<"
                + ",".join(mmc_inadequate)
                + ">"
                + " do not have enough candidates to perform mean matching."
            )

        # Initialize models
        self.models: Dict[str, Dict] = {var: {0: None} for var in self.response_vars}

        self.time_log = TimeLog(_TIMED_EVENTS)

    def __repr__(self):
        summary_string = " " * 14 + "Class: KernelDataSet\n" + self._ids_info()
        return summary_string

    def _insert_new_model(self, var, model):
        """
        Inserts a new model if save_mdoels > 0.
        Deletes the prior one if save_models == 1.
        """
        current_iter = self.iteration_count(var)
        if self.save_models == 0:
            return
        else:
            self.models[var][current_iter + 1] = model
        if self.save_models == 1:
            del self.models[var][current_iter]

    def get_model(self, var, iteration=None):
        """
        Return the model for a specific variable, iteration.

        Parameters
        ----------
        var: str
            The variable that was imputed
        iteration: int
            The model iteration to return. Keep in mind if:
                - save_models == 0, no models are saved
                - save_models == 1, only the last model iteration is saved
                - save_models == 2, all model iterations are saved

        Returns: RandomForestRegressor or RandomForestClassifier
            The model used to impute this specific variable, iteration.

        """

        iteration = self._default_iteration(iteration)
        try:
            return self.models[var][iteration]
        except Exception:
            raise ValueError("Iteration was not saved")

    def _mean_match(
        self,
        model,
        mmc: int,
        is_categorical: bool,
        bachelor_features: DataFrame,
        candidate_features: DataFrame = None,
        candidate_values: np.array = None,
    ):
        """
        Performs mean matching. Logic:
            if categorical:
                Return class based on random draw weighted by
                class probability for each sample.
            if numeric:
                For each sample prediction, obtain the mmc closest
                candidate_values and collect the associated
                candidate_features. Choose 1 randomly.

        For a graphical example, see:
        https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

        Parameters
        ----------
        model
            The model
        mmc
            The mean matching candidates
        is_categorical
            Is the feature we are imputing categorical
        bachelor_features
            The features of the variable to be imputed
        candidate_features
            The features of the candidates
        candidate_values
            The real values of the candidates

        Returns
        -------
            An array of imputed values.

        """
        mean_match_s = datetime.now()

        if is_categorical:
            # Select category according to their probability for each sample
            model_predict_s = datetime.now()
            bachelor_preds = model.predict_proba(bachelor_features)
            self.time_log.add_time("model_predict", model_predict_s)
            imp_values = [
                self._random_state.choice(model.classes_, p=p, size=1)[0]
                for p in bachelor_preds
            ]
        else:
            # Collect the candidates, and the predictions for the candidates and bachelors
            model_predict_s = datetime.now()
            bachelor_preds = np.array(model.predict(bachelor_features))
            candidate_preds = np.array(model.predict(candidate_features))
            self.time_log.add_time("model_predict", model_predict_s)
            candidate_values = np.array(candidate_values)

            # Determine the nearest neighbors of the bachelor predictions in the candidate predictions
            knn = NearestNeighbors(n_neighbors=mmc, algorithm="ball_tree")
            knn.fit(candidate_preds.reshape(-1, 1))
            knn_indices = knn.kneighbors(
                bachelor_preds.reshape(-1, 1), return_distance=False
            )
            index_choice: List[int] = [
                self._random_state.choice(i) for i in knn_indices
            ]
            imp_values = candidate_values[index_choice]

        self.time_log.add_time("mean_match", mean_match_s)
        return imp_values

    # Models are updated here, and only here.
    def mice(self, iterations: int = 5, verbose: bool = False, **kw_fit):
        """
        Perform mice given dataset.

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

        mice_s = datetime.now()

        iterations_at_start = self.iteration_count()
        iter_range = range(
            iterations_at_start + 1, iterations_at_start + iterations + 1
        )

        # Required shut mypy up.
        assert isinstance(self.mean_match_candidates, Dict)

        for iteration in iter_range:
            if verbose:
                print(str(iteration) + " ", end="")
            for var in self.response_vars:
                if verbose:
                    print(" | " + var, end="")

                x, y = self._make_xy(var=var)
                non_missing_ind = self.na_where[var] == False
                candidate_features = x[non_missing_ind]
                candidate_values = y[non_missing_ind]

                fit_s = datetime.now()
                if var in self.categorical_variables:
                    current_model = _default_rf_classifier(
                        random_state=self._random_state, **kw_fit
                    )
                else:
                    current_model = _default_rf_regressor(
                        random_state=self._random_state, **kw_fit
                    )

                current_model.fit(X=candidate_features, y=candidate_values)
                self.time_log.add_time("model_fit", fit_s)

                self._insert_new_model(var=var, model=current_model)

                bachelor_features = x[self.na_where[var]]
                mmc = self.mean_match_candidates[var]

                if mmc == 0:
                    model_predict_s = datetime.now()
                    imp_values = current_model.predict(bachelor_features)
                    self.time_log.add_time("model_predict", model_predict_s)
                else:
                    is_categorical = var in self.categorical_variables
                    if is_categorical:
                        candidate_features = candidate_values = None
                    else:
                        ind = self.na_where[var] == False
                        candidate_features = x[ind]
                        candidate_values = y[ind]

                    imp_values = self._mean_match(
                        model=current_model,
                        mmc=mmc,
                        is_categorical=is_categorical,
                        bachelor_features=bachelor_features,
                        candidate_features=candidate_features,
                        candidate_values=candidate_values,
                    )

                self._insert_new_data(var, imp_values)
            if verbose:
                print("\n", end="")
        self.time_log.add_time("mice", mice_s)

    def impute_new_data(
        self,
        new_data: "DataFrame",
        iterations: int = None,
        save_all_iterations: bool = True,
        verbose: bool = False,
    ) -> ImputedDataSet:
        """
        Impute a new dataset

        Uses the models obtained while running MICE to impute new data,
        without fitting new models. Pulls mean matching candidates from
        the original data.

        save_models must be > 0. If save_models == 1, the last model
        obtained in mice is used for every iteration. If save_models > 1,
        the model obtained at each iteration is used to impute the new
        data for that iteration. If specified iterations is greater than
        the number of iterations run so far using mice, the last model
        is used for each additional iteration.

        Parameters
        ----------
        new_data: pandas DataFrame
            The new data to impute
        iterations: None, int
            The iterations to run. If None, the same number
            of iterations in the kernel are used. If iterations
            is greater than the number of MICE iterations, the
            latest model will be used for any additional iterations.
        save_all_iterations: bool
            Whether to save about all of the imputed values
            in every iteration, or just the last iteration.
        verbose: bool
            Print progress

        Returns
        -------
        ImputedDataSet

        """

        impute_new_data_s = datetime.now()

        if set(new_data.columns) != set(self.data.columns):
            raise ValueError("Columns are not the same as kernel data")

        if self.save_models < 1:
            raise ValueError("No models were saved.")

        imputed_data_set = ImputedDataSet(
            new_data,
            # copied because it can be edited if there are no
            # missing values in the new data response variables
            variable_schema=self.variable_schema.copy(),
            mean_match_candidates=self.mean_match_candidates,
            save_all_iterations=save_all_iterations,
            random_state=self._random_state,
        )

        curr_iters = self.iteration_count()
        iterations = self._default_iteration(iterations)
        iter_range = range(1, iterations + 1)
        iter_vars = imputed_data_set.response_vars
        assert isinstance(self.mean_match_candidates, Dict)

        for iteration in iter_range:
            if verbose:
                print("\n" + str(iteration) + " ", end="")

            # Determine which model iteration to grab
            if self.save_models == 1 or iteration > curr_iters:
                itergrab = curr_iters
            else:
                itergrab = iteration

            for var in iter_vars:
                if verbose:
                    if var == iter_vars[-1]:
                        endcap = "\n"
                    else:
                        endcap = ""
                    print(" | " + var, end=endcap)

                x, y = imputed_data_set._make_xy(var)
                kernelx, kernely = self._make_xy(var)
                bachelor_features = x[imputed_data_set.na_where[var]]
                mmc = self.mean_match_candidates[var]

                if mmc == 0:
                    imp_values = self.get_model(var, itergrab).predict(
                        bachelor_features
                    )
                else:
                    is_categorical = var in self.categorical_variables
                    if is_categorical:
                        candidate_features = candidate_values = None
                    else:
                        ind = self.na_where[var] == False
                        candidate_features = kernelx[ind]
                        candidate_values = kernely[ind]

                    imp_values = self._mean_match(
                        model=self.get_model(var, itergrab),
                        mmc=mmc,
                        is_categorical=is_categorical,
                        bachelor_features=bachelor_features,
                        candidate_features=candidate_features,
                        candidate_values=candidate_values,
                    )

                imputed_data_set._insert_new_data(var, imp_values)

        self.time_log.add_time("impute_new_data", impute_new_data_s)
        return imputed_data_set

    def get_feature_importance(self, iteration: int = None) -> DataFrame:
        """
        Return a matrix of feature importance. The cells
        represent the normalized feature importance of the
        columns to impute the rows. This is calculated
        internally by RandomForestRegressor/Classifier.

        Parameters
        ----------
        iteration: int
            The iteration to return the feature importance for.
            Right now, the model must be saved to return importance

        Returns
        -------
        pandas DataFrame
            A pandas DataFrame with variable column names and
            indexes.

        """
        # Should change this to save importance as models are updated, so
        # we can still get feature importance even if models are not saved.

        iteration = self._default_iteration(iteration)

        importance_matrix = DataFrame(
            columns=sorted(self.predictor_vars),
            index=sorted(self.response_vars),
            dtype=np.double,
        )

        for ivar in self.response_vars:
            importance_dict = dict(
                zip(
                    self.variable_schema[ivar],
                    self.get_model(ivar, iteration).feature_importances_,
                )
            )
            for pvar in importance_dict:
                importance_matrix.loc[ivar, pvar] = np.round(importance_dict[pvar], 3)

        return importance_matrix

    def plot_feature_importance(self, iteration: int = None, **kw_plot):
        """
        Plot the feature importance. See get_feature_importance()
        for more details.

        Parameters
        ----------
        iteration: int
            The iteration to plot the feature importance of.
        kw_plot
            Additional arguments sent to sns.heatmap()

        """
        import seaborn as sns

        importance_matrix = self.get_feature_importance(iteration=iteration)
        print(sns.heatmap(importance_matrix, **kw_plot))
