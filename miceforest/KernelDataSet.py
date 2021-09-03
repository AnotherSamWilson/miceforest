from .ImputedDataSet import ImputedDataSet
from .TimeLog import TimeLog
from datetime import datetime
from .utils import (
    _get_default_mmc,
    _get_default_mms,
    disallowed_aliases_n_estimators,
    MeanMatchType,
    VarSchemType,
)
from pandas import DataFrame
import numpy as np
from typing import Union, Dict, Any, Callable
from .logger import Logger
from lightgbm import train, Dataset

_TIMED_EVENTS = ["mice", "model_fit", "model_predict", "mean_match", "impute_new_data"]


class KernelDataSet(ImputedDataSet):
    """
    Creates a kernel dataset. This dataset can:
        - Perform MICE on itself
        - Impute new data from models obtained from MICE.

    Parameters
    ----------
    data: DataFrame
        A pandas DataFrame.

    variable_schema: None or list or dict
        If None all variables are used to impute all variables which have
        missing values.
        If list all variables are used to impute the variables in the list
        If dict the values will be used to impute the keys.

    mean_match_candidates:  None or int or float or dict
        If float must be 0.0 < mmc <= 1.0. Interpreted as a percentage of available candidates
        If int must be mmc >= 0. Interpreted as the number of candidates.
        If dict, keys must be variable names, and values must follow two above rules.

        The number of mean matching candidates to use.
        Candidates are _always_ drawn from a kernel dataset, even
        when imputing new data.

        Mean matching follows the following rules based on variable type:
            Categorical:
                If mmc = 0, the class with the highest probability is chosen.
                If mmc > 0, return class based on random draw weighted by
                    class probability for each sample.
            Numeric:
                If mmc = 0, the predicted value is used
                If mmc > 0, obtain the mmc closest candidate
                    predictions and collect the associated
                    real candidate values. Choose 1 randomly.

        For more information, see:
        https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

    mean_match_subset: None or int or float or dict.
        If float must be 0.0 < mms <= 1.0. Interpreted as a percentage of available candidates
        If int must be mms >= 0. Interpreted as the number of candidates. If 0, no subsetting is done.
        If dict, keys must be variable names, and values must follow two above rules.

        The number of candidates to search in mean matching. Set
        to a lower number to speed up mean matching. Must be greater
        than mean_match_candidates, and above 0. If a float <= 1.0 is
        passed, it is interpreted as the percentage of the candidates
        to take as a subset. If an int > 1 is passed, it is interpreted
        as the count of candidates to take as a subset.

        Mean matching can take a while on larger datasets. It is recommended to carefully
        select this value for each variable if dealing with very large data.

    mean_match_function: Calable, default = None
        Must take the following parameters:
            mmc: int,
            candidate_preds: np.ndarray,
            bachelor_preds: np.ndarray,
            candidate_values: np.ndarray,
            cat_dtype: CategoricalDtype,
            random_state: np.random.RandomState,

        A default mean matching function will be used if None.

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
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        save_all_iterations: bool = True,
        save_models: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        super().__init__(
            data=data,
            variable_schema=variable_schema,
            mean_match_candidates=mean_match_candidates,
            mean_match_subset=mean_match_subset,
            mean_match_function=mean_match_function,
            save_all_iterations=save_all_iterations,
            random_state=random_state,
        )

        self.save_models = save_models

        # Format mean_match_candidates before priming datasets
        available_candidates = {
            var: (-self.data[var].isna()).sum() for var in self.response_vars
        }
        mean_match_candidates = self._format_mm(
            mean_match_candidates, available_candidates, _get_default_mmc
        )
        mean_match_subset = self._format_mm(
            mean_match_subset, available_candidates, _get_default_mms
        )

        # Ensure mmc and mms make sense:
        # mmc <= mms <= available candidates for each var
        for var in self.response_vars:
            assert (
                mean_match_candidates[var] <= mean_match_subset[var]
            ), f"{var} mean_match_candidates > mean_match_subset"
            assert (
                mean_match_subset[var] <= available_candidates[var]
            ), f"{var} mean_match_subset > available candidates"

        self.mean_match_candidates = mean_match_candidates
        self.mean_match_subset = mean_match_subset

        # Initialize models and time_log
        self.models: Dict[str, Dict] = {var: {0: None} for var in self.response_vars}
        self.time_log = TimeLog(_TIMED_EVENTS)

    def __repr__(self):
        summary_string = " " * 14 + "Class: KernelDataSet\n" + self._ids_info()
        return summary_string

    def _mm_type_handling(self, mm, available_candidates) -> int:
        if isinstance(mm, float):
            assert (mm > 0.0) & (mm <= 1.0)
            ret = int(mm * available_candidates)
        elif isinstance(mm, int):
            assert mm >= 0
            ret = int(mm)
        else:
            raise ValueError(
                "mean_match_candidates type not recognized. "
                + "Any supplied values must be a float <= 1.0 or int >= 1"
            )

        return ret

    def _format_mm(
        self, mm, available_candidates, defaulting_function
    ) -> Dict[str, int]:
        """
        mean_match_candidates and mean_match_subset both require similar formatting.
        The only real difference is the default value based on the number of available
        candidates, which is what the defaulting_function is for.
        """
        if mm is None:
            mm = {
                var: int(defaulting_function(available_candidates[var]))
                for var in self.response_vars
            }
        # If a static value was passed
        elif isinstance(mm, (int, float)):
            mm = {
                var: self._mm_type_handling(mm, available_candidates[var])
                for var in self.response_vars
            }
        elif isinstance(mm, dict):
            if not set(mm).issubset(set(self.response_vars)):
                raise ValueError(
                    "Some keys in mean_matching aren't being imputed. "
                    + "Do all variables in variable_schema have missing values?."
                )
            mm = {
                var: self._mm_type_handling(mm[var], available_candidates[var])
                if var in mm.keys()
                else int(defaulting_function(available_candidates[var]))
                for var in self.response_vars
            }
        else:
            raise ValueError("mean_match_candidates couldn't be interpreted.")

        return mm

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

    def _get_lgb_params(self, var, vsp, **kwlgb):

        seed = self._random_state.randint(1000000, size=1)[0]

        # binary is included in multiclass because we want the multi-output from .predict()
        if var in list(self.categorical_dtypes):
            n_c = len(self.categorical_dtypes[var].categories)
            if n_c > 2:
                obj = {"objective": "multiclass", "num_classes": n_c}
            else:
                obj = {"objective": "binary"}
        else:
            obj = {"objective": "regression"}

        default_lgb_params = {
            **obj,
            "boosting": "random_forest",
            "n_estimators": 48,
            "max_depth": 8,
            "min_data_in_leaf": 5,
            "min_sum_hessian_in_leaf": 0.0,
            "min_gain_to_split": 0.0,
            "bagging_fraction": 0.632,
            "feature_fraction": 0.632,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": seed,
        }

        params = {**default_lgb_params, **kwlgb, **vsp}

        assert not any(x in params for x in disallowed_aliases_n_estimators), (
            "Please use n_estimators "
            "instead of other aliases to "
            "control number of trees."
        )

        return params

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

    def _format_variable_parameters(self, variable_parameters):
        """
        Unpacking will expect an empty dict at a minimum.
        This function collects parameters if they were
        provided, and returns empty dicts if they weren't.
        """
        if variable_parameters is None:
            vsp = {var: {} for var in self.response_vars}
        else:
            vsp_vars = list(variable_parameters)
            assert set(vsp_vars).issubset(
                self.response_vars
            ), "Some variable_parameters are not being imputed."
            vsp = {
                var: variable_parameters[var] if var in vsp_vars else {}
                for var in self.response_vars
            }

        return vsp

    # Models are updated here, and only here.
    def mice(
        self,
        iterations: int = 5,
        verbose: bool = False,
        variable_parameters: Dict[str, Dict[str, Any]] = None,
        **kwlgb,
    ):
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
            lightgbm. Applied to all models.

        """

        logger = Logger(verbose)

        mice_s = datetime.now()

        iterations_at_start = self.iteration_count()
        iter_range = range(
            iterations_at_start + 1, iterations_at_start + iterations + 1
        )
        vsp = self._format_variable_parameters(variable_parameters)

        # Required to shut mypy up.
        assert isinstance(self.mean_match_candidates, dict)
        assert isinstance(self.mean_match_subset, dict)

        for iteration in iter_range:
            logger.log(str(iteration) + " ", end="")
            for var in self.imputation_order:
                logger.log(" | " + var, end="")

                x, y = self._make_xy(var=var)
                candidate_non_missing_ind = np.where(self.na_where[var] == False)[0]
                candidate_features = x.iloc[candidate_non_missing_ind, :]
                candidate_target = y.iloc[candidate_non_missing_ind]

                lgbpars = self._get_lgb_params(var, vsp[var], **kwlgb)
                n_estimators = lgbpars.pop("n_estimators")
                train_pointer = Dataset(data=candidate_features, label=candidate_target)
                fit_s = datetime.now()
                current_model = train(
                    params=lgbpars,
                    train_set=train_pointer,
                    num_boost_round=n_estimators,
                    verbose_eval=False,
                )
                self.time_log.add_time("model_fit", fit_s)

                self._insert_new_model(var=var, model=current_model)

                bachelor_features = x[self.na_where[var]]
                mmc = self.mean_match_candidates[var]
                assert isinstance(mmc, int)  # mypy
                is_categorical = self.is_categorical(var)
                candidate_non_missing_subset = self._random_state.choice(
                    candidate_non_missing_ind,
                    size=self.mean_match_subset[var],
                    replace=False,
                )

                predict_s = datetime.now()
                candidate_preds = current_model.predict(
                    candidate_features.loc[candidate_non_missing_subset, :]
                )
                bachelor_preds = current_model.predict(bachelor_features)
                self.time_log.add_time("model_predict", predict_s)
                candidate_values = candidate_target.loc[
                    candidate_non_missing_subset
                ].values

                meanmatch_s = datetime.now()
                imp_values = self.mean_match_function(
                    mmc=mmc,
                    candidate_preds=candidate_preds,
                    bachelor_preds=bachelor_preds,
                    candidate_values=candidate_values,
                    random_state=self._random_state,
                    cat_dtype=self.categorical_dtypes[var] if is_categorical else None,
                )
                self.time_log.add_time("mean_match", meanmatch_s)
                self._insert_new_data(var, imp_values)

            logger.log("\n", end="")

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

        logger = Logger(verbose)

        impute_new_data_s = datetime.now()

        if set(new_data.columns) != set(self.data.columns):
            raise ValueError("Columns are not the same as kernel data")

        if self.save_models < 1:
            raise ValueError("No models were saved.")

        # User might not be strict about keeping track of their categories.
        # We don't force these, because we don't want to have to create a new copy
        # of the dataframe.
        for cv in self._get_cat_vars():
            assert (
                self.categorical_dtypes[cv] == new_data[cv].dtype
            ), f"{cv} categorical dtype is not consistent. See KernelDataset.categorical_dtypes for required type."

        imputed_data_set = ImputedDataSet(
            data=new_data,
            initialization_data=self.data,
            # copied because it can be edited if there are no
            # missing values in the new data response variables
            variable_schema=self.variable_schema.copy(),
            mean_match_candidates=self.mean_match_candidates,
            mean_match_subset=self.mean_match_subset,
            mean_match_function=self.mean_match_function,
            save_all_iterations=save_all_iterations,
            random_state=self._random_state,
        )

        curr_iters = self.iteration_count()
        iterations = self._default_iteration(iterations)
        iter_range = range(1, iterations + 1)
        iter_vars = imputed_data_set.imputation_order

        # mypy
        assert isinstance(self.mean_match_candidates, dict)
        assert isinstance(self.mean_match_subset, dict)

        for iteration in iter_range:
            logger.log(str(iteration) + " ", end="")

            # Determine which model iteration to grab
            if self.save_models == 1 or iteration > curr_iters:
                itergrab = curr_iters
            else:
                itergrab = iteration

            for var in iter_vars:
                logger.log(" | " + var, end="")

                # Colate bachelor information
                x, y = imputed_data_set._make_xy(var)
                kernelx, kernely = self._make_xy(var)
                bachelor_features = x[imputed_data_set.na_where[var]]

                # Colate candidate information
                candidate_non_missing_ind = np.where(self.na_where[var] == False)[0]
                candidate_features = kernelx.iloc[candidate_non_missing_ind, :]
                candidate_target = kernely.iloc[candidate_non_missing_ind]
                mmc = self.mean_match_candidates[var]
                mms = self.mean_match_subset[var]
                assert isinstance(mmc, int)
                assert isinstance(mms, int)
                current_model = self.get_model(var, itergrab)
                is_categorical = self.is_categorical(var)

                candidate_non_missing_subset = self._random_state.choice(
                    candidate_non_missing_ind, size=mms, replace=False
                )

                predict_s = datetime.now()
                candidate_preds = current_model.predict(
                    candidate_features.loc[candidate_non_missing_subset, :]
                )
                bachelor_preds = current_model.predict(bachelor_features)
                self.time_log.add_time("model_predict", predict_s)
                candidate_values = candidate_target.loc[
                    candidate_non_missing_subset
                ].values

                meanmatch_s = datetime.now()
                imp_values = self.mean_match_function(
                    mmc=mmc,
                    candidate_preds=candidate_preds,
                    bachelor_preds=bachelor_preds,
                    candidate_values=candidate_values,
                    random_state=self._random_state,
                    cat_dtype=self.categorical_dtypes[var] if is_categorical else None,
                )
                self.time_log.add_time("mean_match", meanmatch_s)
                imputed_data_set._insert_new_data(var, imp_values)

            logger.log("\n", end="")

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
                    self.get_model(ivar, iteration).feature_importance(),
                )
            )
            for pvar in importance_dict:
                importance_matrix.loc[ivar, pvar] = importance_dict[pvar]

        importance_matrix.rename_axis("Imputed Variables", axis=0, inplace=True)
        importance_matrix.rename_axis("Feature Variables", axis=1, inplace=True)

        return importance_matrix

    def plot_feature_importance(
        self, normalize: bool = True, iteration: int = None, **kw_plot
    ):
        """
        Plot the feature importance. See get_feature_importance()
        for more details.

        Parameters
        ----------
        iteration: int
            The iteration to plot the feature importance of.
        normalize: book
            Should the values be normalize from 0-1?
            If False, values are raw from Booster.feature_importance()
        kw_plot
            Additional arguments sent to sns.heatmap()

        """
        import seaborn as sns

        importance_matrix = self.get_feature_importance(iteration=iteration)
        if normalize:
            importance_matrix = importance_matrix.divide(
                importance_matrix.sum(1), 0
            ).round(2)

        params = {
            **{
                "cmap": "coolwarm",
                "annot": True,
                "fmt": ".2f",
                "annot_kws": {"size": 16},
            },
            **kw_plot,
        }

        print(sns.heatmap(importance_matrix, **params))
