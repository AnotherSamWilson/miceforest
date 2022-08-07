from .compat import pd_DataFrame
import inspect
from copy import deepcopy
from typing import Callable, Union, Dict, Set
from numpy import dtype


_REGRESSIVE_OBJECTIVES = [
    "regression",
    "regression_l1",
    "poisson",
    "huber",
    "fair",
    "mape",
    "cross_entropy",
    "cross_entropy_lambda" "quantile",
    "tweedie",
    "gamma",
]

_CATEGORICAL_OBJECTIVES = [
    "binary",
    "multiclass",
    "multiclassova",
]

AVAILABLE_MEAN_MATCH_ARGS = [
    "mean_match_candidates",
    "lgb_booster",
    "bachelor_preds",
    "bachelor_features",
    "candidate_values",
    "candidate_features",
    "candidate_preds",
    "random_state",
    "hashed_seeds",
]

_DEFAULT_MMC = 5

_t_mmc = Union[int, Dict[str, int], Dict[int, int]]
_t_obj_func = Dict[str, Callable]
_t_np_dt = Dict[str, str]


class MeanMatchScheme:
    def __init__(
        self,
        mean_match_candidates: _t_mmc,
        mean_match_functions: _t_obj_func,
        lgb_model_pred_functions: _t_obj_func,
        objective_pred_dtypes: Dict[str, str],
    ):
        """
        Stores information and methods surrounding how mean matching should
        behave for each variable. This class is responsible for determining:

        * The mean matching function
        * How predictions are obtained from a lightgbm model
        * The datatype of the predictions
        * The number of mean matching candidates
        * Which variables will have candidate predictions compiled

        During the imputation process, the following series of events occur
        for each variable:

        1) ImputationKernel trains a lightgbm model
        2) MeanMatchScheme gets predictions from lightgbm model based on the model objective.
        3) MeanMatchScheme performs mean matching on the predictions.

        Parameters
        ----------
        mean_match_candidates: int or dict, default = 5
            The number of mean matching candidates to pull.
            One of these candidates will be chosen randomly
            as the imputation value. To skip mean matching,
            set to 0.

            If int, that value is used for all variables.

            If a dict is passed, it should be of the form:
            {variable: int} pairs. Any variables not specified
            will use the default. Variables can be passed as
            column names or indexes.

            For more information about mean matching, see:
            https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

        mean_match_functions: dict
            A dict of {objective: callable} pairs. The objective should be
            a lightgbm objective. The provided function will be used to
            perform mean matching for all models with the given objective.

            RULES FOR FUNCTIONS:
                1) The only arguments allowed are those listed in
                   miceforest.mean_match_schemes.AVAILABLE_MEAN_MATCH_ARGS
                2) Not all of those arguments are required, the process
                   will only pass arguments that the function requires

        lgb_model_pred_functions: dict
            A dict of {objective: callable} pairs. The objective should be
            a lightgbm objective. The provided function will be used to
            obtain predictions from lightgbm models with the paired objective.

            For example, passing: {"binary": func, "regression": func2} will call
            func(model, data) to obtain the predictions from the model, if the
            model objective was binary.

        objective_pred_dtypes: dict
            A dict of {objective: np.datatype} pairs.
            Datatype must be the string datatype name, i.e. "float32"
            How prediction data types will be cast. Casting to a smaller bit
            value can be beneficial when compiling candidates. Depending on
            the data, smaller bit rates tend to result in imputations of
            sufficient quality, while taking up much less space.

        """

        self.mean_match_functions: Dict[str, Callable] = {}
        self.objective_args: Dict[str, Set[str]] = {}
        self.objective_pred_funcs: Dict[str, Callable] = {}
        self.objective_pred_dtypes = objective_pred_dtypes.copy()

        for objective, function in mean_match_functions.items():
            self._add_mmf(objective, function)

        for objective, function in lgb_model_pred_functions.items():
            self._add_lgbpred(objective, function)

        self.mean_match_candidates = mean_match_candidates
        self._mmc_formatted = False

    def _add_mmf(self, objective: str, func: Callable):
        obj_args = set(inspect.getfullargspec(func).args)
        assert obj_args.issubset(
            set(AVAILABLE_MEAN_MATCH_ARGS)
        ), f"arg not available to mean match function for objective {objective}, check arguments."
        self.mean_match_functions[objective] = func
        self.objective_args[objective] = obj_args

    def _add_lgbpred(self, objective: str, func: Callable):
        obj_args = set(inspect.getfullargspec(func).args)
        assert obj_args == {
            "data",
            "model",
        }, f"functions in lgb_model_pred_functions should only have 2: parameters, model and data."
        self.objective_pred_funcs[objective] = func

    def _format_mean_match_candidates(self, data, available_candidates):

        var_indx_list = list(available_candidates)
        assert not self._mmc_formatted, "mmc are already formatted"

        if isinstance(self.mean_match_candidates, int):
            mmc_formatted = {v: self.mean_match_candidates for v in var_indx_list}

        elif isinstance(self.mean_match_candidates, dict):
            mmc_formatted = {}
            for v, mmc in self.mean_match_candidates.items():
                if isinstance(v, str):
                    assert isinstance(
                        data, pd_DataFrame
                    ), "columns cannot be names unless data is pandas DF"
                    v_ind = data.columns.tolist().index(v)
                    assert (
                        v_ind in var_indx_list
                    ), f"{v} is not having a model built, should not get mmc."
                    assert (
                        mmc <= available_candidates[v_ind]
                    ), f"{v} doesn't have enough candidates for mmc {mmc}"
                    mmc_formatted[v_ind] = mmc

                else:
                    mmc_formatted[v] = mmc

            for v in var_indx_list:
                if v not in list(mmc_formatted):
                    mmc_formatted[v] = _DEFAULT_MMC

        else:
            raise ValueError(
                "malformed mean_match_candidates was passed to MeanMatchScheme"
            )

        self.mean_match_candidates = mmc_formatted
        self._mmc_formatted = True

    def copy(self):
        """
        Return a copy of this schema.

        Returns
        -------
        A copy of this MeanMatchSchema
        """
        return deepcopy(self)

    def set_mean_match_candidates(self, mean_match_candidates: _t_mmc):
        """
        Set the mean match candidates

        Parameters
        ----------
        mean_match_candidates: int or dict
            The number of mean matching candidates to pull.
            One of these candidates will be chosen randomly
            as the imputation value. To skip mean matching,
            set to 0.

            If int, that value is used for all variables.

            If a dict is passed, it should be of the form:
            {variable: int} pairs. Any variables not specified
            will use the default. Variables can be passed as
            column names or indexes.

            For more information about mean matching, see:
            https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching
        """
        self.mean_match_candidates = mean_match_candidates
        self._mmc_formatted = False

    def set_mean_match_function(self, mean_match_functions: _t_obj_func):
        """
        Overwrite the current mean matching functions for certain
        objectives.

        Parameters
        ----------
        mean_match_functions: dict
            A dict of {objective: callable} pairs. The objective should be
            a lightgbm objective. The provided function will be used to
            perform mean matching for all models with the given objective.

            RULES FOR FUNCTIONS:
                1) The only arguments allowed are those listed in
                   miceforest.mean_match_schemes.AVAILABLE_MEAN_MATCH_ARGS
                2) Not all of those arguments are required, the process
                   will only pass arguments that the function requires

        """
        for objective, function in mean_match_functions.items():
            self._add_mmf(objective, function)

    def set_lgb_model_pred_functions(self, lgb_model_pred_functions: _t_obj_func):
        """
        Overwrite the current prediction functions for certain
        objectives.

        Parameters
        ----------
        lgb_model_pred_functions: dict
            A dict of {objective: callable} pairs. The objective should be
            a lightgbm objective. The provided function will be used to
            obtain predictions from lightgbm models with the paired objective.

            For example, passing: {"binary": func, "regression": func2} will call
            func(model, data) to obtain the predictions from the model, if the
            model objective was binary.
        """
        for objective, function in lgb_model_pred_functions.items():
            self._add_lgbpred(objective, function)

    def set_objective_pred_dtypes(self, objective_pred_dtypes: Dict[str, str]):
        """
        Overwrite the current datatypes for certain objectives.
        Predictions obtained from lightgbm are stored as these
        datatypes.

        Parameters
        ----------
        objective_pred_dtypes: dict
            A dict of {objective: np.datatype} pairs.
            Datatype must be the string datatype name, i.e. "float32"
            How prediction data types will be cast. Casting to a smaller bit
            value can be beneficial when compiling candidates. Depending on
            the data, smaller bit rates tend to result in imputations of
            sufficient quality, while taking up much less space.

        """
        self.objective_pred_dtypes.update(objective_pred_dtypes)

    def get_objectives_requiring_candidate_preds(self):
        """
        Easy way to determine which lightgbm objectives will
        require candidate predictions to run mean matching.

        Returns
        -------
        list of objectives.
        """
        objs = []
        for objective, obj_args in self.objective_args.items():
            if "candidate_preds" in obj_args:
                objs.append(objective)
        return objs

    def get_mean_match_args(self, objective):
        """
        Determine what arguments we need to procure for a mean matching function.

        Parameters
        ----------
        objective: str
            The objective of the model that will create the candidate
            and bachelor predictions.

        Returns
        -------
        list of arguments required by this objectives mean matching function.
        """
        try:
            obj_args = self.objective_args[objective]

        except:
            raise ValueError(
                f"Could not get mean matching args for objective {objective}"
                + "Add this objective to the MeanMatchSchema by using .set_mean_match_function()"
            )

        return obj_args

    def model_predict(self, model, data, dtype=None):
        """
        Get the predictions from a model on data using the
        internal prediction functions.

        Parameters
        ----------
        model: Booster
            The model

        data: pd.DataFrame or np.array
            The data to get predictions for

        dtype: string or np.datatype
            Returns prediction as this datatype.
            Datatypes are kept track of internally, however
            you can overwrite the data type with this parameter.

        Returns
        -------
        np.ndarray of predictions

        """
        objective = model.params["objective"]
        if dtype is None:
            dtype = self.objective_pred_dtypes[objective]
        pred_func = self.objective_pred_funcs[objective]
        preds = pred_func(model=model, data=data).astype(dtype)
        return preds

    def _mean_match(self, variable, objective, **kwargs):
        """
        Perform mean matching

        Parameters
        ----------
        variable: int
            The variable being imputed

        objective: str
            The lightgbm objective used to create predictions for
            the variable.

        kwargs
            miceforest will automatically determine which objects
            need to be generated based on the mean matching function
            arguments. These arguments are passed as kwargs.

        Returns
        -------
        np.ndarray of imputation values.

        """
        obj_args = self.objective_args[objective]
        mmf = self.mean_match_functions[objective]
        mmc = self.mean_match_candidates[variable]
        if "mean_match_candidates" in obj_args:
            kwargs["mean_match_candidates"] = mmc

        for oa in obj_args:
            assert oa in list(
                kwargs
            ), f"objective {objective} requires {oa}, but it wasn't passed."

        mmf_args = {arg: val for arg, val in kwargs.items() if arg in obj_args}
        imp_values = mmf(**mmf_args)
        return imp_values
