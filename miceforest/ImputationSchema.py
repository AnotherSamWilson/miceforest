from numpy import concatenate, unique
from .utils import _copy_and_remove, _setequal, _list_union
from typing import Optional, Union, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


class _ImputationSchema:
    """
    Imputation Schema

    Contains information about how a dataset should be imputed.
    This class should not be instantiated directly.

    validation_data is the data that is to be imputed using this
    schema.

    variable_schema is validated and edited accordingly here

    mean_match_candidates are pulled from the kernel, so
    validation of mean_match_candidates is left to the kernel
    class, but stored here.
    """

    def __init__(
        self,
        validation_data: "DataFrame",
        variable_schema: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
    ):

        self.na_where = validation_data.isnull()
        self.na_counts = self.na_where.sum()
        self.data_dtypes = validation_data.dtypes
        self.data_shape = validation_data.shape
        self.data_variables = list(validation_data.columns)
        self.vars_with_any_missing = list(self.na_counts[self.na_counts > 0].keys())
        self.mean_match_candidates = mean_match_candidates

        if variable_schema is None:
            variable_schema = self.vars_with_any_missing

        if isinstance(variable_schema, list):
            variable_schema = {
                var: _copy_and_remove(self.data_variables, [var])
                for var in variable_schema
            }

        if isinstance(variable_schema, dict):
            self_impute_attempt = {
                key: (key in value) for key, value in variable_schema.items()
            }
            if any(self_impute_attempt.values()):
                self_impute_vars = [
                    key for key, value in self_impute_attempt.items() if value
                ]
                raise ValueError(
                    ",".join(self_impute_vars)
                    + " variables cannot be used to impute itself."
                )

        # Delete any variables that have no missing values.
        not_imputable = [
            var
            for var in list(variable_schema)
            if var not in self.vars_with_any_missing
        ]
        for rnm in not_imputable:
            del variable_schema[rnm]

        # Store values pertaining to variable schema
        self.variable_schema = variable_schema
        self.predictor_vars = unique(
            concatenate([value for key, value in variable_schema.items()])
        ).tolist()

        self.response_vars = list(variable_schema)
        self.all_vars = unique(self.response_vars + self.predictor_vars)
        self._all_imputed_vars = _list_union(self.vars_with_any_missing, self.all_vars)
        self.n_imputed_vars = len(variable_schema)
        self.not_imputable = not_imputable

        # Get a list of variables that are being used to impute
        # other variables, but are not being imputed themselves.
        self.static_predictors = list(
            set(self.predictor_vars)
            & (set(self.vars_with_any_missing) - set(self.response_vars))
        )
        self.static_predictors.sort()

    def equal_schemas(self, imp_sch, fail=True):
        """
        Checks if two imputation schemas are similar enough
        """
        checks = {
            "response_vars": _setequal(self.response_vars, imp_sch.response_vars),
            "predictor_varsset": _setequal(self.predictor_vars, imp_sch.predictor_vars),
            "na_where": all(self.na_where == imp_sch.na_where),
            "mean_match_candidates": (
                _setequal(self.mean_match_candidates, imp_sch.mean_match_candidates)
            ),
        }
        failed_checks = [key for key, value in checks.items() if value is False]
        if len(failed_checks) > 0:
            if fail:
                raise ValueError(
                    "Inconsistency in schemas in regards to " + ",".join(failed_checks)
                )
            else:
                return False
        else:
            return True
