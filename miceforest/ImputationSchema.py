from numpy import concatenate
from .utils import _copy_and_remove, _setequal, _list_union, MeanMatchType, VarSchemType
from typing import Union, List, TYPE_CHECKING, Callable
from pandas import unique

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
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        imputation_order: Union[str, List[str]] = "ascending",
    ):

        self.na_where = validation_data.isnull()
        self.na_counts = self.na_where.sum()
        self.data_dtypes = validation_data.dtypes
        self.data_shape = validation_data.shape
        self.data_variables = list(validation_data.columns)
        self.vars_with_any_missing = list(self.na_counts[self.na_counts > 0].keys())
        self.mean_match_candidates = mean_match_candidates
        self.mean_match_subset = mean_match_subset

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
        self.all_vars = unique(self.response_vars + self.predictor_vars).tolist()
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

        # Get list of categorical variables
        self.categorical_variables = [
            i for i in self.all_vars if (self.data_dtypes == "category")[i]
        ]
        self.categorical_dtypes = {
            var: validation_data[var].dtype for var in self.categorical_variables
        }

        # Only import sklearn if we really need to.
        if mean_match_function is None:
            from .default_mean_match import default_mean_match

            self.mean_match_function = default_mean_match
        else:
            self.mean_match_function = mean_match_function

        # Format imputation order
        if isinstance(imputation_order, list):
            # subset because response_vars can be removed if imputing new data with no missing values
            # for certain variables.
            assert set(self.response_vars).issubset(
                imputation_order
            ), "response vars not subset of imputation_order"
            imputation_order = [i for i in imputation_order if i in self.response_vars]
        elif isinstance(imputation_order, str):
            assert imputation_order in [
                "ascending",
                "descending",
                "roman",
                "arabic",
            ], "imputation_order not recognized"
            if imputation_order == "ascending":
                missing_order = self.na_counts.sort_values(ascending=True).index.values
                imputation_order = [i for i in missing_order if i in self.response_vars]
            elif imputation_order == "descending":
                missing_order = self.na_counts.sort_values(ascending=False).index.values
                imputation_order = [i for i in missing_order if i in self.response_vars]
            elif imputation_order == "roman":
                imputation_order = self.response_vars.copy()
            elif imputation_order == "arabic":
                imputation_order = self.response_vars.copy()
                imputation_order.reverse()

        self.imputation_order = imputation_order

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

    def is_categorical(self, var):
        assert var in self.all_vars, "Variable not recognized."
        return True if var in list(self.categorical_dtypes) else False

    def _varfilter(self, vrs, response, predictor) -> List[str]:
        """
        Extracts predictor and response variables
        from a list of variables.
        """
        if not response and not predictor:
            return vrs
        if response:
            vrs = _list_union(vrs, self.response_vars)
        if predictor:
            vrs = _list_union(vrs, self.predictor_vars)
        return vrs

    def _get_cat_vars(self, response=True, predictor=False) -> List[str]:
        cat_vars = self._varfilter(
            vrs=self.categorical_variables, response=response, predictor=predictor
        )
        return cat_vars

    def _get_num_vars(self, response=True, predictor=False):
        categorical_variables = self._get_cat_vars(True, True)
        num_vars = [v for v in self.all_vars if v not in categorical_variables]
        num_vars = self._varfilter(vrs=num_vars, response=response, predictor=predictor)
        return num_vars
