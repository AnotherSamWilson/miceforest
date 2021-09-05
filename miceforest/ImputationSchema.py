from numpy import concatenate
from .utils import (
    _copy_and_remove,
    _setequal,
    _list_union,
    MeanMatchType,
    VarSchemType,
    _get_default_mmc,
    _get_default_mms
)
from typing import Union, List, TYPE_CHECKING, Callable, Dict
from pandas import unique

if TYPE_CHECKING:
    from pandas import DataFrame


class _ImputationSchema:
    """
    Imputation Schema

    Contains information about how a dataset should be imputed.
    This class should not be instantiated directly.

    kernel_data is the raw data from the kernel that will be used
    to impute data passed to impute_data. For kernels, these two
    datasets are the same.
    """

    def __init__(
        self,
        kernel_data: "DataFrame",
        impute_data: "DataFrame",
        variable_schema: VarSchemType = None,
        mean_match_candidates: MeanMatchType = None,
        mean_match_subset: MeanMatchType = None,
        mean_match_function: Callable = None,
        imputation_order: Union[str, List[str]] = "ascending",
        verbose: bool = False
    ):

        # Add strict column enforcements later.
        assert set(kernel_data.columns).issubset(impute_data.columns)

        # Store impute_data meta data. Do not store information about kernel_data.
        self.na_where = impute_data.isnull()
        self.na_counts = self.na_where.sum()
        self.data_dtypes = impute_data.dtypes
        self.data_shape = impute_data.shape
        self.data_variables = list(impute_data.columns)
        self.vars_with_any_missing = list(self.na_counts[self.na_counts > 0].keys())

        # Enforce types
        kernel_dtypes = kernel_data.dtypes
        unmatched_types = [var for var in kernel_data.columns if kernel_dtypes[var] != self.data_dtypes[var]]
        if len(unmatched_types) > 0:
            raise ValueError(",".join(unmatched_types) + " don't match kernel dtypes. Check categories.")

        # Formatting of variable_schema.
        if variable_schema is None:
            variable_schema = self.vars_with_any_missing

        if isinstance(variable_schema, list):
            variable_schema = {
                var: _copy_and_remove(self.data_variables, [var])
                for var in variable_schema
            }

        if isinstance(variable_schema, dict):
            self_impute_attempt = [var for var in list(variable_schema) if var in variable_schema[var]]
            if len(self_impute_attempt) > 0:
                raise ValueError(
                    ",".join(self_impute_attempt)
                    + " variables cannot be used to impute itself."
                )

        # Delete any variables from variable_schema keys that have no missing values.
        not_imputable = [
            var
            for var in list(variable_schema)
            if var not in self.vars_with_any_missing
        ]
        if verbose and len(not_imputable) > 0:
            print(
                ",".join(not_imputable)
                + " will not be imputed because they contained no missing values."
            )
        for rnm in not_imputable:
            del variable_schema[rnm]

        # variable_schema is specific to impute_data.
        self.variable_schema = variable_schema

        # Any variables that will be used as features during imputation.
        self.predictor_vars = unique(
            concatenate([value for key, value in variable_schema.items()])
        ).tolist()

        # A list of variables imputed with a model.
        self.response_vars = list(variable_schema)

        # All variables used during the process.
        self.all_vars = unique(self.response_vars + self.predictor_vars).tolist()

        # Any variable that needs imputation to be a part of the process (used during initialization)
        self._all_imputed_vars = _list_union(self.vars_with_any_missing, self.all_vars)
        self.n_imputed_vars = len(variable_schema)
        self.not_imputable = not_imputable

        # Get a list of variables that are being used to impute
        # other variables, but are not being imputed themselves.
        self.static_predictors = [
            var for var in self.predictor_vars
            if var in (set(self.vars_with_any_missing) - set(self.response_vars))
        ]
        if verbose and len(self.static_predictors) > 0:
            print(
                ",".join(self.static_predictors)
                + " are being used to predict other variables, but are not being imputed themselves."
            )

        # Get list of categorical variables
        self.categorical_variables = [
            i for i in self.all_vars if (self.data_dtypes == "category")[i]
        ]
        self.categorical_dtypes = {
            var: kernel_data[var].dtype for var in self.categorical_variables
        }

        # Format mean_match_candidates before priming datasets
        available_candidates = {
            var: (-kernel_data[var].isna()).sum() for var in self.response_vars
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

        # Only import sklearn if we really need to.
        if mean_match_function is None:
            from .default_mean_match import default_mean_match
            self.mean_match_function = default_mean_match
        else:
            self.mean_match_function = mean_match_function

        # Format imputation order
        if isinstance(imputation_order, list):
            # subset because response_vars can be removed if imputing new data with no missing values.
            assert set(self.response_vars).issubset(
                imputation_order
            ), "imputation_order does not include all variables to be imputed."
            imputation_order = [i for i in imputation_order if i in self.response_vars]
        elif isinstance(imputation_order, str):
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
            else:
                raise ValueError("imputation_order not recognized.")

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
            mm = {
                var: self._mm_type_handling(mm[var], available_candidates[var])
                if var in mm.keys()
                else int(defaulting_function(available_candidates[var]))
                for var in self.response_vars
            }
        else:
            raise ValueError("mean_match_candidates couldn't be interpreted.")

        return mm

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
