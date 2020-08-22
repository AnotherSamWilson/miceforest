from numpy import concatenate
from .utils import _distinct_from_list


class ImputationSchema:
    """
    Imputation Schema

    Contains information about how a dataset should be imputed.
    This class should not be instantiated directly.
    """
    def __init__(self,
                 variable_schema,
                 validation_data,
                 verbose):

        na_where = validation_data.isnull()
        self.na_where = na_where
        na_counts = na_where.sum()
        self.na_counts = na_counts
        data_dtypes = validation_data.dtypes
        self.data_dtypes = data_dtypes
        data_shape = validation_data.shape
        self.data_shape = data_shape
        self.vars_with_any_missing = list(na_counts[na_counts > 0].keys())
        self.categorical_variables = list(data_dtypes[data_dtypes == "category"].keys())

        # Format variable_schema appropriately.
        # Values passed can be None, list, or dict.
        if variable_schema is None:
            variable_schema = list(validation_data.columns)
        if variable_schema.__class__ == list:
            variable_schema = {key:sorted(list(set(variable_schema) - {key})) for key in variable_schema}
        elif variable_schema.__class__ == dict:
            self_impute_attempt = {key:(key in value) for key,value in variable_schema.items()}
            if any(self_impute_attempt.values()):
                self_impute_vars = [key for key,value in self_impute_attempt.items() if value]
                raise ValueError(','.join(self_impute_vars) + ' variables cannot be used to impute itself.')
        else:
            raise ValueError('variable_schema type not recognized.')

        # Remove any variables from imputation dict that have no missing values.
        not_imputable = set(variable_schema) & set(na_counts[na_counts == 0].keys())
        for rnm in not_imputable:
            del variable_schema[rnm]

        # Store values pertaining to variable schema
        self.variable_schema = variable_schema
        self.predictor_vars = _distinct_from_list(concatenate([value for key,value in variable_schema.items()]))
        self.response_vars = list(variable_schema)
        self.all_vars = _distinct_from_list(self.predictor_vars + self.response_vars)
        self.n_imputed_vars = len(variable_schema)
        self.not_imputable = not_imputable
        self.static_predictors = list(set(self.predictor_vars) & \
                                     (set(self.vars_with_any_missing) -
                                      set(self.response_vars))
                                      )
        self.static_predictors.sort()

        if verbose & len(self.static_predictors) > 0:
            print('<' + ','.join(self.static_predictors) + '> ' +
                  'Are used as predictors but not being imputed.\n' +
                  'This is not recommended, since these variables\n' +
                  'will remain static throughout the process after\n' +
                  'having been imputed with random sampling.')

    def get_var_pred_list(self,var):
        return self.variable_schema[var]

    def get_var_cat_preds(self,var):
        pvars = self.get_var_pred_list(var)
        catpreds = list(set(self.categorical_variables) & set(pvars))
        catpreds.sort()
        return catpreds
