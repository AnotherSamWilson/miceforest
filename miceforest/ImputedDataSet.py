from .ImputationSchema import _ImputationSchema
import numpy as np
from pandas import DataFrame
from .utils import (
    ensure_rng,
    _distinct_from_list,
    _copy_and_remove,
    _list_union,
    _var_comparison,
)
from typing import Optional, Union, List, Dict, Callable


class ImputedDataSet(_ImputationSchema):
    """
    Imputed Data Set

    This class should not be instantiated directly.
    Instead, use derived method MultipleImputedKernel.

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
        The number of mean matching candidates to use. Mean matching
        allows the process to impute more realistic values.
        Candidates are _always_ drawn from a kernel dataset.
        Mean matching follows the following rules based on variable type:
            Categorical:
                If mmc = 0, the predicted class is used. If mmc > 0, return
                class based on random draw weighted by class probability
                for each sample.
            Numeric:
                If mmc = 0, the predicted value is used. If mmc > 0, obtain
                the mmc closest candidate predictions and collect the associated
                real candidate values. Choose 1 randomly.

        For more information, see:
        https://github.com/AnotherSamWilson/miceforest#Predictive-Mean-Matching

    save_all_iterations: boolean, optional(default=False)
        Save all iterations that have been imputed, or just the latest.
        Saving all iterations allows for additional plotting,
        but may take more memory

    random_state: None,int, or numpy.random.RandomState
        Ensures a random state throughout the process
        
    initial_imputation: None, str
        Allows user to specify how to impute at initialization (ie 0th imputation)
        Categorical:
            - if not None the mode is used 
        Numeric:
            - 'mean', 'mode' or 'median'
        ##TODO add callble that returns single value eg just like np.mean or user specified

    """

    def __init__(
        self,
        data: DataFrame,
        variable_schema: Union[List[str], Dict[str, List[str]]] = None,
        mean_match_candidates: Union[int, Dict[str, int]] = None,
        save_all_iterations: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
        initial_imputation: Union[str, dict, Callable] = None, 
    ):

        super().__init__(
            variable_schema=variable_schema,
            mean_match_candidates=mean_match_candidates,
            validation_data=data,
        )

        self._random_state = ensure_rng(random_state)

        self.data = data
        self.save_all_iterations = save_all_iterations
        self.categorical_variables = list(
            self.data_dtypes[self.data_dtypes == "category"].keys()
        )

        # Right now variables are filled in with random draws
        # from their original distribution. Add options in the future.
        self.imputation_values: Dict[str, Dict] = {
            var: dict() for var in self._all_imputed_vars
        }
        # added custom initial imputation 
        for var in self._all_imputed_vars:
            if initial_imputation is None:
                self.imputation_values[var] = {
                    0: self._random_state.choice(
                        data[var].dropna(), size=self.na_counts[var]
                        )
                    }
                
            elif initial_imputation in ['mean', 'median', 'mode']:
                # right now impute numerical with initial_imputation options -> mean, mode, median 
                # and categorical with mode 
                if var in self.categorical_variables:
                    self._initial_impute(data, var, 'mode')
                    
                if var not in self.categorical_variables:
                    self._initial_impute(data, var, initial_imputation)
            else:
                raise NotImplementedError('Currently this initial imputation is not implemented')
                
                    
                
    def _initial_impute(self, data, var, imputation_method):
        """ Applies intial imputation method
        ### TODO accept any function transforms pandas series/array to single value
        """
        self.imputation_values[var] = {
                            0: np.array([getattr(data[var] , imputation_method)()] * self.na_counts[var]).flatten()
                            }
        
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
  Imputed Variables: {self.n_imputed_vars}
save_all_iterations: {self.save_all_iterations}"""
        return summary_string

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

        # If var is None, we want the meta iteration level. This must fail
        # if called inside iteration updates, which would mean certain variables
        # have different numbers of iterations.
        if var is None:
            var_iterations = [
                np.max(list(itr))
                for var, itr in self.imputation_values.items()
                if var in self.response_vars
            ]
            distinct_iterations = _distinct_from_list(var_iterations)
            if len(distinct_iterations) > 1:
                raise ValueError(
                    "Inconsistent state - cannot get meta iteration count."
                )
            else:
                return next(iter(distinct_iterations))
        else:
            # Extract the number of iterations so far for a specific dataset, variable
            return np.max(list(self.imputation_values[var]))

    def _default_iteration(self, iteration: Optional[int], **kwargs) -> int:
        """
        If iteration is not specified it is assumed to
        be the last iteration run in many cases.
        """
        if iteration is None:
            return self.iteration_count(**kwargs)
        else:
            return iteration

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
        num_vars = [v for v in self.data.columns if v not in self.categorical_variables]
        num_vars = self._varfilter(vrs=num_vars, response=response, predictor=predictor)
        return num_vars

    def _make_xy(self, var: str, iteration: int = None):
        """
        Make the predictor and response set used to train the model.
        Must be defined in ImputedDataSet because this method is called
        directly in KernelDataSet.impute_new_data()

        If iteration is None, it returns the most up-to-date imputations
        for each variable.
        """
        xvars = self.variable_schema[var]
        completed_data = self.complete_data(iteration=iteration, all_vars=True)
        to_convert = _list_union(self.categorical_variables, xvars)
        for ctc in to_convert:
            completed_data[ctc] = completed_data[ctc].cat.codes
        x = completed_data[xvars]
        y = completed_data[var]
        return x, y

    def _insert_new_data(self, var: str, new_data: np.ndarray):
        current_iter = self.iteration_count(var)
        if not self.save_all_iterations:
            del self[var, current_iter]
        self[var, current_iter + 1] = new_data

    def complete_data(self, iteration: int = None, all_vars: bool = False) -> DataFrame:
        """
        Replace missing values with imputed values.

        Parameters
        ----------
        iteration: int
            The iteration to return.
            If None, returns the most up-to-date iterations,
            even if different between variables.
            If not none, iteration must have been saved in
            imputed values.
        all_vars: bool
            Should all variables in the imputation schema be
            imputed, or just the ones specified to be imputed?

        Returns
        -------
        pandas DataFrame
            The completed data

        """
        imputed_dataframe = self.data.copy()

        # Need to impute all variables used in variable_schema if we are running model
        # Just impute specified variables if the user wants it.
        ret_vars = self._all_imputed_vars if all_vars else self.response_vars

        for var in ret_vars:
            itrn = self._default_iteration(iteration=iteration, var=var)
            imputed_dataframe.loc[self.na_where[var], var] = self[var, itrn]
        return imputed_dataframe

    def _cross_check_numeric(self, variables: Optional[List[str]]) -> List[str]:

        numeric_imputed_vars = _copy_and_remove(variables, self.categorical_variables)

        if variables is None:
            variables = numeric_imputed_vars
        else:
            if any([var not in numeric_imputed_vars for var in variables]):
                raise ValueError(
                    "Specified variable is not in imputed numeric variables."
                )

        return variables

    def get_means(self, variables: List[str] = None):
        """
        Return a dict containing the average imputation value
        for specified variables at each iteration.
        """
        num_vars = self._get_num_vars()
        variables = _var_comparison(variables, num_vars)

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
            for var in variables
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
        if self.iteration_count() < 2 or not self.save_all_iterations:
            raise ValueError("There is only one iteration.")

        num_vars = self._get_num_vars()
        if variables is None:
            variables = num_vars
        elif any([v not in num_vars for v in variables]):
            raise ValueError("variables were either not numeric or not imputed.")

        mean_dict = self.get_means(variables=variables)

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

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

    def _prep_multi_plot(
        self,
        variables: List[str],
    ):
        plots = len(variables)
        plotrows, plotcols = int(np.ceil(np.sqrt(plots))), int(
            np.ceil(plots / np.ceil(np.sqrt(plots)))
        )
        return plots, plotrows, plotcols

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
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        iteration = self._default_iteration(iteration)

        num_vars = self._get_num_vars()
        if variables is None:
            variables = num_vars
        elif any([v not in num_vars for v in variables]):
            raise ValueError("variables were either not numeric or not imputed.")

        plots, plotrows, plotcols = self._prep_multi_plot(variables)
        gs = gridspec.GridSpec(plotrows, plotcols)
        fig, ax = plt.subplots(plotrows, plotcols, squeeze=False)

        for v in range(plots):
            var = variables[v]
            axr, axc = next(iter(gs[v].rowspan)), next(iter(gs[v].colspan))
            plt.sca(ax[axr, axc])
            ax[axr, axc] = sns.kdeplot(
                self.data[var].dropna(), color="red", linewidth=2
            )
            ax[axr, axc] = sns.kdeplot(self[var, iteration], color="black", linewidth=1)

        plt.subplots_adjust(**adj_args)
