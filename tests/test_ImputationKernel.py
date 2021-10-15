

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import miceforest as mf
from datetime import datetime
from miceforest.mean_matching_functions import mean_match_kdtree_classification
from matplotlib.pyplot import close

# Make random state and load data
# Define data
random_state = np.random.RandomState(5)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston.columns = [str(i) for i in boston.columns]
boston["3"] = boston["3"].astype("category")
boston["8"] = boston["8"].astype("category")
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)

def test_defaults_pandas():

    new_data = boston_amp.loc[range(10), :].copy()

    s = datetime.now()
    kernel = mf.ImputationKernel(
        data=boston_amp,
        datasets=3
    )

    kernel.mice(iterations=2)
    kernel.complete_data(0, inplace=True)
    assert all(kernel.working_data.isnull().sum() == 0)
    assert kernel.models[0][0][2].params['objective'] == 'regression'
    assert kernel.models[0][3][2].params['objective'] == 'binary'
    assert kernel.models[0][8][2].params['objective'] == 'multiclass'

    # Make sure we didn't touch the original data
    assert all(boston_amp.isnull().sum() > 0)

    imp_ds = kernel.impute_new_data(new_data)
    imp_ds.complete_data(0,inplace=True)
    assert all(imp_ds.working_data.isnull().sum(0) == 0)
    assert new_data.isnull().sum().sum() > 0
    print(datetime.now() - s)


def test_complex_pandas():
    
    working_set = boston_amp.copy()

    # Switch our category columns to integer codes.
    # Replace -1 with np.NaN or lightgbm will complain.
    working_set["3"] = working_set["3"].cat.codes
    working_set["8"] = working_set["8"].cat.codes
    working_set["3"].replace(-1,np.NaN, inplace=True)
    working_set["8"].replace(-1, np.NaN, inplace=True)
    new_data = working_set.loc[range(10), :].copy()

    # Customize everything.
    vs = {"1": ["2","3","4","5"], "2": ["6","7"], "3": ["1","2","8"]}
    mmc = {"1": 4, "2": 0.01, "3": 0}
    ds = {"2": 100, "3": 0.5}

    imputed_var_names = list(vs)
    non_imputed_var_names = [str(x) for x in range(13) if str(x) not in vs]

    def mmf(
            mmc,
            model,
            candidate_features,
            bachelor_features,
            candidate_values,
            random_state
    ):
        if mmc > 0:
            imp_values = random_state.choice(candidate_values, size=bachelor_features.shape[0])
        else:
            bachelor_preds = model.predict(bachelor_features)
            imp_values = bachelor_preds
        return imp_values

    kernel = mf.ImputationKernel(
        data=working_set,
        datasets=3,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_function=mmf,
        categorical_feature=[3,8],
        copy_data=False
    )

    assert kernel.mean_match_candidates == {1: 4, 2: 3, 3: 0}, "mean_match_candidates initialization failed"
    assert kernel.data_subset == {1: 380, 2: 100, 3: 190}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [3, 8], "categorical recognition failed."

    nround = 2
    kernel.mice(nround - 1, variable_parameters={"1": {"n_estimators": 15}}, n_estimators=10, verbose=True)
    assert kernel.models[0][1][nround - 1].params['num_iterations'] == 15
    assert kernel.models[0][2][nround - 1].params['num_iterations'] == 10
    kernel.mice(1, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10, verbose=True)
    assert kernel.iteration_count() == nround, "iteration counting is incorrect."
    assert kernel.models[0][1][nround].params['num_iterations'] == 15
    assert kernel.models[0][2][nround].params['num_iterations'] == 10

    # Make sure we only impute variables in variable_schema
    compdat = kernel.complete_data(0)
    assert all(compdat[imputed_var_names].isnull().sum() == 0)
    assert all(compdat[non_imputed_var_names].isnull().sum() > 0)

    # Test the ability to tune parameters with custom setup
    optimization_steps = 2
    op, ol = kernel.tune_parameters(
        dataset=0,
        optimization_steps=optimization_steps,
        variable_parameters={1: {"bagging_fraction": 0.9, "feature_fraction_bynode": (0.85, 0.9)}},
        bagging_fraction=0.8,
        feature_fraction_bynode=(0.70,0.75),
        verbose=True
    )
    assert op[1]["bagging_fraction"] == 0.9
    assert op[2]["bagging_fraction"] == 0.8
    assert (op[1]["feature_fraction_bynode"] >= 0.85) and (op[1]["feature_fraction_bynode"] <= 0.9)
    assert (op[2]["feature_fraction_bynode"] >= 0.70) and (op[2]["feature_fraction_bynode"] <= 0.75)
    kernel.mice(1, variable_parameters=op, verbose=True)
    model_2_params = kernel.models[0][2][nround + 1].params
    model_1_params = kernel.models[0][1][nround + 1].params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)

    new_imp_dat = kernel.impute_new_data(new_data=new_data, verbose=True)
    new_imp_complete = new_imp_dat.complete_data(0)
    assert all(new_imp_complete[["1","2","3"]].isnull().sum() == 0)

    # Plotting on multiple imputed dataset
    new_imp_dat.plot_mean_convergence()
    close()
    new_imp_dat.plot_imputed_distributions()
    close()

    # Plotting on Multiple Imputed Kernel
    kernel.plot_feature_importance(0)
    close()
    kernel.plot_mean_convergence()
    close()
    kernel.plot_imputed_distributions()
    close()



def test_defaults_numpy():
    
    working_set = boston_amp.copy()

    working_set["3"] = working_set["3"].cat.codes
    working_set["8"] = working_set["8"].cat.codes
    working_set["3"].replace(-1,np.NaN, inplace=True)
    working_set["8"].replace(-1, np.NaN, inplace=True)
    new_data = working_set.loc[range(10), :].copy()
    working_set = working_set.values
    new_data = new_data.values

    s = datetime.now()
    kernel = mf.ImputationKernel(
        data=working_set,
        datasets=3,
        categorical_feature=[3,8],
        mean_match_function=mean_match_kdtree_classification
    )

    kernel.mice(iterations=1, verbose=True)

    # Complete data with copy.
    comp_dat = kernel.complete_data(0, inplace=False)

    # We didn't complete data in place. Make sure we created
    # a copy, and did not affect internal data or original data.
    assert all(np.isnan(comp_dat).sum(0) == 0)
    assert all(np.isnan(kernel.working_data).sum(0) > 0)
    assert all(np.isnan(working_set).sum(0) > 0)

    # Complete data in place
    kernel.complete_data(0, inplace=True)

    # We completed data in place. Make sure we only affected
    # the kernel.working_data and not the original data.
    assert all(np.isnan(kernel.working_data).sum(0) == 0)
    assert all(np.isnan(working_set).sum(0) > 0)



    imp_ds = kernel.impute_new_data(new_data)
    imp_ds.complete_data(0,inplace=True)
    assert all(np.isnan(imp_ds.working_data).sum(0) == 0)
    assert np.isnan(new_data).sum() > 0
    print(datetime.now() - s)


def test_complex_numpy():

    working_set = boston_amp.copy()

    # Switch our category columns to integer codes.
    # Replace -1 with np.NaN or lightgbm will complain.
    working_set["3"] = working_set["3"].cat.codes
    working_set["8"] = working_set["8"].cat.codes
    working_set["3"].replace(-1,np.NaN, inplace=True)
    working_set["8"].replace(-1, np.NaN, inplace=True)
    new_data = working_set.loc[range(100), :].copy()

    working_set = working_set.values
    new_data = new_data.values

    # Customize everything.
    vs = {1: [2,3,4,5], 2: [6,7], 3: [1,2,8]}
    mmc = {1: 4, 2: 0.01, 3: 0}
    ds = {2: 100, 3: 0.5}

    imputed_var_names = list(vs)
    non_imputed_var_names = [x for x in range(13) if x not in vs]

    def mmf(
            mmc,
            model,
            candidate_features,
            bachelor_features,
            candidate_values,
            random_state
    ):
        if mmc > 0:
            imp_values = random_state.choice(candidate_values, size=bachelor_features.shape[0])
        else:
            bachelor_preds = model.predict(bachelor_features)
            imp_values = bachelor_preds
        return imp_values

    kernel = mf.ImputationKernel(
        data=working_set,
        datasets=3,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_function=mmf,
        categorical_feature=[3,8],
        copy_data=False
    )

    assert kernel.mean_match_candidates == {1: 4, 2: 3, 3: 0}, "mean_match_candidates initialization failed"
    assert kernel.data_subset == {1: 380, 2: 100, 3: 190}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [3, 8], "categorical recognition failed."

    nround = 2
    kernel.mice(nround - 1, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10, verbose=True)
    assert kernel.models[0][1][nround - 1].params['num_iterations'] == 15
    assert kernel.models[0][2][nround - 1].params['num_iterations'] == 10
    kernel.mice(1, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10, verbose=True)
    assert kernel.iteration_count() == nround, "iteration counting is incorrect."
    assert kernel.models[0][1][nround].params['num_iterations'] == 15
    assert kernel.models[0][2][nround].params['num_iterations'] == 10

    # Complete data with copy. Make sure only correct datasets and variables were affected.
    compdat = kernel.complete_data(0, inplace=False)
    assert all(np.isnan(compdat[:,imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(compdat[:,non_imputed_var_names]).sum(0) > 0)

    # Should have no affect on working_data
    assert all(np.isnan(kernel.working_data).sum(0) > 0)

    # Should have no affect on working_set
    assert all(np.isnan(working_set).sum(0) > 0)

    # Now complete the data in place
    kernel.complete_data(0, inplace=True)

    # Should have affect on working_data and original data
    assert all(np.isnan(kernel.working_data[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(working_set[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(kernel.working_data[:, non_imputed_var_names]).sum(0) > 0)
    assert all(np.isnan(working_set[:, non_imputed_var_names]).sum(0) > 0)

    # Test the ability to tune parameters with custom setup
    optimization_steps = 2
    op, ol = kernel.tune_parameters(
        dataset=0,
        optimization_steps=optimization_steps,
        variable_parameters={1: {"bagging_fraction": 0.9, "feature_fraction_bynode": (0.85, 0.9)}},
        bagging_fraction=0.8,
        feature_fraction_bynode=(0.70,0.75),
        verbose=True
    )
    assert op[1]["bagging_fraction"] == 0.9
    assert op[2]["bagging_fraction"] == 0.8
    assert (op[1]["feature_fraction_bynode"] >= 0.85) and (op[1]["feature_fraction_bynode"] <= 0.9)
    assert (op[2]["feature_fraction_bynode"] >= 0.70) and (op[2]["feature_fraction_bynode"] <= 0.75)
    kernel.mice(1, variable_parameters=op, verbose=True)
    model_2_params = kernel.models[0][2][nround + 1].params
    model_1_params = kernel.models[0][1][nround + 1].params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)

    new_imp_dat = kernel.impute_new_data(new_data=new_data, copy_data=True, verbose=True)

    # Not in place
    new_imp_complete = new_imp_dat.complete_data(0, inplace=False)
    assert all(np.isnan(new_imp_complete[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(new_imp_complete[:, non_imputed_var_names]).sum(0) > 0)

    # Should have no affect on working_data or original data
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)
    assert all(np.isnan(new_data[:, imputed_var_names]).sum(0) > 0)

    # complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_imp_dat.working_data[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(new_data[:, non_imputed_var_names]).sum(0) > 0)

    # Alter in place
    new_imp_dat = kernel.impute_new_data(new_data=new_data, copy_data=False, verbose=True)

    # Before completion, nan's should still exist in data:
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data not in place
    new_imp_complete = new_imp_dat.complete_data(0, inplace=False)
    assert all(np.isnan(new_imp_complete[:, non_imputed_var_names]).sum(0) > 0)
    assert all(np.isnan(new_imp_complete[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_data[:, non_imputed_var_names]).sum(0) > 0)
    assert all(np.isnan(new_data[:, imputed_var_names]).sum(0) == 0)
    assert all(np.isnan(new_imp_dat.working_data[:, non_imputed_var_names]).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data[:, imputed_var_names]).sum(0) == 0)


    # Plotting on multiple imputed dataset
    new_imp_dat.plot_mean_convergence()
    close()
    new_imp_dat.plot_imputed_distributions()
    close()

    # Plotting on Multiple Imputed Kernel
    kernel.plot_feature_importance(0)
    close()
    kernel.plot_mean_convergence()
    close()
    kernel.plot_imputed_distributions()
    close()