

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import miceforest as mf
from datetime import datetime
from miceforest import (
    mean_match_fast_cat,
    mean_match_shap
)
from matplotlib.pyplot import close
from tempfile import mkstemp

# Make random state and load data
# Define data
random_state = np.random.RandomState(5)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston.columns = [str(i) for i in boston.columns]
boston["3"] = boston["3"].map({0: 'a', 1: 'b'}).astype('category')
boston["8"] = boston["8"].astype("category")
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)


def test_defaults_pandas():

    new_data = boston_amp.loc[range(10), :].copy()

    kernel = mf.ImputationKernel(
        data=boston_amp,
        datasets=2,
        save_models=1
    )
    kernel.mice(iterations=2, compile_candidates=True, verbose=True)

    kernel2 = mf.ImputationKernel(
        data=boston_amp,
        datasets=1,
        save_models=1
    )
    kernel2.mice(iterations=2)

    # Test appending and then test kernel.
    kernel.append(kernel2)
    kernel.compile_candidate_preds()


    # Test mice after appendage
    kernel.mice(1, verbose=True)

    kernel.complete_data(0, inplace=True)
    assert all(kernel.working_data.isnull().sum() == 0)
    assert kernel.get_model(0, 0, 3).params['objective'] == 'regression'
    assert kernel.get_model(0, 3, 3).params['objective'] == 'binary'
    assert kernel.get_model(0, 8, 3).params['objective'] == 'multiclass'

    # Make sure we didn't touch the original data
    assert all(boston_amp.isnull().sum() > 0)

    imp_ds = kernel.impute_new_data(new_data, verbose=True)
    imp_ds.complete_data(2,inplace=True)
    assert all(imp_ds.working_data.isnull().sum(0) == 0)
    assert new_data.isnull().sum().sum() > 0


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
    vs = {"1": ["2","3","4","5"], "2": ["6","7"], "3": ["1","2","8"], "4": ["8","9","10"]}
    mmc = {"1": 4, "2": 0.01, "3": 0}
    ds = {"2": 100, "3": 0.5}
    io = ["2", "3", "1"]

    imputed_var_names = io
    non_imputed_var_names = [str(x) for x in range(13) if str(x) not in io]

    from miceforest.builtin_mean_match_schemes import mean_match_shap
    mean_match_custom = mean_match_shap.copy()
    mean_match_custom.set_mean_match_candidates(mmc)

    kernel = mf.ImputationKernel(
        data=working_set,
        datasets=2,
        variable_schema=vs,
        mean_match_scheme=mean_match_shap,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        categorical_feature=[3,8],
        copy_data=False
    )
    kernel2 = mf.ImputationKernel(
        data=working_set,
        datasets=1,
        variable_schema=vs,
        mean_match_scheme=mean_match_shap,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        categorical_feature=[3,8],
        copy_data=False
    )
    new_file, filename = mkstemp()
    kernel2.save_kernel(filename)
    kernel2 = mf.utils.load_kernel(filename)

    assert kernel.data_subset == {1: 380, 2: 100, 3: 190, 4: 380}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [3, 8], "categorical recognition failed."

    # This section tests many things:
        # After saving / loading a kernel, and appending 2 kernels together:
            # mice can continue
            # Aliases are fixed, even when different aliases are passed
            # variable specific parameters supercede globally specified parameters
            # The parameters come through the actual model
    nround = 2
    kernel.mice(
        nround - 1,
        compile_candidates=True,
        variable_parameters={"1": {"n_iter": 15}},
        num_trees=10,
        verbose=True
    )
    kernel.compile_candidate_preds()
    kernel2.mice(nround - 1, variable_parameters={"1": {"n_estimators": 15}}, n_estimators=10, verbose=True)
    kernel.append(kernel2)
    kernel.compile_candidate_preds()
    assert kernel.get_model(0, 1, nround - 1).num_trees() == 15
    assert kernel.get_model(0, 2, nround - 1).num_trees() == 10
    kernel.mice(1, variable_parameters={1: {"n_iter": 15}}, num_trees=10, verbose=True)
    assert kernel.iteration_count() == nround, "iteration counting is incorrect."
    assert kernel.get_model(0, 1, nround).num_trees() == 15
    assert kernel.get_model(0, 2, nround).num_trees() == 10

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
    model_2_params = kernel.get_model(0, 2, nround + 1).params
    model_1_params = kernel.get_model(0, 1, nround + 1).params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)

    new_imp_dat = kernel.impute_new_data(new_data=new_data, verbose=True)
    new_imp_complete = new_imp_dat.complete_data(0)
    new_imp_dat.na_where[2]
    new_imp_dat.imputation_values[0,2,3]
    assert all(new_imp_complete[["1","2","3","4"]].isnull().sum() == 0)

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
        mean_match_scheme=mean_match_fast_cat
    )
    kernel.mice(iterations=1, verbose=True)
    kernel.compile_candidate_preds()

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

    # Specify that models should be built for variables 1, 2, 3, 4
    vs = {1: [2,3,4,5], 2: [6,7], 3: [1,2,8], 4: [8,9,10]}
    mmc = {1: 4, 2: 0.01, 3: 0}
    ds = {2: 100, 3: 0.5}
    # Only variables 1, 2, 3 should be imputed using mice.
    io = [2,3,1]
    niv = np.setdiff1d(np.arange(working_set.shape[1]), io)
    nivs = np.setdiff1d(np.arange(working_set.shape[1]), list(vs))


    kernel = mf.ImputationKernel(
        data=working_set,
        datasets=2,
        variable_schema=vs,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        mean_match_scheme=mean_match_fast_cat,
        categorical_feature=[3,8],
        copy_data=False,
        save_loggers=True
    )

    kernel2 = mf.ImputationKernel(
        data=working_set,
        datasets=1,
        variable_schema=vs,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        mean_match_scheme=mean_match_fast_cat,
        categorical_feature=[3,8],
        copy_data=False,
        save_loggers=True
    )
    new_file, filename = mkstemp()
    kernel2.save_kernel(filename)
    kernel2 = mf.utils.load_kernel(filename)

    assert kernel.data_subset == {2: 100, 3: 190, 1: 380, 4: 380}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [3, 8], "categorical recognition failed."

    nround = 2
    kernel.mice(nround - 1, variable_parameters={1: {"n_iter": 15}}, num_trees=10, verbose=True)
    kernel.compile_candidate_preds()
    kernel2.mice(nround - 1, variable_parameters={1: {"n_iter": 15}}, num_trees=10, verbose=True)
    kernel.append(kernel2)
    kernel.compile_candidate_preds()
    assert kernel.get_model(0, 1, nround - 1).num_trees() == 15
    assert kernel.get_model(0, 2, nround - 1).num_trees() == 10
    kernel.mice(1, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10, verbose=True)
    assert kernel.iteration_count() == nround, "iteration counting is incorrect."
    assert kernel.get_model(0, 1, nround).num_trees() == 15
    assert kernel.get_model(0, 2, nround).num_trees() == 10

    # Complete data with copy. Make sure only correct datasets and variables were affected.
    compdat = kernel.complete_data(0, inplace=False)
    assert all(np.isnan(compdat[:,io]).sum(0) == 0)
    assert all(np.isnan(compdat[:,niv]).sum(0) > 0)

    # Should have no affect on working_data
    assert all(np.isnan(kernel.working_data).sum(0) > 0)

    # Should have no affect on working_set
    assert all(np.isnan(working_set).sum(0) > 0)

    # Now complete the data in place
    kernel.complete_data(0, inplace=True)

    # Should have affect on working_data and original data
    assert all(np.isnan(kernel.working_data[:, io]).sum(0) == 0)
    assert all(np.isnan(working_set[:, io]).sum(0) == 0)
    assert all(np.isnan(kernel.working_data[:, niv]).sum(0) > 0)
    assert all(np.isnan(working_set[:, niv]).sum(0) > 0)

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
    model_2_params = kernel.get_model(0, 2, nround + 1).params
    model_1_params = kernel.get_model(0, 1, nround + 1).params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)

    new_imp_dat = kernel.impute_new_data(new_data=new_data, copy_data=True, verbose=True)

    # Not in place
    new_imp_complete = new_imp_dat.complete_data(0, inplace=False)
    assert all(np.isnan(new_imp_complete[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_imp_complete[:, nivs]).sum(0) > 0)


    # Should have no affect on working_data or original data
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)
    assert all(np.isnan(new_data[:, list(vs)]).sum(0) > 0)

    # complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_imp_dat.working_data[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_data[:, nivs]).sum(0) > 0)

    # Alter in place
    new_imp_dat = kernel.impute_new_data(new_data=new_data, copy_data=False, verbose=True)

    # Before completion, nan's should still exist in data:
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data not in place
    new_imp_complete = new_imp_dat.complete_data(0, inplace=False)
    assert all(np.isnan(new_imp_complete[:, nivs]).sum(0) > 0)
    assert all(np.isnan(new_imp_complete[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_data[:, nivs]).sum(0) > 0)
    assert all(np.isnan(new_data[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_imp_dat.working_data[:, nivs]).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data[:, list(vs)]).sum(0) == 0)


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
