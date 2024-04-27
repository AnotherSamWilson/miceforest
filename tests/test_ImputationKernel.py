
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf
from datetime import datetime
from matplotlib.pyplot import close
from tempfile import mkstemp

# Make random state and load data
# Define data
random_state = np.random.RandomState(1991)
iris = pd.concat(load_iris(as_frame=True, return_X_y=True), axis=1)
iris['sp'] = iris['target'].astype('category')
del iris['target']
iris.rename({
    'sepal length (cm)': 'sl',
    'sepal width (cm)': 'ws',
    'petal length (cm)': 'pl',
    'petal width (cm)': 'pw',
}, axis=1, inplace=True)
iris['bc'] = pd.Series(np.random.binomial(n=1, p=0.5, size=150)).astype('category')
iris_amp = mf.ampute_data(iris, perc=0.25, random_state=random_state)


def test_defaults_pandas():

    new_data = iris_amp.loc[range(10), :].copy()

    kernel = mf.ImputationKernel(
        data=iris_amp,
        datasets=2,
        save_models=1
    )
    kernel.mice(iterations=2, compile_candidates=True, verbose=True)

    kernel2 = mf.ImputationKernel(
        data=iris_amp,
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
    assert kernel.get_model(0, 'bc', 3).params['objective'] == 'binary'
    assert kernel.get_model(0, 'sp', 3).params['objective'] == 'multiclass'

    # Make sure we didn't touch the original data
    assert all(iris_amp.isnull().sum() > 0)

    imp_ds = kernel.impute_new_data(new_data, verbose=True)
    imp_ds.complete_data(2,inplace=True)
    assert all(imp_ds.working_data.isnull().sum(0) == 0)
    assert new_data.isnull().sum().sum() > 0

    # Make sure fully-recognized data can be passed through with no changes
    imp_fr = kernel.impute_new_data(iris)
    comp_fr = imp_fr.complete_data(0)
    assert np.all(comp_fr == iris), "values of fully-recognized data were modified"
    assert imp_fr.iteration_count() == -1

    # Make sure single rows can be imputed
    single_row = new_data.iloc[[0], :]
    imp_sr = kernel.impute_new_data(single_row)
    assert np.all(imp_sr.complete_data(0).dtypes == single_row.dtypes)


def test_complex_pandas():
    
    working_set = iris_amp.copy()
    new_data = working_set.loc[range(10), :].copy()

    # Customize everything.
    vs = {
        'sl': ['ws', 'pl', 'pw', 'sp', 'bc'],
        'ws': ['sl'],
        'pl': ['sp', 'bc'],
        'sp': ['sl', 'ws', 'pl', 'pw', 'bc'],
        'pw': ['sl', 'ws', 'pl', 'sp', 'bc'],
    }
    mmc = {"sl": 4, 'ws': 0.01, "pl": 0}
    ds = {"sl": 100, "ws": 0.5}
    io = ['pw', 'pl', 'ws', 'sl']

    imputed_var_names = io
    non_imputed_var_names = [c for c in iris_amp if c not in imputed_var_names]

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
        categorical_feature='auto',
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
        categorical_feature='auto',
        copy_data=False
    )
    new_file, filename = mkstemp()
    kernel2.save_kernel(filename)
    kernel2 = mf.utils.load_kernel(filename)

    assert kernel.data_subset == {0: 100, 1: 56, 3: 113, 2: 113, 4: 113}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [4, 5], "categorical recognition failed."

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
        variable_parameters={"sl": {"n_iter": 15}},
        num_trees=10,
        verbose=True
    )
    kernel.compile_candidate_preds()
    kernel2.mice(nround - 1, variable_parameters={"sl": {"n_estimators": 15}}, n_estimators=10, verbose=True)
    kernel.append(kernel2)
    kernel.compile_candidate_preds()
    assert kernel.get_model(0, 0, nround - 1).num_trees() == 15
    assert kernel.get_model(0, 1, nround - 1).num_trees() == 10
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
    assert all(new_imp_complete[imputed_var_names].isnull().sum() == 0)

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
    
    iris_np = iris.copy()
    iris_np["sp"] = iris_np["sp"].cat.codes
    iris_np = iris_np.values
    iris_np_amp = mf.ampute_data(iris_np, perc=0.25)
    new_data = iris_np_amp[range(10), :].copy()

    s = datetime.now()
    kernel = mf.ImputationKernel(
        data=iris_np_amp,
        datasets=3,
        categorical_feature=[4],
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
    assert all(np.isnan(iris_np_amp).sum(0) > 0)

    # Complete data in place
    kernel.complete_data(0, inplace=True)

    # We completed data in place. Make sure we only affected
    # the kernel.working_data and not the original data.
    assert all(np.isnan(kernel.working_data).sum(0) == 0)
    assert all(np.isnan(iris_np_amp).sum(0) > 0)

    imp_ds = kernel.impute_new_data(new_data)
    imp_ds.complete_data(0,inplace=True)
    assert all(np.isnan(imp_ds.working_data).sum(0) == 0)
    assert np.isnan(new_data).sum() > 0

    # Make sure fully-recognized data can be passed through with no changes
    imp_fr = kernel.impute_new_data(iris_np)
    comp_fr = imp_fr.complete_data(0)
    assert np.all(comp_fr == iris_np), "values of fully-recognized data were modified"
    assert imp_fr.iteration_count() == -1


def test_complex_numpy():

    iris_np = iris.copy()
    iris_np["sp"] = iris_np["sp"].cat.codes
    iris_np = iris_np.values
    iris_np_amp = mf.ampute_data(iris_np, perc=0.25)
    new_data = iris_np_amp[range(25), :].copy()

    # Specify that models should be built for variables 1, 2, 3, 4
    # Customize everything.
    vs = {
        0: [1, 2, 3, 4, 5],
        1: [0],
        2: [4, 5],
        3: [0, 1, 2, 4, 5],
        4: [0, 1, 2, 3, 5],
    }
    mmc = {0: 4, 1: 0.1, 2: 0}
    ds = {0: 100, 1: 0.5}
    io = [0, 1, 2, 3, 4]
    niv = [v for v in range(iris_amp.shape[1]) if v not in list(vs)]

    mmfc = mean_match_fast_cat.copy()
    mmfc.set_mean_match_candidates(mean_match_candidates=mmc)
    kernel = mf.ImputationKernel(
        data=iris_np_amp,
        datasets=2,
        variable_schema=vs,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        mean_match_scheme=mmfc,
        categorical_feature=[4],
        copy_data=False,
        save_loggers=True
    )

    kernel2 = mf.ImputationKernel(
        data=iris_np_amp,
        datasets=1,
        variable_schema=vs,
        imputation_order=io,
        train_nonmissing=True,
        data_subset=ds,
        mean_match_scheme=mmfc.copy(),
        categorical_feature=[4],
        copy_data=False,
        save_loggers=True
    )
    new_file, filename = mkstemp()
    kernel2.save_kernel(filename)
    kernel2 = mf.utils.load_kernel(filename)

    assert kernel.data_subset == {0: 100, 1: 56, 2: 113, 3: 113, 4: 113}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [4], "categorical recognition failed."

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
    assert all(np.isnan(iris_np_amp).sum(0) > 0)

    # Now complete the data in place
    kernel.complete_data(0, inplace=True)

    # Should have affect on working_data and original data
    assert all(np.isnan(kernel.working_data[:, io]).sum(0) == 0)
    assert all(np.isnan(iris_np_amp[:, io]).sum(0) == 0)
    assert all(np.isnan(kernel.working_data[:, niv]).sum(0) > 0)
    assert all(np.isnan(iris_np_amp[:, niv]).sum(0) > 0)

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
    assert all(np.isnan(new_imp_complete[:, niv]).sum(0) > 0)


    # Should have no affect on working_data or original data
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)
    assert all(np.isnan(new_data[:, list(vs)]).sum(0) > 0)

    # complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_imp_dat.working_data[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_data[:, niv]).sum(0) > 0)

    # Alter in place
    new_imp_dat = kernel.impute_new_data(new_data=new_data, copy_data=False, verbose=True)

    # Before completion, nan's should still exist in data:
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data not in place
    new_imp_complete = new_imp_dat.complete_data(0, inplace=False)
    assert all(np.isnan(new_imp_complete[:, niv]).sum(0) > 0)
    assert all(np.isnan(new_imp_complete[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_data).sum(0) > 0)
    assert all(np.isnan(new_imp_dat.working_data).sum(0) > 0)

    # Complete data in place
    new_imp_dat.complete_data(0, inplace=True)
    assert all(np.isnan(new_data[:, niv]).sum(0) > 0)
    assert all(np.isnan(new_data[:, list(vs)]).sum(0) == 0)
    assert all(np.isnan(new_imp_dat.working_data[:, niv]).sum(0) > 0)
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
