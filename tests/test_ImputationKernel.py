
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf
from datetime import datetime
from matplotlib.pyplot import close
from tempfile import mkstemp
import dill

# Make random state and load data
# Define data
random_state = np.random.RandomState(1991)
iris = pd.concat(load_iris(as_frame=True, return_X_y=True), axis=1)
# iris = iris.sample(100000, replace=True)
iris['sp'] = iris['target'].map({0: 'Category1', 1: 'Category2', 2: 'Category3'}).astype('category')
del iris['target']
iris.rename({
    'sepal length (cm)': 'sl',
    'sepal width (cm)': 'ws',
    'petal length (cm)': 'pl',
    'petal width (cm)': 'pw',
}, axis=1, inplace=True)
iris['bi'] = pd.Series(np.random.binomial(n=1, p=0.5, size=iris.shape[0])).map({0: 'FOO', 1: 'BAR'}).astype('category')
iris['ui8'] = iris['sl'].round(0).astype('UInt8')
iris['ws'] = iris['ws'].astype('float32')
iris.reset_index(drop=True, inplace=True)
amputed_variables = ['sl', 'ws', 'pl', 'sp', 'bi', 'ui8']
iris_amp = mf.ampute_data(
    iris, 
    variables=amputed_variables,
    perc=0.25, 
    random_state=random_state
)

new_amputed_data = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
new_nonmissing_data = iris.loc[range(20), :].reset_index(drop=True).copy()

# Make special datasets that have weird edge cases
# Multiple columns with all missing values
# sp is categorical, and pw had no missing 
# values in the original kernel data
new_amputed_data_special_1 = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
for col in ['sp', 'pw']:
    new_amputed_data_special_1[col] = np.nan
    dtype = iris[col].dtype
    new_amputed_data_special_1[col] = new_amputed_data_special_1[col].astype(dtype)

# Some columns with no missing values
new_amputed_data_special_2 = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
new_amputed_data_special_2[['sp', 'ui8']] = iris.loc[range(20), ['sp', 'ui8']]


def make_and_test_kernel(**kwargs):

    kwargs = {
        'data':iris_amp,
        'num_datasets':2,
        'variable_schema':vs,
        'mean_match_candidates':mmc,
        'data_subset':ds,
        'mean_match_strategy':'normal',
        'save_all_iterations_data':True,
    }

    # Build a normal kernel, run mice, save, load, and run mice again
    kernel = mf.ImputationKernel(**kwargs)
    assert kernel.iteration_count() == 0
    kernel.mice(iterations=2, verbose=True)
    assert kernel.iteration_count() == 2
    new_file, filename = mkstemp()
    with open(filename, 'wb') as file:
        dill.dump(kernel, file)
    del kernel
    with open(filename, 'rb') as file:
        kernel = dill.load(file)
    kernel.mice(iterations=1, verbose=True)
    assert kernel.iteration_count() == 3

    modeled_variables = kernel.model_training_order
    imputed_variables = kernel.imputed_variables

    # pw has no missing values.
    assert 'pw' not in imputed_variables
    
    # Make a completed dataset
    completed_data = kernel.complete_data(dataset=0, inplace=False)

    # Make sure the data was imputed
    assert all(completed_data[imputed_variables].isnull().sum() == 0)

    # Make sure the dtypes didn't change
    for col, series in iris_amp.items():
        dtype = series.dtype
        assert completed_data[col].dtype == dtype

    # Make sure the working data wasn't imputed
    assert all(kernel.working_data[imputed_variables].isnull().sum() > 0)

    # Impute the data in place now
    kernel.complete_data(0, inplace=True)

    # Assert we actually imputed the working data
    assert all(kernel.working_data[imputed_variables].isnull().sum() == 0)

    # Assert the original data was not touched
    assert all(iris_amp[imputed_variables].isnull().sum() > 0)

    # Make sure the models were trained the way we expect
    for variable in modeled_variables:
        if variable == 'sp':
            objective = 'multiclass'
        elif variable == 'bi':
            objective = 'binary'
        else:
            objective = 'regression'
        assert kernel.get_model(variable=variable, dataset=0, iteration=1).params['objective'] == objective
        assert kernel.get_model(variable=variable, dataset=0, iteration=2).params['objective'] == objective
        assert kernel.get_model(variable=variable, dataset=1, iteration=1).params['objective'] == objective
        assert kernel.get_model(variable=variable, dataset=1, iteration=2).params['objective'] == objective

    # Impute a new dataset, and complete the data
    imputed_new_data = kernel.impute_new_data(new_amputed_data, verbose=True)
    imputed_dataset_0 = imputed_new_data.complete_data(dataset=0, iteration=2, inplace=False)
    imputed_dataset_1 = imputed_new_data.complete_data(dataset=1, iteration=2, inplace=False)

    # Assert we didn't just impute the same thing for all values
    assert not np.all(imputed_dataset_0 == imputed_dataset_1)

    # Make sure we can impute the special cases
    imputed_data_special_1 = kernel.impute_new_data(new_amputed_data_special_1)
    imputed_data_special_2 = kernel.impute_new_data(new_amputed_data_special_2)
    imputed_dataset_special_1 = imputed_data_special_1.complete_data(0)
    imputed_dataset_special_2 = imputed_data_special_2.complete_data(0)
    assert not np.any(imputed_dataset_special_1[modeled_variables].isnull())
    assert not np.any(imputed_dataset_special_2[modeled_variables].isnull())

    return kernel


def test_defaults():

    kernel_normal = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy='normal',
        save_all_iterations_data=True,
    )
    kernel_fast = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy='fast',
        save_all_iterations_data=True,
    )
    kernel_shap = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy='shap',
        save_all_iterations_data=True,
    )


def test_complex():
    
    # Customize everything.
    vs = {
        'sl': ['ws', 'pl', 'pw', 'sp', 'bi'],
        'ws': ['sl'],
        'pl': ['sp', 'bi'],
        # 'sp': ['sl', 'ws', 'pl', 'pw', 'bc'], # Purposely don't train a variable that does have missing values
        'pw': ['sl', 'ws', 'pl', 'sp', 'bi'],
        'bi': ['ws', 'pl', 'sp'],
        'ui8': ['sp', 'ws'],
    }
    mmc = {"sl": 4, 'ws': 0, "bi": 5}
    ds = {"sl": int(iris_amp.shape[0] / 2), "ws": 50}

    imputed_var_names = list(vs)
    non_imputed_var_names = [c for c in iris_amp if c not in imputed_var_names]

    # Build a normal kernel, run mice, save, load, and run mice again
    kernel = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_strategy='normal',
        save_all_iterations_data=True,
    )
    normal_ind = kernel_normal.impute_new_data(new_data)
    
    kernel_fast = mf.ImputationKernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_strategy='fast',
        save_all_iterations_data=True,
    )
    kernel_fast.mice(iterations=2, verbose=True)
    new_file, filename = mkstemp()
    with open(filename, 'wb') as file:
        dill.dump(kernel_fast, file)
    del kernel_fast
    with open(filename, 'rb') as file:
        kernel_fast = dill.load(file)
    kernel_fast.mice(iterations=1, verbose=True)
    fast_ind = kernel_fast.impute_new_data(new_data)

    kernel_shap = mf.ImputationKernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_strategy={'sl': 'shap', 'ws': 'fast', 'ui8': 'fast', 'bi': 'normal'},
        save_all_iterations_data=True,
    )
    kernel_shap.mice(iterations=2, verbose=True)
    new_file, filename = mkstemp()
    with open(filename, 'wb') as file:
        dill.dump(kernel_shap, file)
    del kernel_shap
    with open(filename, 'rb') as file:
        kernel_shap = dill.load(file)
    kernel_shap.mice(iterations=1, verbose=True)
    shap_ind = kernel_shap.impute_new_data(new_data)

    kernel_normal.data_subset
    kernel_normal.model_training_order
    kernel_normal.mean_match_candidates
    kernel_normal.modeled_but_not_imputed_variables
    

    assert kernel.data_subset == {0: 100, 1: 56, 3: 113, 2: 113, 4: 113}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"


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
