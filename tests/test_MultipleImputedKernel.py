import pytest
import miceforest as mf
from miceforest.utils import _get_default_mmc
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from matplotlib.pyplot import close

# Set random state and load data from sklearn
random_state = np.random.RandomState(1991)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston[3] = boston[3].astype("category")
boston[8] = boston[8].astype("category")
boston.columns = [str(i) for i in boston.columns]
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)


def test_kernel():
    kernel = mf.MultipleImputedKernel(
        boston_amp, save_all_iterations=True, random_state=random_state
    )
    assert kernel.iteration_count() == 0
    assert kernel.categorical_variables == [3, 8]
    assert kernel.mean_match_candidates == {
        i: _get_default_mmc() for i in range(len(boston_amp.columns))
    }


def test_cust_schema_pandas():
    mmc = {"1": 4, "2": 0.01, "3": 0}
    mms = {"2": 100, "3": 0.5}
    vs = {"1": ["2","3","4","5"], "2": ["6","7"], "3": ["1","2","8"]}

    def mmf(
            mmc,
            model,
            candidate_features,
            bachelor_features,
            candidate_values,
            random_state
    ):
        bachelor_preds = model.predict(bachelor_features)
        imp_values = random_state.choice(candidate_values, size=bachelor_preds.shape[0])

        return imp_values

    kernel = mf.MultipleImputedKernel(
        boston_amp,
        datasets=3,
        variable_schema=vs,
        mean_match_candidates=mmc,
        mean_match_subset=mms,
        mean_match_function=mmf,
        save_all_iterations=True,
        random_state=random_state
    )

    assert kernel.mean_match_candidates == {1: 4, 2: 3, 3: 0}, "mean_match_candidates initialization failed"
    assert kernel.mean_match_subset == {1: 380, 2: 100, 3: 190}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [3, 8], "categorical recognition failed."

    nround = 2
    kernel.mice(nround - 1, variable_parameters={"1": {"n_estimators": 15}}, n_estimators=10)
    assert kernel[0].models[1][nround - 1].params['num_iterations'] == 15
    assert kernel[0].models[2][nround - 1].params['num_iterations'] == 10
    kernel.mice(1, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10)
    assert kernel.iteration_count() == nround, "iteration counting is incorrect."
    assert kernel[0].models[1][nround].params['num_iterations'] == 15
    assert kernel[0].models[2][nround].params['num_iterations'] == 10

    # Test the ability to tune parameters with custom
    optimization_steps = 2
    op, ol = kernel[0].tune_parameters(
        optimization_steps=optimization_steps,
        variable_parameters={1: {"bagging_fraction": 0.9, "feature_fraction_bynode": (0.85, 0.9)}},
        bagging_fraction=0.8,
        feature_fraction_bynode=(0.70,0.75)
    )
    assert op[1]["bagging_fraction"] == 0.9
    assert op[2]["bagging_fraction"] == 0.8
    assert (op[1]["feature_fraction_bynode"] >= 0.85) and (op[1]["feature_fraction_bynode"] <= 0.9)
    assert (op[2]["feature_fraction_bynode"] >= 0.70) and (op[2]["feature_fraction_bynode"] <= 0.75)
    kernel.mice(1, variable_parameters=op)
    model_2_params = kernel[0].models[2][nround + 1].params
    model_1_params = kernel[0].models[1][nround + 1].params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)


    compdat = kernel.complete_data(0)
    assert all(compdat[["1","2","3"]].isnull().sum() == 0)

    kernel2 = mf.KernelDataSet(
        boston_amp,
        variable_schema=vs,
        mean_match_candidates=mmc,
        mean_match_subset=mms,
        mean_match_function=mmf,
        save_all_iterations=True,
        random_state=random_state
    )
    kernel2.mice(nround + 1)
    kernel.append(kernel2)
    kernel.mice(1)

    new_imp_dat = kernel.impute_new_data(new_data = boston_amp.loc[range(250)])
    new_imp_complete = new_imp_dat.complete_data(0)
    assert all(new_imp_complete[["1","2","3"]].isnull().sum() == 0)

    # Plotting on multiple imputed dataset
    new_imp_dat.plot_mean_convergence()
    close()
    new_imp_dat.plot_imputed_distributions()
    close()
    new_imp_dat.plot_correlations()
    close()

    # Plotting on Multiple Imputed Kernel
    kernel.plot_correlations()
    close()
    kernel.plot_feature_importance()
    close()
    kernel.plot_mean_convergence()
    close()
    kernel.plot_imputed_distributions()
    close()


def test_cust_schema_numpy():
    mmc = {1: 4, 2: 0.01, 3: 0}
    mms = {2: 100, 3: 0.5}
    vs = {1: [2,3,4,5], 2: [6,7], 3: [1,2,8]}

    def mmf(
            mmc,
            model,
            candidate_features,
            bachelor_features,
            candidate_values,
            random_state
    ):
        bachelor_preds = model.predict(bachelor_features)
        imp_values = random_state.choice(candidate_values, size=bachelor_preds.shape[0])

        return imp_values

    kernel = mf.MultipleImputedKernel(
        boston_amp.values,
        datasets=3,
        variable_schema=vs,
        mean_match_candidates=mmc,
        mean_match_subset=mms,
        mean_match_function=mmf,
        save_all_iterations=True,
        random_state=random_state
    )

    assert kernel.mean_match_candidates == {1: 4, 2: 3, 3: 0}, "mean_match_candidates initialization failed"
    assert kernel.mean_match_subset == {1: 380, 2: 100, 3: 190}, "mean_match_subset initialization failed"
    assert kernel.iteration_count() == 0, "iteration initialization failed"
    assert kernel.categorical_variables == [], "categorical recognition failed."

    nround = 2
    kernel.mice(nround, variable_parameters={1: {"n_estimators": 15}}, n_estimators=10)
    assert kernel[0].models[1][nround].params['num_iterations'] == 15
    assert kernel[0].models[2][nround].params['num_iterations'] == 10

    # Test the ability to tune parameters with custom
    optimization_steps = 2
    op, ol = kernel[0].tune_parameters(
        optimization_steps=optimization_steps,
        variable_parameters={1: {"bagging_fraction": 0.9, "feature_fraction_bynode": (0.85, 0.9)}},
        bagging_fraction=0.8,
        feature_fraction_bynode=(0.70,0.75)
    )
    assert op[1]["bagging_fraction"] == 0.9
    assert op[2]["bagging_fraction"] == 0.8
    assert (op[1]["feature_fraction_bynode"] >= 0.85) and (op[1]["feature_fraction_bynode"] <= 0.9)
    assert (op[2]["feature_fraction_bynode"] >= 0.70) and (op[2]["feature_fraction_bynode"] <= 0.75)
    kernel.mice(1, variable_parameters=op)
    model_2_params = kernel[0].models[2][nround + 1].params
    model_1_params = kernel[0].models[1][nround + 1].params
    assert model_2_params["bagging_fraction"] == 0.8
    assert model_1_params["bagging_fraction"] == 0.9
    assert (model_2_params["feature_fraction_bynode"] >= 0.70) and (model_2_params["feature_fraction_bynode"] <= 0.75)
    assert (model_1_params["feature_fraction_bynode"] >= 0.85) and (model_1_params["feature_fraction_bynode"] <= 0.9)


    compdat = kernel.complete_data(0)
    assert isinstance(compdat, np.ndarray)
    assert np.isnan(compdat[:,[1,2,3]]).sum() == 0

    kernel2 = mf.KernelDataSet(
        boston_amp.values,
        variable_schema=vs,
        mean_match_candidates=mmc,
        mean_match_subset=mms,
        mean_match_function=mmf,
        save_all_iterations=True,
        random_state=random_state
    )
    kernel2.mice(nround + 1)
    kernel.append(kernel2)
    kernel.mice(1)

    new_imp_dat = kernel.impute_new_data(new_data = boston_amp.loc[range(250)].values)
    new_imp_complete = new_imp_dat.complete_data(0)
    assert np.isnan(new_imp_complete[:,[1,2,3]]).sum() == 0

    # Plotting on multiple imputed dataset
    new_imp_dat.plot_mean_convergence()
    close()
    new_imp_dat.plot_imputed_distributions()
    close()
    new_imp_dat.plot_correlations()
    close()

    # Plotting on Multiple Imputed Kernel
    kernel.plot_correlations()
    close()
    kernel.plot_feature_importance()
    close()
    kernel.plot_mean_convergence()
    close()
    kernel.plot_imputed_distributions()
    close()