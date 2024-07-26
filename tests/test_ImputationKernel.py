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
iris["sp"] = (
    iris["target"]
    .map({0: "Category1", 1: "Category2", 2: "Category3"})
    .astype("category")
)
del iris["target"]
iris.rename(
    {
        "sepal length (cm)": "sl",
        "sepal width (cm)": "ws",
        "petal length (cm)": "pl",
        "petal width (cm)": "pw",
    },
    axis=1,
    inplace=True,
)
iris["bi"] = (
    pd.Series(np.random.binomial(n=1, p=0.5, size=iris.shape[0]))
    .map({0: "FOO", 1: "BAR"})
    .astype("category")
)
iris["ui8"] = iris["sl"].round(0).astype("UInt8")
iris["ws"] = iris["ws"].astype("float32")
iris.reset_index(drop=True, inplace=True)
amputed_variables = ["sl", "ws", "pl", "sp", "bi", "ui8"]
iris_amp = mf.ampute_data(
    iris, variables=amputed_variables, perc=0.25, random_state=random_state
)
na_where = {var: np.where(iris_amp[var].isnull())[0] for var in iris_amp.columns}
notnan_where = {
    var: np.setdiff1d(np.arange(iris_amp.shape[0]), na_where[var], assume_unique=True)[
        0
    ]
    for var in iris_amp.columns
}

new_amputed_data = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
new_nonmissing_data = iris.loc[range(20), :].reset_index(drop=True).copy()

# Make special datasets that have weird edge cases
# Multiple columns with all missing values
# sp is categorical, and pw had no missing
# values in the original kernel data
new_amputed_data_special_1 = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
for col in ["sp", "pw"]:
    new_amputed_data_special_1[col] = np.nan
    dtype = iris[col].dtype
    new_amputed_data_special_1[col] = new_amputed_data_special_1[col].astype(dtype)

# Some columns with no missing values
new_amputed_data_special_2 = iris_amp.loc[range(20), :].reset_index(drop=True).copy()
new_amputed_data_special_2[["sp", "ui8"]] = iris.loc[range(20), ["sp", "ui8"]]


def make_and_test_kernel(**kwargs):

    # kwargs = {
    #     "data": iris_amp,
    #     "num_datasets": 2,
    #     "mean_match_strategy": "normal",
    #     "save_all_iterations_data": True,
    # }

    # Build a normal kernel, run mice, save, load, and run mice again
    kernel = mf.ImputationKernel(**kwargs)
    assert kernel.iteration_count() == 0
    kernel.mice(iterations=2, verbose=True)
    assert kernel.iteration_count() == 2
    new_file, filename = mkstemp()
    with open(filename, "wb") as file:
        dill.dump(kernel, file)
    del kernel
    with open(filename, "rb") as file:
        kernel = dill.load(file)
    kernel.mice(iterations=1, verbose=True)
    assert kernel.iteration_count() == 3

    modeled_variables = kernel.model_training_order
    imputed_variables = kernel.imputed_variables

    # pw has no missing values.
    assert "pw" not in imputed_variables

    # Make a completed dataset
    completed_data = kernel.complete_data(dataset=0, inplace=False)

    # Make sure the data was imputed
    assert all(completed_data[imputed_variables].isnull().sum() == 0)

    # Make sure the dtypes didn't change
    for col, series in iris_amp.items():
        dtype = series.dtype
        assert completed_data[col].dtype == dtype

    # Make sure the working data wasn't imputed
    for var, naw in na_where.items():
        if len(naw) > 0:
            assert kernel.working_data.loc[naw, var].isnull().mean() == 1.0

    # Make sure the original nonmissing data wasn't changed
    for var, naw in notnan_where.items():
        assert completed_data.loc[naw, var] == iris_amp.loc[naw, var]

    # Impute the data in place now
    kernel.complete_data(0, inplace=True)

    # Assert we actually imputed the working data
    assert all(kernel.working_data[imputed_variables].isnull().sum() == 0)

    # Assert the original data was not touched
    assert all(iris_amp[imputed_variables].isnull().sum() > 0)

    # Make sure the models were trained the way we expect
    for variable in modeled_variables:
        if variable == "sp":
            objective = "multiclass"
        elif variable == "bi":
            objective = "binary"
        else:
            objective = "regression"
        assert (
            kernel.get_model(variable=variable, dataset=0, iteration=1).params[
                "objective"
            ]
            == objective
        )
        assert (
            kernel.get_model(variable=variable, dataset=0, iteration=2).params[
                "objective"
            ]
            == objective
        )
        assert (
            kernel.get_model(variable=variable, dataset=1, iteration=1).params[
                "objective"
            ]
            == objective
        )
        assert (
            kernel.get_model(variable=variable, dataset=1, iteration=2).params[
                "objective"
            ]
            == objective
        )

    # Impute a new dataset, and complete the data
    imputed_new_data = kernel.impute_new_data(new_amputed_data, verbose=True)
    imputed_dataset_0 = imputed_new_data.complete_data(
        dataset=0, iteration=2, inplace=False
    )
    imputed_dataset_1 = imputed_new_data.complete_data(
        dataset=1, iteration=2, inplace=False
    )

    # Assert we didn't just impute the same thing for all values
    assert not np.all(imputed_dataset_0 == imputed_dataset_1)

    # Make sure we can impute the special cases
    imputed_data_special_1 = kernel.impute_new_data(new_amputed_data_special_1)

    # Before we do anything else, make sure saving / loading works
    new_file, filename = mkstemp()
    with open(filename, "wb") as file:
        dill.dump(imputed_data_special_1, file)
    del imputed_data_special_1
    with open(filename, "rb") as file:
        imputed_data_special_1 = dill.load(file)

    imputed_data_special_2 = kernel.impute_new_data(new_amputed_data_special_2)
    imputed_dataset_special_1 = imputed_data_special_1.complete_data(0)
    imputed_dataset_special_2 = imputed_data_special_2.complete_data(0)
    assert not np.any(imputed_dataset_special_1[modeled_variables].isnull())
    assert not np.any(imputed_dataset_special_2[modeled_variables].isnull())

    # Reproducibility
    random_seed_array = np.random.randint(
        9999, size=new_amputed_data_special_1.shape[0], dtype="uint32"
    )
    imputed_data_special_3 = kernel.impute_new_data(
        new_data=new_amputed_data_special_1,
        random_seed_array=random_seed_array,
        random_state=1,
    )
    imputed_data_special_4 = kernel.impute_new_data(
        new_data=new_amputed_data_special_1,
        random_seed_array=random_seed_array,
        random_state=1,
    )
    assert imputed_data_special_3.complete_data(0).equals(
        imputed_data_special_4.complete_data(0)
    )

    # Ensure kernel imputes new data on a subset of datasets deterministically
    if kernel.num_datasets > 1:
        datasets = list(range(kernel.num_datasets))
        datasets.remove(0)
        imputed_data_special_5 = kernel.impute_new_data(
            new_data=new_amputed_data_special_1,
            datasets=datasets,
            random_seed_array=random_seed_array,
            random_state=1,
            verbose=True,
        )
        imputed_data_special_6 = kernel.impute_new_data(
            new_data=new_amputed_data_special_1,
            datasets=datasets,
            random_seed_array=random_seed_array,
            random_state=1,
        )
        assert imputed_data_special_5.complete_data(1).equals(
            imputed_data_special_6.complete_data(1)
        )

    mv = kernel.modeled_variables

    # Test tuning parameters
    kernel.tune_parameters(
        optimization_steps=2,
        use_gbdt=True,
        random_state=1,
        variable_parameters={
            mv[0]: {
                "min_data_in_leaf": (1, 10),
                "cat_l2": 0.5,
            }
        },
        extra_trees=[True, False],
    )
    op = kernel.optimal_parameters[mv[0]]
    assert "extra_trees" in list(op)
    assert op["cat_l2"] == 0.5
    assert 1 <= op["min_data_in_leaf"] <= 10

    kernel.tune_parameters(
        optimization_steps=2,
        use_gbdt=False,
        random_state=1,
        variable_parameters={
            mv[0]: {
                "min_data_in_leaf": (1, 10),
                "cat_l2": 0.5,
            }
        },
        extra_trees=[True, False],
    )
    op = kernel.optimal_parameters[mv[0]]
    assert "extra_trees" in list(op)
    assert op["cat_l2"] == 0.5
    assert 1 <= op["min_data_in_leaf"] <= 10

    # Test plotting
    kernel.plot_imputed_distributions()
    kernel.plot_feature_importance(dataset=0)

    return kernel


def test_defaults():

    kernel_normal = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy="normal",
        save_all_iterations_data=True,
    )
    kernel_fast = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy="fast",
        save_all_iterations_data=True,
    )
    kernel_shap = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_strategy="shap",
        save_all_iterations_data=True,
    )
    kernel_iwp = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        mean_match_candidates=0,
        save_all_iterations_data=True,
    )


def test_complex():

    # Customize everything.
    vs = {
        "sl": ["ws", "pl", "pw", "sp", "bi"],
        "ws": ["sl"],
        "pl": ["sp", "bi"],
        # 'sp': ['sl', 'ws', 'pl', 'pw', 'bc'], # Purposely don't train a variable that does have missing values
        "pw": ["sl", "ws", "pl", "sp", "bi"],
        "bi": ["ws", "pl", "sp"],
        "ui8": ["sp", "ws"],
    }
    mmc = {"sl": 4, "ws": 0, "bi": 5}
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
        mean_match_strategy="normal",
        save_all_iterations_data=True,
    )
    assert kernel.data_subset == {
        "sl": 75,
        "ws": 50,
        "pl": 0,
        "bi": 0,
        "ui8": 0,
        "pw": 0,
    }, "mean_match_subset initialization failed"

    kernel_fast = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_strategy="fast",
        save_all_iterations_data=True,
    )

    mmc_shap = mmc.copy()
    mmc_shap["ws"] = 1
    kernel_shap = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc_shap,
        data_subset=ds,
        mean_match_strategy="shap",
        save_all_iterations_data=True,
    )

    mixed_mms = {"sl": "shap", "ws": "fast", "ui8": "fast", "bi": "normal"}
    kernel_mixed = make_and_test_kernel(
        data=iris_amp,
        num_datasets=2,
        variable_schema=vs,
        mean_match_candidates=mmc,
        data_subset=ds,
        mean_match_strategy=mixed_mms,
        save_all_iterations_data=True,
    )
