from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf


# Make random state and load data
# Define data
random_state = np.random.RandomState(1991)
iris = pd.concat(load_iris(as_frame=True, return_X_y=True), axis=1)
iris["sp"] = iris["target"].astype("category")
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
iris["bc"] = pd.Series(np.random.binomial(n=1, p=0.5, size=150)).astype("category")
iris_amp = mf.ampute_data(iris, perc=0.25, random_state=random_state)
rows = iris_amp.shape[0]
random_seed_array = np.random.choice(range(1000), size=rows, replace=False).astype(
    "int32"
)


def test_pandas_reproducibility():

    datasets = 2
    kernel = mf.ImputationKernel(
        data=iris_amp, num_datasets=datasets, initialize_empty=False, random_state=2
    )

    kernel2 = mf.ImputationKernel(
        data=iris_amp, num_datasets=datasets, initialize_empty=False, random_state=2
    )

    assert kernel.complete_data(0).equals(
        kernel2.complete_data(0)
    ), "random_state initialization failed to be deterministic"
    assert kernel.complete_data(1).equals(
        kernel2.complete_data(1)
    ), "random_state initialization failed to be deterministic"

    # Run mice for 2 iterations
    kernel.mice(2)
    kernel2.mice(2)

    assert kernel.complete_data(0).equals(
        kernel2.complete_data(0)
    ), "random_state after mice() failed to be deterministic"
    assert kernel.complete_data(1).equals(
        kernel2.complete_data(1)
    ), "random_state after mice() failed to be deterministic"

    kernel_imputed_as_new = kernel.impute_new_data(
        iris_amp, random_state=4, random_seed_array=random_seed_array
    )

    # Generate and impute new data as a reordering of original
    new_order = np.arange(rows)
    random_state.shuffle(new_order)
    new_data = iris_amp.loc[new_order].reset_index(drop=True)
    new_seeds = random_seed_array[new_order]
    new_imputed = kernel.impute_new_data(
        new_data, random_state=4, random_seed_array=new_seeds
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = (
            kernel_imputed_as_new.complete_data(dataset=0)
            .loc[new_order]
            .reset_index(drop=True)
        )
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert (
            (reordered_kernel_completed == new_data_completed).all().all()
        ), "Seeds did not cause deterministic imputations when data was reordered."

    # Generate and impute new data as a subset of original
    new_ind = [0, 1, 4, 7, 8, 10]
    new_data = iris_amp.loc[new_ind].reset_index(drop=True)
    new_seeds = random_seed_array[new_ind]
    new_imputed = kernel.impute_new_data(
        new_data, random_state=4, random_seed_array=new_seeds
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = (
            kernel_imputed_as_new.complete_data(dataset=0)
            .loc[new_ind]
            .reset_index(drop=True)
        )
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert (
            (reordered_kernel_completed == new_data_completed).all().all()
        ), "Seeds did not cause deterministic imputations when data was reordered."

    # Generate and impute new data as a reordering of original
    new_order = np.arange(rows)
    random_state.shuffle(new_order)
    new_data = iris_amp.loc[new_order].reset_index(drop=True)
    new_imputed = kernel.impute_new_data(
        new_data, random_state=4, random_seed_array=random_seed_array
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = (
            kernel_imputed_as_new.complete_data(dataset=0)
            .loc[new_order]
            .reset_index(drop=True)
        )
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert (
            not (reordered_kernel_completed == new_data_completed).all().all()
        ), "Different seeds caused deterministic imputations for all rows / columns."
