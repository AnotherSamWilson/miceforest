
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
rows = boston.shape[0]
boston.columns = [str(i) for i in boston.columns]
boston["3"] = boston["3"].map({0: 'a', 1: 'b'}).astype('category')
boston["8"] = boston["8"].astype("category")
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)
random_seed_array = np.random.choice(
    range(1000),
    size=rows,
    replace=False
).astype("int32")

def test_pandas_reproducibility():

    datasets = 2
    kernel = mf.ImputationKernel(
        data=boston_amp,
        datasets=datasets,
        initialization="random",
        save_models=2,
        random_state=2
    )

    kernel2 = mf.ImputationKernel(
        data=boston_amp,
        datasets=datasets,
        initialization="random",
        save_models=2,
        random_state=2
    )

    assert kernel.complete_data(0).equals(kernel2.complete_data(0)), (
        "random_state initialization failed to be deterministic"
    )

    # Run mice for 2 iterations
    kernel.mice(2)
    kernel2.mice(2)

    assert kernel.complete_data(0).equals(kernel2.complete_data(0)), (
        "random_state after mice() failed to be deterministic"
    )

    kernel_imputed_as_new = kernel.impute_new_data(
        boston_amp,
        random_state=4,
        random_seed_array=random_seed_array
    )

    # Generate and impute new data as a reordering of original
    new_order = np.arange(rows)
    random_state.shuffle(new_order)
    new_data = boston_amp.loc[new_order]
    new_seeds = random_seed_array[new_order]
    new_imputed = kernel.impute_new_data(
        new_data,
        random_state=4,
        random_seed_array=new_seeds
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = kernel_imputed_as_new.complete_data(dataset=0).loc[new_order]
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert (reordered_kernel_completed == new_data_completed).all().all(), (
            "Seeds did not cause deterministic imputations when data was reordered."
        )

    # Generate and impute new data as a subset of original
    new_ind = [0,1,4,7,8,10]
    new_data = boston_amp.loc[new_ind]
    new_seeds = random_seed_array[new_ind]
    new_imputed = kernel.impute_new_data(
        new_data,
        random_state=4,
        random_seed_array=new_seeds
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = kernel_imputed_as_new.complete_data(dataset=0).loc[new_ind]
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert (reordered_kernel_completed == new_data_completed).all().all(), (
            "Seeds did not cause deterministic imputations when data was reordered."
        )

    # Generate and impute new data as a reordering of original
    new_order = np.arange(rows)
    random_state.shuffle(new_order)
    new_data = boston_amp.loc[new_order]
    new_imputed = kernel.impute_new_data(
        new_data,
        random_state=4,
        random_seed_array=random_seed_array
    )

    # Expect deterministic imputations at the record level, since seeds were passed.
    for i in range(datasets):
        reordered_kernel_completed = kernel_imputed_as_new.complete_data(dataset=0).loc[new_order]
        new_data_completed = new_imputed.complete_data(dataset=0)

        assert not (reordered_kernel_completed == new_data_completed).all().all(), (
            "Different seeds caused deterministic imputations for all rows / columns."
        )
