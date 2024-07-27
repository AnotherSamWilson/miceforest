from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.metrics import roc_auc_score


def make_dataset(seed):

    random_state = np.random.RandomState(seed)
    iris = pd.concat(load_iris(return_X_y=True, as_frame=True), axis=1)
    iris["bi"] = random_state.binomial(
        1, (iris["target"] == 0).map({True: 0.9, False: 0.10}), size=150
    )
    iris["bi"] = iris["bi"].astype("category")
    iris["sp"] = iris["target"].map({0: "A", 1: "B", 2: "C"}).astype("category")
    del iris["target"]
    iris.rename(
        {
            "sepal length (cm)": "sl",
            "sepal width (cm)": "sw",
            "petal length (cm)": "pl",
            "petal width (cm)": "pw",
        },
        axis=1,
        inplace=True,
    )
    iris_amp = mf.utils.ampute_data(iris, perc=0.20, random_state=random_state)

    return iris, iris_amp


def get_numeric_performance(kernel, variables, iris):
    r_squares = {}
    iterations = kernel.iteration_count()
    for col in variables:
        ind = kernel.na_where[col]
        orig = iris.loc[ind, col]
        imps = kernel[col, iterations, 0]
        r_squares[col] = np.corrcoef(orig, imps)[0, 1] ** 2
    r_squares = pd.Series(r_squares)
    return r_squares


def get_imp_mse(kernel, variables, iris):
    mses = {}
    iterations = kernel.iteration_count()
    for col in variables:
        ind = kernel.na_where[col]
        orig = iris.loc[ind, col]
        imps = kernel[col, iterations, 0]
        mses[col] = ((orig - imps) ** 2).sum()
    mses = pd.Series(mses)
    return mses


def get_mean_pred_mse(kernel: mf.ImputationKernel, variables, iris):
    mses = {}
    for col in variables:
        ind = kernel.na_where[col]
        orig = iris.loc[ind, col]
        target = kernel._get_nonmissing_values(col)
        pred = target.mean()
        mses[col] = ((orig - pred) ** 2).sum()
    mses = pd.Series(mses)
    return mses


def get_categorical_performance(kernel: mf.ImputationKernel, variables, iris):

    rocs = {}
    accs = {}
    rand_accs = {}
    iterations = kernel.iteration_count()
    for col in variables:
        ind = kernel.na_where[col]
        model = kernel.get_model(col, 0, -1)
        cand = kernel._make_label(col, seed=model.params["seed"])
        orig = iris.loc[ind, col]
        imps = kernel[col, iterations, 0]
        bf = kernel.get_bachelor_features(col)
        preds = model.predict(bf)
        rocs[col] = roc_auc_score(orig, preds, multi_class="ovr", average="macro")
        accs[col] = (imps == orig).mean()
        rand_accs[col] = np.sum(
            cand.value_counts(normalize=True) * orig.value_counts(normalize=True)
        )
    rocs = pd.Series(rocs)
    accs = pd.Series(accs)
    rand_accs = pd.Series(rand_accs)
    return rocs, accs, rand_accs


def test_defaults():

    for i in range(10):
        # i = 3
        print(i)
        iris, iris_amp = make_dataset(i)
        kernel_1 = mf.ImputationKernel(
            iris_amp,
            num_datasets=1,
            data_subset=0,
            mean_match_candidates=3,
            initialize_empty=True,
            random_state=i,
        )
        kernel_1.mice(4, verbose=False)
        kernel_1.complete_data(0, inplace=True)

        rocs, accs, rand_accs = get_categorical_performance(
            kernel_1, ["bi", "sp"], iris
        )
        assert np.all(accs > rand_accs)
        assert np.all(rocs > 0.6)

        # sw Just doesn't have the information density to pass this test reliably.
        # It's definitely the hardest variable to model.
        mses = get_imp_mse(kernel_1, ["sl", "pl", "pw"], iris)
        mpses = get_mean_pred_mse(kernel_1, ["sl", "pl", "pw"], iris)
        assert np.all(mpses > mses)


def test_no_mean_match():

    for i in range(10):
        # i = 0
        iris, iris_amp = make_dataset(i)
        kernel_1 = mf.ImputationKernel(
            iris_amp,
            num_datasets=1,
            data_subset=0,
            mean_match_candidates=0,
            initialize_empty=True,
            random_state=i,
        )
        kernel_1.mice(4, verbose=False)
        kernel_1.complete_data(0, inplace=True)

        rocs, accs, rand_accs = get_categorical_performance(
            kernel=kernel_1, variables=["bi", "sp"], iris=iris
        )
        assert np.all(accs > rand_accs)
        assert np.all(rocs > 0.5)

        # sw Just doesn't have the information density to pass this test reliably.
        # It's definitely the hardest variable to model.
        mses = get_imp_mse(kernel_1, ["sl", "pl", "pw"], iris)
        mpses = get_mean_pred_mse(kernel_1, ["sl", "pl", "pw"], iris)
        assert np.all(mpses > mses)


def test_custom_params():

    for i in range(10):
        # i = 0
        iris, iris_amp = make_dataset(i)
        kernel_1 = mf.ImputationKernel(
            iris_amp,
            num_datasets=1,
            data_subset=0,
            mean_match_candidates=1,
            initialize_empty=True,
            random_state=i,
        )
        kernel_1.mice(
            iterations=4,
            verbose=False,
            boosting="random_forest",
            num_iterations=200,
            min_data_in_leaf=2,
        )
        kernel_1.complete_data(0, inplace=True)

        rocs, accs, rand_accs = get_categorical_performance(
            kernel=kernel_1, variables=["bi", "sp"], iris=iris
        )
        assert np.all(accs > rand_accs)
        assert np.all(rocs > 0.5)

        # sw Just doesn't have the information density to pass this test reliably.
        # It's definitely the hardest variable to model.
        mses = get_imp_mse(kernel_1, ["sl", "pl", "pw"], iris)
        mpses = get_mean_pred_mse(kernel_1, ["sl", "pl", "pw"], iris)
        assert np.all(mpses > mses)
