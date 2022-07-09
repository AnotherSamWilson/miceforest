

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.metrics import roc_auc_score


random_state = np.random.RandomState(5)
iris = pd.concat(load_iris(return_X_y=True, as_frame=True), axis=1)
iris["binary"] = random_state.binomial(1,(iris["target"] + 0.2) / 2.5, size=150)
iris["target"] = iris["target"].astype("category")
iris["binary"] = iris["binary"].astype("category")
iris.columns = [c.replace(" ", "") for c in iris.columns]
iris = pd.concat([iris] * 2, axis=0, ignore_index=True)
iris_amp = mf.utils.ampute_data(iris, perc=0.20)
iris_new = iris.iloc[random_state.choice(iris.index, iris.shape[0], replace=False)].reset_index(drop=True)
iris_new_amp = mf.utils.ampute_data(iris_new, perc=0.20)


def mse(x, y):
    return np.mean((x-y) ** 2)

iterations = 2

kernel_sm2 = mf.ImputationKernel(
    iris_amp,
    datasets=1,
    data_subset=0.75,
    mean_match_candidates=3,
    random_state=random_state
)
kernel_sm2.mice(
    iterations,
    boosting='random_forest',
    num_iterations=100,
    num_leaves=31
)

kernel_sm1 = mf.ImputationKernel(
    iris_amp,
    datasets=1,
    data_subset=0.75,
    mean_match_candidates=3,
    save_models=1,
    random_state=random_state
)
kernel_sm1.mice(
    iterations,
    boosting='random_forest',
    num_iterations=100,
    num_leaves=31
)

def test_sm2_mice_cat():

    # Binary
    col = 5
    ind = kernel_sm2.na_where[col]
    orig = iris.values[ind, col]
    imps = kernel_sm2[0, col, iterations]
    preds = kernel_sm2.get_raw_prediction(col)
    roc = roc_auc_score(orig, preds[ind])
    acc = (imps == orig).mean()
    assert roc > 0.6
    assert acc > 0.6

    # Multiclass
    col = 4
    ind = kernel_sm2.na_where[col]
    orig = iris.values[ind, col]
    imps = kernel_sm2[0, col, iterations]
    preds = kernel_sm2.get_raw_prediction(col)
    roc = roc_auc_score(orig, preds[ind,:], multi_class='ovr', average='macro')
    acc = (imps == orig).mean()
    assert roc > 0.7
    assert acc > 0.7

def test_sm2_mice_reg():
    # Square error of the model predictions should be less than
    # if we just predicted the mean every time.
    imputed_errors = {}
    modeled_errors = {}
    random_sample_error = {}
    for col in [0,1,2,3]:
        ind = kernel_sm2.na_where[col]
        nonmissind = np.delete(range(iris.shape[0]), ind)
        orig = iris.iloc[ind, col]
        preds = kernel_sm2.get_raw_prediction(col)
        imps = kernel_sm2[0, col, iterations]
        random_sample_error[col] = mse(orig, np.mean(iris.iloc[nonmissind, col]))
        modeled_errors[col] = mse(orig, preds[ind])
        imputed_errors[col] = mse(orig, imps)
        assert random_sample_error[col] > modeled_errors[col]
        assert random_sample_error[col] > imputed_errors[col]


def test_sm1_mice_cat():

    # Binary
    col = 5
    ind = kernel_sm1.na_where[col]
    orig = iris.values[ind, col]
    imps = kernel_sm1[0, col, iterations]
    preds = kernel_sm1.get_raw_prediction(col)
    roc = roc_auc_score(orig, preds[ind])
    acc = (imps == orig).mean()
    assert roc > 0.6
    assert acc > 0.6

    # Multiclass
    col = 4
    ind = kernel_sm1.na_where[col]
    orig = iris.values[ind, col]
    imps = kernel_sm1[0, col, iterations]
    preds = kernel_sm1.get_raw_prediction(col)
    roc = roc_auc_score(orig, preds[ind,:], multi_class='ovr', average='macro')
    acc = (imps == orig).mean()
    assert roc > 0.7
    assert acc > 0.7


def test_sm1_mice_reg():
    # Square error of the model predictions should be less than
    # if we just predicted the mean every time.
    imputed_errors = {}
    modeled_errors = {}
    random_sample_error = {}
    for col in [0,1,2,3]:
        ind = kernel_sm1.na_where[col]
        nonmissind = np.delete(range(iris.shape[0]), ind)
        orig = iris.iloc[ind, col]
        preds = kernel_sm1.get_raw_prediction(col)
        imps = kernel_sm1[0, col, iterations]
        random_sample_error[col] = mse(orig, np.mean(iris.iloc[nonmissind, col]))
        modeled_errors[col] = mse(orig, preds[ind])
        imputed_errors[col] = mse(orig, imps)
        assert random_sample_error[col] > modeled_errors[col]
        assert random_sample_error[col] > imputed_errors[col]


################################
### IMPUTE NEW DATA TESTING

new_imp_sm2 = kernel_sm2.impute_new_data(iris_new_amp)
new_imp_sm1 = kernel_sm1.impute_new_data(iris_new_amp)


def test_sm2_ind_cat():

    # Binary
    col = 5
    ind = new_imp_sm2.na_where[col]
    orig = iris_new.values[ind, col]
    imps = new_imp_sm2[0, col, iterations]
    acc = (imps == orig).mean()
    assert acc > 0.6

    # Multiclass
    col = 4
    ind = new_imp_sm2.na_where[col]
    orig = iris_new.values[ind, col]
    imps = new_imp_sm2[0, col, iterations]
    acc = (imps == orig).mean()
    assert acc > 0.7

def test_sm2_ind_reg():
    # Square error of the model predictions should be less than
    # if we just predicted the mean every time.
    imputed_errors = {}
    random_sample_error = {}
    for col in [0,1,2,3]:
        ind = new_imp_sm2.na_where[col]
        nonmissind = np.delete(range(iris.shape[0]), ind)
        orig = iris_new.iloc[ind, col]
        imps = new_imp_sm2[0, col, iterations]
        random_sample_error[col] = mse(orig, np.mean(iris.iloc[nonmissind, col]))
        imputed_errors[col] = mse(orig, imps)
        assert random_sample_error[col] > imputed_errors[col]

def test_sm1_ind_cat():

    # Binary
    col = 5
    ind = new_imp_sm1.na_where[col]
    orig = iris_new.values[ind, col]
    imps = new_imp_sm1[0, col, iterations]
    acc = (imps == orig).mean()
    assert acc > 0.6

    # Multiclass
    col = 4
    ind = new_imp_sm1.na_where[col]
    orig = iris_new.values[ind, col]
    imps = new_imp_sm1[0, col, iterations]
    acc = (imps == orig).mean()
    assert acc > 0.7

def test_sm1_ind_reg():
    # Square error of the model predictions should be less than
    # if we just predicted the mean every time.
    imputed_errors = {}
    random_sample_error = {}
    for col in [0,1,2,3]:
        ind = new_imp_sm1.na_where[col]
        nonmissind = np.delete(range(iris.shape[0]), ind)
        orig = iris_new.iloc[ind, col]
        imps = new_imp_sm1[0, col, iterations]
        random_sample_error[col] = mse(orig, np.mean(iris.iloc[nonmissind, col]))
        imputed_errors[col] = mse(orig, imps)
        assert random_sample_error[col] > imputed_errors[col]
