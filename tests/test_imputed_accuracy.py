
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import miceforest as mf

# Define data
random_state = np.random.RandomState(5)
boston = pd.DataFrame(load_boston(return_X_y=True)[0])
boston.columns = [str(i) for i in boston.columns]
boston["3"] = boston["3"].astype("category")
boston["8"] = boston["8"].astype("category")
boston_amp = mf.ampute_data(boston, perc=0.25, random_state=random_state)
new_data = boston_amp.loc[range(10),:]


kernel = mf.ImputationKernel(boston_amp, random_state=random_state)
iterations = 2
kernel.mice(
    iterations,
    boosting='random_forest',
    num_iterations=50,
    num_leaves=31
)

def mse(x, y):
    return np.mean((x-y) ** 2)

def test_classification_defaults():
    # Commented out because this binary variable is extremely unbalanced
    # Binary
    # col = 3
    # preds = kernel.get_raw_prediction(col)
    # ind = kernel.na_where[col]
    # orig = boston.values[ind,col]
    # roc = roc_auc_score(orig, preds[ind])
    # assert roc > 0.6

    # Multiclass
    col = 8
    ind = kernel.na_where[col]
    orig = boston.values[ind, col]
    preds = kernel.get_raw_prediction(col)
    roc = roc_auc_score(orig, preds[ind,:], multi_class='ovr', average='macro')
    assert roc > 0.7


def test_regression_defaults_pretune():
    # Square error of the model predictions should be less than
    # if we just predicted the mean every time.
    imputed_errors = {}
    modeled_errors = {}
    random_sample_error = {}
    for col in [0,1,2,4,5,6,7,9,10,11,12]:
        ind = kernel.na_where[col]
        nonmissind = np.delete(range(boston.shape[0]), ind)
        preds = kernel.get_raw_prediction(col)
        imps = kernel[0, col, iterations]
        random_sample_error[col] = mse(boston.iloc[ind, col], np.mean(boston.iloc[nonmissind, col]))
        modeled_errors[col] = mse(boston.iloc[ind, col], preds[ind])
        imputed_errors[col] = mse(boston.iloc[ind, col], imps)
        assert random_sample_error[col] > modeled_errors[col]
        assert random_sample_error[col] > imputed_errors[col]
