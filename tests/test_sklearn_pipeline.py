import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import miceforest as mf
import pandas as pd


def make_dataset(seed):

    iris = pd.concat(load_iris(return_X_y=True, as_frame=True), axis=1)
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
    iris_amp = mf.utils.ampute_data(iris, perc=0.20)

    return iris_amp


def test_pipeline():

    iris_amp_train = make_dataset(1)
    iris_amp_test = make_dataset(2)

    kernel = mf.ImputationKernel(iris_amp_train, num_datasets=1)

    pipe = Pipeline(
        [
            ("impute", kernel),
            ("scaler", StandardScaler()),
        ]
    )

    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    X_train_t = pipe.fit_transform(X=iris_amp_train, y=None, impute__iterations=2)
    X_test_t = pipe.transform(iris_amp_test)

    assert not np.any(np.isnan(X_train_t))
    assert not np.any(np.isnan(X_test_t))
