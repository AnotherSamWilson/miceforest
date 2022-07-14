
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import miceforest as mf
X, y = make_classification(random_state=0)
X = mf.utils.ampute_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def test_pipeline():
    kernel = mf.ImputationKernel(X_train, datasets=1)

    pipe = Pipeline([
        ('impute', kernel),
        ('scaler', StandardScaler()),
    ])

    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    X_train_t = pipe.fit_transform(
        X_train,
        y_train,
        impute__iterations=2
    )
    X_test_t = pipe.transform(X_test)

    assert not np.any(np.isnan(X_train_t))
    assert not np.any(np.isnan(X_test_t))