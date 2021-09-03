import numpy as np
from typing import List
from pandas import CategoricalDtype

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    raise ImportError(
        "sklearn.neighbors.NearestNeighbors is required for "
        + "the default mean matching function."
    )


def default_mean_match(
    mmc: int,
    candidate_preds: np.ndarray,
    bachelor_preds: np.ndarray,
    candidate_values: np.ndarray,
    cat_dtype: CategoricalDtype,
    random_state: np.random.RandomState,
):
    """
    The default mean matching function that comes with miceforest. This can be replaced.
    This function is called upon by other classes to perform mean matching. This function
    is called with all parameters every time. If replacing this function with your own,
    you must include all of the parameters above. Descriptions of what exactly will be passed
    to the function are detailed in Datatype Handling below.

    Datatype Handling:
    ------------------
    The datatype of the imputed variable determines how parameters are passed.
    In every case, the parameters passed will equal the following:
        - candidate_preds: Output from lightgbm.Booster.predict()
        - bachelor_preds: output lightgbm.Booster.predict()
        - candidate_values: The original values of non-missing data. If categorical, in the form of cat.codes
        - cat_dtype: pd.CategoricalDtype of the imputed variable from kernel.category_dtypes, or None.

    What to expect in each possible scenario:
        - Categorical
            - Multiclass
                - candidate_preds: np.ndarray of shape (candidates, # classes)
                - bachelor_preds: np.ndarray of shape (bachelors, # classes)
                - candidate_values: np.ndarray of cat.codes of shape (n,)
                - cat_dtype: pd.CategoricalDtype of the imputed variable from kernel.category_dtypes
            - Binary
                - candidate_preds: np.ndarray of shape (candidates,)
                - bachelor_preds: np.ndarray of shape (bachelors,)
                - candidate_values: np.ndarray of cat.codes of shape (n,)
                - cat_dtype: pd.CategoricalDtype of the imputed variable from kernel.category_dtypes
        - Numeric
            - candidate_preds: np.ndarray of shape (candidates,)
            - bachelor_preds: np.ndarray of shape (bachelors,)
            - candidate_values: np.ndarray of numerics of shape (n,)
            - cat_dtype: None

    Parameters
    ----------
    mmc: The number of mean matching candidates.
    candidate_preds: The predictions associated with the candidates from which imputations will
    be pulled.
    bachelor_preds: The predictions associated with the bachelors which we need to procure
    imputed values for.
    candidate_values: The real (not predicted) values of the candidates from the original dataset.
    cat_dtype: If the variable in question is categorical, the pd.CategoricalDtype is passed.
    random_state: The random state from the process calling this function is passed.

    Returns
    -------
    The imputation values
    """

    if mmc == 0:
        if cat_dtype is None:
            imp_values = bachelor_preds
        else:
            # binary objective only outputs a single array: P(y = 1).
            if len(cat_dtype.categories) <= 2:
                bachelor_preds = np.array(
                    [1 - bachelor_preds, bachelor_preds]
                ).transpose()
            imp_values = cat_dtype.categories[np.argmax(bachelor_preds, axis=1)]

    else:

        if cat_dtype is None:

            # lightgbm predict for regression is shape (n,).
            # NearestNeighbors expects (n,1)
            bachelor_preds = bachelor_preds.reshape(-1, 1)
            candidate_preds = candidate_preds.reshape(-1, 1)

            knn = NearestNeighbors(n_neighbors=mmc, algorithm="ball_tree", n_jobs=-1)
            knn.fit(candidate_preds)
            knn_indices = knn.kneighbors(bachelor_preds, return_distance=False)
            index_choice: List[int] = [random_state.choice(i) for i in knn_indices]
            imp_values = candidate_values[index_choice]

        else:

            # binary objective only outputs a single array: P(y = 1).
            if len(cat_dtype.categories) <= 2:
                bachelor_preds = np.array(
                    [1 - bachelor_preds, bachelor_preds]
                ).transpose()

            imp_values = [
                random_state.choice(cat_dtype.categories.values, p=p, size=1)[0]
                for p in bachelor_preds
            ]

    return imp_values
