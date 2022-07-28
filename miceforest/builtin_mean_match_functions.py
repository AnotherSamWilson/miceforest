import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    raise ImportError(
        "scipy.spatial.KDTree is required for " + "the default mean matching function."
    )


def _to_2d(x):
    if x.ndim == 1:
        x.shape = (-1, 1)


def _mean_match_reg(
    mean_match_candidates,
    bachelor_preds,
    candidate_preds,
    candidate_values,
    random_state,
    hashed_seeds,
):
    """
    Determines the values of candidates which will be used to impute the bachelors
    """

    if mean_match_candidates == 0:
        imp_values = bachelor_preds

    else:
        _to_2d(bachelor_preds)
        _to_2d(candidate_preds)

        num_bachelors = bachelor_preds.shape[0]

        # balanced_tree = False fixes a recursion issue for some reason.
        # https://github.com/scipy/scipy/issues/14799
        kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
        _, knn_indices = kd_tree.query(
            bachelor_preds, k=mean_match_candidates, workers=-1
        )

        # We can skip the random selection process if mean_match_candidates == 1
        if mean_match_candidates == 1:
            index_choice = knn_indices

        else:
            # Use the random_state if seed_array was not passed. Faster
            if hashed_seeds is None:
                ind = random_state.randint(mean_match_candidates, size=(num_bachelors))
            # Use the random_seed_array if it was passed. Deterministic.
            else:
                ind = hashed_seeds % mean_match_candidates

            index_choice = knn_indices[np.arange(num_bachelors), ind]

        imp_values = np.array(candidate_values)[index_choice]

    return imp_values


def _mean_match_binary_accurate(
    mean_match_candidates,
    bachelor_preds,
    candidate_preds,
    candidate_values,
    random_state,
    hashed_seeds,
):
    """
    Determines the values of candidates which will be used to impute the bachelors
    """

    if mean_match_candidates == 0:
        imp_values = np.floor(bachelor_preds + 0.5)

    else:
        _to_2d(bachelor_preds)
        _to_2d(candidate_preds)

        num_bachelors = bachelor_preds.shape[0]

        # balanced_tree = False fixes a recursion issue for some reason.
        # https://github.com/scipy/scipy/issues/14799
        kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
        _, knn_indices = kd_tree.query(
            bachelor_preds, k=mean_match_candidates, workers=-1
        )

        # We can skip the random selection process if mean_match_candidates == 1
        if mean_match_candidates == 1:
            index_choice = knn_indices

        else:
            # Use the random_state if seed_array was not passed. Faster
            if hashed_seeds is None:
                ind = random_state.randint(mean_match_candidates, size=(num_bachelors))
            # Use the random_seed_array if it was passed. Deterministic.
            else:
                ind = hashed_seeds % mean_match_candidates

            index_choice = knn_indices[np.arange(num_bachelors), ind]

        imp_values = np.array(candidate_values)[index_choice]

    return imp_values


def _mean_match_binary_fast(
    mean_match_candidates, bachelor_preds, random_state, hashed_seeds
):
    """
    Chooses 0/1 randomly based on probability obtained from prediction.
    If mean_match_candidates is 0, choose class with highest probability.
    """
    if mean_match_candidates == 0:
        imp_values = np.floor(bachelor_preds + 0.5)

    else:
        num_bachelors = bachelor_preds.shape[0]
        if hashed_seeds is None:
            imp_values = random_state.binomial(n=1, p=bachelor_preds)
        else:
            imp_values = []
            for i in range(num_bachelors):
                np.random.seed(seed=hashed_seeds[i])
                imp_values.append(np.random.binomial(n=1, p=bachelor_preds[i]))

            imp_values = np.array(imp_values)

    return imp_values


def _mean_match_multiclass_fast(
    mean_match_candidates, bachelor_preds, random_state, hashed_seeds
):
    """
    If mean_match_candidates is 0, choose class with highest probability.
    Otherwise, randomly choose class weighted by class probabilities.
    """
    if mean_match_candidates == 0:
        imp_values = np.argmax(bachelor_preds, axis=1)

    else:
        num_bachelors = bachelor_preds.shape[0]

        # Turn bachelor_preds into discrete cdf:
        bachelor_preds = bachelor_preds.cumsum(axis=1)

        # Randomly choose uniform numbers 0-1
        if hashed_seeds is None:
            # This is the fastest way to adjust for numeric
            # imprecision of float16 dtype. Actually ends up
            # barely taking any time at all.
            bp_dtype = bachelor_preds.dtype
            unif = np.minimum(
                random_state.uniform(0, 1, size=num_bachelors).astype(bp_dtype),
                bachelor_preds[:, -1],
            )
        else:
            unif = []
            for i in range(num_bachelors):
                np.random.seed(seed=hashed_seeds[i])
                unif.append(np.random.uniform(0, 1, size=1)[0])
            unif = np.array(unif)

        # Choose classes according to their cdf.
        # Distribution will match probabilities
        imp_values = np.array(
            [
                np.searchsorted(bachelor_preds[i, :], unif[i])
                for i in range(num_bachelors)
            ]
        )

    return imp_values


def _mean_match_multiclass_accurate(
    mean_match_candidates,
    bachelor_preds,
    candidate_preds,
    candidate_values,
    random_state,
    hashed_seeds,
):
    """
    Performs nearest neighbors search on class probabilities.
    """
    if mean_match_candidates == 0:
        imp_values = np.argmax(bachelor_preds, axis=1)

    else:
        _to_2d(bachelor_preds)
        _to_2d(candidate_preds)

        num_bachelors = bachelor_preds.shape[0]
        kd_tree = KDTree(candidate_preds, leafsize=16, balanced_tree=False)
        _, knn_indices = kd_tree.query(
            bachelor_preds, k=mean_match_candidates, workers=-1
        )

        # We can skip the random selection process if mean_match_candidates == 1
        if mean_match_candidates == 1:
            index_choice = knn_indices

        else:
            # Come up with random numbers 0-mean_match_candidates, with priority given to hashed_seeds
            if hashed_seeds is None:
                ind = random_state.randint(mean_match_candidates, size=(num_bachelors))
            else:
                ind = hashed_seeds % mean_match_candidates

            index_choice = knn_indices[np.arange(knn_indices.shape[0]), ind]

        imp_values = np.array(candidate_values)[index_choice]

    return imp_values
