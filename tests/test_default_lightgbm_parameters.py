import numpy as np
import pytest

from miceforest.default_lightgbm_parameters import _LOG_SPACE_SEARCH, _sample_parameters


def test_sample_parameters_handles_scalar_list_and_tuple_inputs():
    rng = np.random.RandomState(10)
    params = {
        "learning_rate": (0.05, 0.2),
        "feature_fraction": [0.5, 0.75, 1.0],
        "min_data_in_leaf": (1, 8),
        "regular_param": 42,
    }

    sampled = _sample_parameters(params, rng, "random")

    assert 0.05 <= sampled["learning_rate"] <= 0.2
    assert sampled["feature_fraction"] in params["feature_fraction"]
    assert 1 <= sampled["min_data_in_leaf"] <= 8
    assert isinstance(sampled["min_data_in_leaf"], int)
    assert sampled["regular_param"] == 42
    assert "seed" in sampled
    assert sampled["seed"] >= 0


def test_sample_parameters_respects_log_space_search():
    rng = np.random.RandomState(0)
    params = {name: (0.1, 1.0) for name in _LOG_SPACE_SEARCH[:2]}
    sampled = _sample_parameters(params, rng, "random")

    for name in params:
        assert params[name][0] <= sampled[name] <= params[name][1]


def test_sample_parameters_requires_random_strategy():
    rng = np.random.RandomState(1)
    with pytest.raises(AssertionError):
        _sample_parameters({}, rng, "grid")


def test_sample_parameters_validates_tuple_bounds():
    rng = np.random.RandomState(2)
    with pytest.raises(AssertionError):
        _sample_parameters({"bad": (1, 1)}, rng, "random")

    with pytest.raises(AssertionError):
        _sample_parameters({"bad": (1, 2, 3)}, rng, "random")
