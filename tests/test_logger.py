import time

import pytest

from miceforest.logger import Logger


def test_logger_records_time_and_summary_structure():
    logger = Logger(name="unit-test", timed_levels=["dataset", "iteration"], verbose=False)
    key = (0, 0)
    logger.set_start_time(key)
    time.sleep(0.01)
    logger.record_time(key)

    assert key in logger.time_seconds
    assert key not in logger.started_timers
    assert logger.time_seconds[key] >= 0

    summary = logger.get_time_spend_summary()
    assert summary.index.names == ["dataset", "iteration"]
    assert pytest.approx(summary.iloc[0]) == logger.time_seconds[key]


def test_logger_prevents_duplicate_starts_and_missing_records():
    logger = Logger(name="safety", timed_levels=["dataset", "iteration"])
    key = (1, 2)

    logger.set_start_time(key)
    with pytest.raises(AssertionError):
        logger.set_start_time(key)

    with pytest.raises(AssertionError):
        logger.record_time((3, 4))


def test_logger_repr_contains_name():
    logger = Logger(name="pretty", timed_levels=["only"])
    assert "pretty" in repr(logger)
