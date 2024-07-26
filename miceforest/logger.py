from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from pandas import Series


class Logger:
    def __init__(
        self,
        name: str,
        timed_levels: List[str],
        verbose: bool = False,
    ):
        """
        miceforest logger.

        Parameters
        ----------
        name: str
            Name of this logger
        datasets: int
            How many datasets are in this logger
        variable_names: list[str]
            The names of the variables being acted on
        iterations: int
            How many iterations are being run
        timed_events: list[str]
            A list of the events that are going to be timed
        verbose: bool
            Should information be printed.
        """
        self.name = name
        self.verbose = verbose
        self.initialization_time = datetime.now()
        self.timed_levels = timed_levels
        self.started_timers: dict = {}

        if self.verbose:
            print(f"Initialized logger with name {name} and {len(timed_levels)} levels")

        self.time_seconds: Dict[Any, float] = {}

    def __repr__(self):
        summary_string = f"miceforest logger: {self.name}"
        return summary_string

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def set_start_time(self, time_key: Tuple):
        assert len(time_key) == len(self.timed_levels)
        assert time_key not in list(
            self.started_timers
        ), f"Timer {time_key} already started"
        self.started_timers[time_key] = datetime.now()

    def record_time(self, time_key: Tuple):
        """
        Compares the current time with the start time, and records the time difference
        in our time log in the appropriate register. Times can stack for a context.
        """
        assert time_key in list(self.started_timers), f"Timer {time_key} never started"
        seconds = (datetime.now() - self.started_timers[time_key]).total_seconds()
        del self.started_timers[time_key]
        if time_key in self.time_seconds:
            self.time_seconds[time_key] += seconds
        else:
            self.time_seconds[time_key] = seconds

    def get_time_spend_summary(self):
        """
        Returns a frame of the total time taken per variable, event.
        Returns a pandas dataframe if pandas is installed. Otherwise, np.array.
        """
        summary = Series(self.time_seconds)
        summary.index.names = self.timed_levels
        return summary
