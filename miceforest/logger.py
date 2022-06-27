from datetime import datetime as dt
import numpy as np
from .compat import pd_DataFrame, PANDAS_INSTALLED


class Logger:
    def __init__(
            self,
            name: str,
            datasets: int,
            variable_names: list,
            iterations: int,
            timed_events: list = [],
            verbose: bool = False
    ) -> None:
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
        self.datasets = datasets
        self.variable_names = variable_names
        self.iterations = iterations
        self.timed_events = timed_events
        self.initialization_time = dt.now()

        if self.verbose:
            print(f"Initialized logger with name {name}")

        if timed_events:
            # Dimensions are Dataset, Variable, Iteration, Event
            self.time_seconds = np.zeros(shape=(
                datasets,
                len(variable_names),
                iterations,
                len(timed_events)
            )).astype("float64")

    def __repr__(self):
        summary_string = f"miceforest logger: {self.name}"
        return summary_string

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def set_start_time(self):
        self._start_time = dt.now()

    def record_time(
            self,
            timed_event: str,
            dataset: int,
            variable_name: str,
            iteration: int,

    ):
        """
        Compares the current time with the start time, and records the time difference
        in our time log in the appropriate register.
        """
        seconds = (dt.now() - self._start_time).total_seconds()
        assert timed_event in self.timed_events, "timed_event is not being recorded"
        assert variable_name in self.variable_names, "variable_name is not being recorded"
        var_indx = self.variable_names.index(variable_name)
        event_indx = self.timed_events.index(timed_event)
        self.time_seconds[dataset, var_indx, iteration, event_indx] = seconds

    def get_time_df_summary(self):
        """
        Returns a frame of the total time taken per variable, event.
        Returns a pandas dataframe if pandas is installed. Otherwise, np.array.
        """
        times = self.time_seconds.sum((0, 2))
        if PANDAS_INSTALLED:
            frame = pd_DataFrame(
                times,
                columns=self.timed_events,
                index=self.variable_names
            )
        else:
            frame = times
        return frame
