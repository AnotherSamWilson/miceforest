from .compat import pd_Series, pd_DataFrame, PANDAS_INSTALLED
from datetime import datetime as dt
from typing import Dict, Any


class Logger:
    def __init__(self, name: str, verbose: bool = False) -> None:
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
        self.initialization_time = dt.now()

        if self.verbose:
            print(f"Initialized logger with name {name}")

        self.time_seconds: Dict[Any, float] = {}

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
        dataset: int,
        variable_name: str,
        iteration: int,
        timed_event: str,
    ):
        """
        Compares the current time with the start time, and records the time difference
        in our time log in the appropriate register. Times can stack for a context.
        """
        seconds = (dt.now() - self._start_time).total_seconds()
        time_key = (dataset, variable_name, iteration, timed_event)
        if time_key in self.time_seconds:
            self.time_seconds[time_key] += seconds
        else:
            self.time_seconds[time_key] = seconds

    def get_time_df_summary(self):
        """
        Returns a frame of the total time taken per variable, event.
        Returns a pandas dataframe if pandas is installed. Otherwise, np.array.
        """

        if PANDAS_INSTALLED:
            dat = pd_Series(self.time_seconds.values(), index=self.time_seconds.keys())
            agg = dat.groupby(level=[1, 3]).sum()
            df = pd_DataFrame(agg).reset_index()
            df.columns = ["Variable", "Event", "Seconds"]
            piv = df.pivot_table(values="Seconds", index="Variable", columns="Event")
            return piv
        else:
            raise ValueError("Returning times as a frame requires pandas")
