from datetime import datetime
import numpy as np


class TimeLog:
    """
    Simple class to keep track of total time spend doing certain events.

    Two arrays are stored, one for variable-specific times, and one for
    global times that apply to the class in general.

    Variable times are stored as a 3D list. 1st dimension is variable,
    second is event, third is iteration.

    Global times are stored of a 2D list.
    """

    def __init__(self, column_names, global_timed_events, variable_timed_events):
        self.variable_times = np.empty(
            shape=(len(column_names), len(variable_timed_events), 0)
        ).tolist()
        self.global_times = np.empty(shape=(len(global_timed_events), 0)).tolist()
        self.column_names = column_names
        self.variable_timed_events = variable_timed_events
        self.global_timed_events = global_timed_events

    def __repr__(self):
        maxlen = max([len(l) for l in list(self.variable_times)])
        lines = ["Total Time in Seconds:"]
        for key, time in self.variable_times.items():
            space_add = maxlen - len(key)
            lines.append(f"""{key}: {' '*space_add}{"%.2f" % time.total_seconds()}""")

        return "\n".join(lines)

    def add_variable_time(self, var, event, s):
        """
        Add the time since s to the total time of event.
        This is done once per variable, event, iteration.
        """
        event_ind = self.variable_timed_events.index(event)
        self.variable_times[var][event_ind].append(datetime.now() - s)

    def add_global_time(self, event, s):
        """
        Add the time since s to the total time of event.
        This is done once per variable, event, iteration.
        """
        event_ind = self.global_timed_events.index(event)
        self.global_times[event_ind].append(datetime.now() - s)
