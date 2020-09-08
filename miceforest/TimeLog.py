from datetime import datetime, timedelta


class TimeLog:
    """
    Simple class to keep track of total time spend doing
    certain events. Can add events when TimeLog is inherited.
    """

    def __init__(self, timed_events):
        self.times = {}
        for e in timed_events:
            self.times[e] = timedelta(0)

    def __repr__(self):
        maxlen = max([len(l) for l in list(self.times)])
        lines = ["Total Time in Seconds:"]
        for key, time in self.times.items():
            space_add = maxlen - len(key)
            lines.append(f"""{key}: {' '*space_add}{"%.2f" % time.total_seconds()}""")

        return "\n".join(lines)

    def add_time(self, event, s):
        """
        Add the time since s to the total time of event.
        """
        self.times[event] += datetime.now() - s

    def add_events(self, timed_events):
        """
        Add events. Used for inheritance
        """
        for e in timed_events:
            self.times[e] = timedelta(0)

    def get_event_time(self, event):
        self.times[event].total_seconds()
