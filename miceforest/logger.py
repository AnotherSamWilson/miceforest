class Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
