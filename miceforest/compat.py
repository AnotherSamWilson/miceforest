"""Compatibility library."""
"""Stolen from lightgbm"""

"""pandas"""
try:
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import read_parquet as pd_read_parquet

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        pass

    def pd_read_parquet(filepath):
        """Dummy function for pandas.read_parquet."""

        pass
