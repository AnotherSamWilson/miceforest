"""Compatibility library."""
"""Stolen from lightgbm"""

"""pandas"""
try:
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        pass
