"""
miceforest, Multiple Imputation by Chained Equations with LightGBM.

Class / method / function documentation can be found in the readthedocs:
https://miceforest.readthedocs.io/en/latest/index.html

Extensive tutorials can be found on the github README:
https://github.com/AnotherSamWilson/miceforest
"""

import importlib.metadata

from .imputation_kernel import ImputationKernel
from .imputed_data import ImputedData
from .utils import ampute_data

__version__ = importlib.metadata.version("miceforest")


__all__ = [
    "ImputedData",
    "ImputationKernel",
    "ampute_data",
]
