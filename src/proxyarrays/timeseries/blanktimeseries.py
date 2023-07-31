"""blanktimeseries.py
A proxy for holding blank time series data such as NaNs, zeros, or a single number.
"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #

# Third-Party Packages #

# Local Packages #
from ..timeproxy import BlankTimeProxy
from .timeseriesproxybase import TimeSeriesProxyBase
from .containertimeseries import ContainerTimeSeries


# Definitions #
# Classes #
class BlankTimeSeries(BlankTimeProxy, TimeSeriesProxyBase):
    """A proxy for holding blank time series data such as NaNs, zeros, or a single number.

    This proxy does not store a blank array, rather it generates an array whenever data would be accessed.
    """

    time_series_type = ContainerTimeSeries
