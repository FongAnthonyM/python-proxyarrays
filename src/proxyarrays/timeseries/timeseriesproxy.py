"""timeseriesproxy.py
A TimeProxy that has been expanded to handle time series data.
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
from ..timeproxy import TimeProxy
from ..timeaxis import ContainerTimeAxis
from .timeseriesproxybase import TimeSeriesProxyBase
from .blanktimeseries import BlankTimeSeries
from .containertimeseries import ContainerTimeSeries


# Definitions #
# Classes #
class TimeSeriesProxy(TimeProxy, TimeSeriesProxyBase):
    """A TimeProxy that has been expanded to handle time series data."""

    default_fill_type = BlankTimeSeries
    time_axis_type = ContainerTimeAxis
    time_series_type = ContainerTimeSeries


# Assign Cyclic Definitions
TimeSeriesProxy.default_return_proxy_type = TimeSeriesProxy
