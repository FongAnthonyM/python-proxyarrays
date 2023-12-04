"""__init__.py
proxies for holding time series.
"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Local Packages #
from .basetimeseries import BaseTimeSeries
from .containertimeseries import ContainerTimeSeries
from .blanktimeseries import BlankTimeSeries
from .timeseriesproxy import TimeSeriesProxy
