"""__init__.py
proxies for holding time series.
"""

# Header #
__package_name__ = "proxyarrays"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.7.0"

# Imports #
# Local Packages #
from .basetimeseries import BaseTimeSeries
from .containertimeseries import ContainerTimeSeries
from .blanktimeseries import BlankTimeSeries
from .timeseriesproxy import TimeSeriesProxy
