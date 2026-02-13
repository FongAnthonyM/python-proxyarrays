"""__init__.py
A time series proxy that wraps file object which contains time series.
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
from ..timeproxy.basetimeproxy import FoundTimeRange
from ..timeseries.basetimeseries import FoundTimeDataRange
