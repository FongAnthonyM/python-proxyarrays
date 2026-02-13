"""__init__.py
proxies for directory/file objects which contain time series data.
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
from .basedirectorytimeseries import BaseDirectoryTimeSeries
from .directorytimeseriesproxy import DirectoryTimeSeriesProxy
