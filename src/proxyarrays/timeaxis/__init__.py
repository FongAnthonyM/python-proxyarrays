"""__init__.py
proxies for timestamps and time information.
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
from .basetimeaxis import BaseTimeAxis
from .containertimeaxis import ContainerTimeAxis
from .blanktimeaxis import BlankTimeAxis
from .timeaxisproxy import TimeAxisProxy
