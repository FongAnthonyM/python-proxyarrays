"""timeaxis.py
A proxy for holding time axis information.
"""

# Header #
__package_name__ = "proxyarrays"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.7.0"

# Imports #
# Standard Libraries #

# Third-Party Packages #

# Local Packages #
from ..timeproxy import TimeProxy
from .basetimeaxis import BaseTimeAxis
from .blanktimeaxis import BlankTimeAxis
from .containertimeaxis import ContainerTimeAxis

# Definitions #
# Classes #
class TimeAxisProxy(TimeProxy, BaseTimeAxis):
    """A TimeProxy that has been expanded to be a time axis."""

    default_return_proxy_leaf = ContainerTimeAxis
    default_fill_type = BlankTimeAxis
    time_axis_type = ContainerTimeAxis

# Assign Cyclic Definitions
TimeAxisProxy.default_return_proxy_node = TimeAxisProxy
TimeAxisProxy.default_return_proxy_type = TimeAxisProxy
