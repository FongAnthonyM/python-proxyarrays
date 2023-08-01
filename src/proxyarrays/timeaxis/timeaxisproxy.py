"""timeaxis.py
A proxy for holding time axis information.
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
from .basetimeaxis import BaseTimeAxis
from .blanktimeaxis import BlankTimeAxis
from .containertimeaxis import ContainerTimeAxis


# Definitions #
# Classes #
class TimeAxisProxyAxis(TimeProxy, BaseTimeAxis):
    """A TimeProxy that has been expanded to be a time axis."""

    default_fill_type = BlankTimeAxis
    time_axis_type = ContainerTimeAxis


# Assign Cyclic Definitions
TimeAxisProxyAxis.default_return_proxy_type = TimeAxisProxyAxis
