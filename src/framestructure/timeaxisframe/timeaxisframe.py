"""timeaxisframe.py
A frame for holding time axis information.
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
from ..timeframe import TimeFrame
from .timeaxisframeinterface import TimeAxisFrameInterface
from .blanktimeaxisframe import BlankTimeAxisFrame
from .timeaxiscontainer import TimeAxisContainer


# Definitions #
# Classes #
class TimeAxisFrame(TimeFrame, TimeAxisFrameInterface):
    """A TimeFrame that has been expanded to be a time axis."""

    default_fill_type = BlankTimeAxisFrame
    time_axis_type = TimeAxisContainer


# Assign Cyclic Definitions
TimeAxisFrame.default_return_frame_type = TimeAxisFrame
