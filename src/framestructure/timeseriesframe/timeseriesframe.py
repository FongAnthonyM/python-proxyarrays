"""timeseriesframe.py
A TimeFrame that has been expanded to handle time series data.
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
from ..timeaxisframe import TimeAxisContainer
from .timeseriesframeinterface import TimeSeriesFrameInterface
from .blanktimeseriesframe import BlankTimeSeriesFrame
from .timeseriescontainer import TimeSeriesContainer


# Definitions #
# Classes #
class TimeSeriesFrame(TimeFrame, TimeSeriesFrameInterface):
    """A TimeFrame that has been expanded to handle time series data."""

    default_fill_type = BlankTimeSeriesFrame
    time_axis_type = TimeAxisContainer
    time_series_type = TimeSeriesContainer


# Assign Cyclic Definitions
TimeSeriesFrame.default_return_frame_type = TimeSeriesFrame
