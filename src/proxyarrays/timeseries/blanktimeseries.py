"""blanktimeseries.py
A proxy for holding blank time series data such as NaNs, zeros, or a single number.
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
from collections.abc import Generator
from datetime import datetime, timedelta
from decimal import Decimal

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..timeproxy import BlankTimeProxy
from .basetimeseries import BaseTimeSeries
from .containertimeseries import ContainerTimeSeries


# Definitions #
# Classes #
class BlankTimeSeries(BlankTimeProxy, BaseTimeSeries):
    """A proxy for holding blank time series data such as NaNs, zeros, or a single number.

    This proxy does not store a blank array, rather it generates an array whenever data would be accessed.
    """

    default_return_proxy_leaf = ContainerTimeSeries
    time_series_type = ContainerTimeSeries

    # Data iterate slice
    def find_data_islice(
        self,
        start: int | None,
        stop: int | None = None,
        step: int | float | timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[BaseTimeSeries, None, None]:
        """Creates a generator which yields data slices.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields data slices.
        """
        inner_slices = self.index_islice_deltatime(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
        )
        return (self.slice(s.start, s.stop, proxy=True) for s in inner_slices)

    def find_data_islice_time(
        self,
        start: datetime | float | int | np.dtype | None = None,
        stop: datetime | float | int | np.dtype | None = None,
        step: int | float | timedelta | None = None,
        istep: int = 1,
    ) -> Generator[BaseTimeSeries, None, None]:
        """Creates a generator which yields nanostamps slices based on times.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields nanostamp slices.
        """
        inner_slices = self.index_islice_time(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
        )

        return (self.slice(s.start, s.stop, proxy=True) for s in inner_slices)  # need to fix


# Assign Cyclic Definitions
BlankTimeProxy.default_return_proxy_type = BlankTimeProxy
