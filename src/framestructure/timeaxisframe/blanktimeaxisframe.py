"""blanktimeaxisframe.py
A frame for generating time axis information.
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
from collections.abc import Iterable
from typing import Any

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..timeframe import BlankTimeFrame
from .timeaxisframeinterface import TimeAxisFrameInterface


# Definitions #
# Classes #
class BlankTimeAxisFrame(BlankTimeFrame, TimeAxisFrameInterface):
    """A frame for generating time axis information."""

    # Instance Methods #
    # Create Data
    def create_data_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        dtype: np.dtype | str | None = None,
        frame: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Creates the data from range style input.

        Args:
            start: The start index to get the data from.
            stop: The stop index to get the data from.
            step: The interval between data to get.
            dtype: The data type to generate.
            frame: Determines if returned object is a Frame or an array, default is this object's setting.
            **kwargs: Keyword arguments for generating data.

        Returns:
            The data requested.
        """
        if (frame is None and self.returns_frame) or frame:
            if start is None:
                start = self._assigned_start
            else:
                start = self.get_nanostamp(start)

            if stop is None:
                stop = self._assigned_end
            else:
                m = (stop - start) % step
                stop = self.get_nanostamp(stop - (m if m > 0 else step))

            new_blank = self.copy()
            new_blank._assign_start(start)
            new_blank._assign_end(stop)
            new_blank.refesh()
            return new_blank
        else:
            return self._create_method(start=start, stop=stop, step=step, dtype=dtype)

    def create_slices_data(
        self,
        slices: Iterable[slice | int | None] | None = None,
        dtype: np.dtype | str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Creates data from slices.

        Args:
            slices: The slices to generate the data from.
            dtype: The data type of the generated data.
            **kwargs: Keyword arguments for creating data.

        Returns:
            The requested data.
        """
        if slices is None:
            start = None
            stop = None
            step = 1

            shape = slice(None)
        else:
            shape = list(slices)

            slice_ = shape[self.axis]
            if isinstance(slice_, int):
                start = slice_
                stop = slice_ + 1
                step = 1
                shape[self.axis] = 0
            else:
                start = slice_.start
                stop = slice_.stop
                step = 1 if slice_.step is None else slice_.step
                shape[self.axis] = slice(None)

        return self._create_method(start=start, stop=stop, step=step, dtype=dtype)[tuple(shape)]
