"""blanktimeaxis.py
A proxy for generating time axis information.
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
from ..timeproxy import BlankTimeProxy
from .basetimeaxis import BaseTimeAxis
from .containertimeaxis import ContainerTimeAxis


# Definitions #
# Classes #
class BlankTimeAxis(BlankTimeProxy, BaseTimeAxis):
    """A proxy for generating time axis information."""
    default_return_proxy_node = ContainerTimeAxis

    # Magic Methods #
    @property
    def ndim(self) -> int:
        """The number of dimensions of this array."""
        return 1

    # Instance Methods #
    # Create Data
    def generate_slices(
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

        return self.generate_time(start=start, stop=stop, step=step, dtype=dtype)[tuple(shape)]


# Assign Cyclic Definitions
BlankTimeAxis.default_return_proxy_type = BlankTimeAxis
