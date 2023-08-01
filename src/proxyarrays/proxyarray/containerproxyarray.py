"""containerproxyarray.py
A proxy array which is a container that wraps an array like object to give it proxy functionality.
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
from typing import Any, Union

# Third-Party Packages #
import numpy as np

# Local Packages #
from .baseproxyarray import BaseProxyArray


# Definitions #
# Classes #
class ContainerProxyArray(BaseProxyArray):  # Todo: Make this a StaticWrapper (StaticWrapper needs to be expanded)
    """A proxy container that wraps an array like object to give it proxy functionality.

    Attributes:
        target_shape: The shape that proxy should be and if resized the shape it will default to.
        is_truncate: Determines if the other proxy's data will be truncated to fit this proxy's shape.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        data: The array to wrap.

    Args:
        data: The numpy array for this proxy to wrap.
        shape: The shape that proxy should be and if resized the shape it will default to.
        mode: Determines if the contents of this proxy are editable or not.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for creating a new numpy array.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        mode: str = "a",
        init: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, int=init, **kwargs)

        # Descriptors #
        self.target_shape: tuple[int] | None = None
        self.is_truncate: bool = False
        self.axis: int = 0

        self.data: np.ndarray | None = None

        # Object Construction #
        if init:
            self.construct(data=data, shape=shape, mode=mode, **kwargs)

    @property
    def shape(self):
        """The shape of this proxy, which is the wrapped data."""
        return self.get_shape()

    # Numpy ndarray Methods
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        if dtype is None:
            return self.data
        else:
            return self.data.as_type(dtype)

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The numpy array for this proxy to wrap.
            shape: The shape that proxy should be and if resized the shape it will default to.
            mode: Determines if the contents of this proxy are editable or not.
            **kwargs: Keyword arguments for creating a new numpy array.
        """
        if shape:
            self.target_shape = shape

        if self.target_shape is not None and self.data is None:
            self.data = np.zeros(shape=self.target_shape, **kwargs)

        if data is not None:
            self.data = data

        if mode is not None:
            self.mode = mode

    # Editable Copy Methods
    def _default_spawn_editable(self, *args: Any, **kwargs: Any) -> BaseProxyArray:
        """The default method for creating an editable version of this proxy.

        Args:
            *args: Arguments to help create the new editable proxy.
            **kwargs: Keyword arguments to help create the new editable proxy.

        Returns:
            An editable version of this proxy.
        """
        copy_ = self.copy()
        copy_.mode = "a"
        if self.data is not None:
            copy_.data = self.data.copy()
        return copy_

    # Getters
    def get_length(self) -> int:
        """Gets the length of this proxy.

        Returns:
            The length of this proxy.
        """
        return self.data.shape[self.axis]

    def get_shape(self) -> tuple[int]:
        """Get the shape of this proxy from the contained proxies/objects.

        Returns:
            The shape of this proxy.
        """
        return self.data.shape

    def get_item(self, item: Any) -> Any:
        """Gets an item from within this proxy based on an input item.

        Args:
            item: The object to be used to get a specific item within this proxy.

        Returns:
            An item within this proxy.
        """
        return self.data[item]

    # Shape
    def validate_shape(self) -> bool:
        """Checks if this proxy has a valid/continuous shape.

        Returns:
            If this proxy has a valid/continuous shape.
        """
        return True

    def resize(
        self,
        shape: Iterable[int] | None = None,
        dtype: np.dtype | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Changes the shape of the proxy without changing its data.

        Args:
            shape: The shape to change this proxy to.
            dtype: The data type of the data to resize to.
            **kwargs: Any other kwargs for reshaping.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if shape is None:
            shape = self.target_shape

        if dtype is None:
            dtype = self.data.dtype

        new_slices = [0] * len(shape)
        old_slices = [0] * len(self.shape)
        for index, (n, o) in enumerate(zip(shape, self.shape)):
            slice_ = slice(None, n if n > o else o)
            new_slices[index] = slice_
            old_slices[index] = slice_

        new_ndarray = np.empty(shape, dtype, **kwargs)
        new_ndarray.fill(np.nan)
        new_ndarray[tuple(new_slices)] = self.data[tuple(old_slices)]

        self.data = new_ndarray

    # Data
    def append(self, data: np.ndarray, axis: int | None = None) -> None:
        """Appends data onto the contained data.

        Args:
            data: The data to append onto the contained data.
            axis: The axis to append the new data.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if not any(data.shape):
            return

        if axis is None:
            axis = self.axis

        if self.data is None:
            self.data = data.copy()
        else:
            self.data = np.append(self.data, data, axis)

    def append_proxy(
        self,
        proxy: BaseProxyArray,
        axis: int | None = None,
        truncate: bool | None = None,
    ) -> None:
        """Appends data from another proxy to this proxy.

        Args:
            proxy: The proxy to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other proxy's data will be truncated to fit this proxy's shape.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if truncate is None:
            truncate = self.is_truncate

        shape = self.shape
        temp_shape = list(self.shape)
        temp_other_shape = list(proxy.shape)
        temp_shape[self.axis] = None
        temp_other_shape[self.axis] = None
        slices = ...
        if not proxy.validate_shape or temp_shape != temp_other_shape:
            if not truncate:
                raise ValueError("the proxy's shape does not match this object's.")
            else:
                slices = [None] * len(shape)
                for index, size in enumerate(shape):
                    slices[index] = slice(None, size)
                slices[axis] = slice(None, None)
                slices = tuple(slices)

        self.data = np.append(self.data, proxy[slices], axis)

    def add_proxies(
        self,
        proxies: Iterable[BaseProxyArray],
        axis: int | None = None,
        truncate: bool | None = None,
    ) -> None:
        """Appends data from other proxies to this proxy.

        Args:
            proxies: The proxies to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other proxies' data will be truncated to fit this proxy's shape.
        """
        if self.mode == "r":
            raise IOError("not writable")

        proxies = list(proxies)

        if self.data is None:
            self.data = proxies.pop(0)[...]

        for proxy in proxies:
            self.append_proxy(proxy, axis=axis, truncate=truncate)  # Can be rewritten to be faster.

    def get_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int | None = None,
        proxy: bool | None = None,
    ) -> Union["ContainerProxyArray", np.ndarray]:
        """Gets a range of data along the main axis.

        Args:
            start: The first index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the data of the range.
            axis: The axis to get the data along.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        if axis is None:
            axis = self.axis

        slices = [slice(None, None)] * len(self.shape)
        slices[axis] = slice(start=start, stop=stop, step=step)

        if (proxy is None and self.returns_proxy) or proxy:
            return ContainerProxyArray(data=self.data[tuple(slices)])
        else:
            return self.data[tuple(slices)]

    def set_range(
        self,
        data: np.ndarray,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int | None = None,
    ) -> None:
        """Set a range of data in this proxy.

        Args:
            data: The data to set in the range.
            start: The first index of the range to set.
            stop: The stop point of the range to set.
            step: The interval to set the data of the range.
            axis: The axis to set the data along.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if start is None:
            start = 0

        if stop is None:
            stop = start + data.shape[axis]

        slices = [0] * len(self.shape)
        for index, ax in enumerate(data.shape):
            slices[index] = slice(None, None)
        slices[axis] = slice(start=start, stop=stop, step=step)

        self.data[tuple(slices)] = data

    # Get Index
    def get_from_index(
        self,
        indices: Iterable[int | slice | Iterable] | int | slice,
        reverse: bool = False,
        proxy: bool | None = True,
    ) -> Any:
        """Gets data from this object if given an index which can be in serval formats.

        Args:
            indices: The indices to find the data from.
            reverse:  Determines, when using a Iterable of indices, if it will be read in reverse order.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested proxy or data.
        """
        if isinstance(indices, int):
            start = indices
        elif len(indices) == 1:
            start = indices[0]
        else:
            raise IndexError("index out of range")

        return self.get_range(start=start, stop=start + 1, proxy=proxy)

    def get_slices_array(self, slices: Iterable[slice | int | None] | None = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested range as an array.
        """
        return self.data[slices]

    def fill_slices_array(
        self,
        data_array: np.ndarray,
        array_slices: Iterable[slice] | None = None,
        slices: Iterable[slice | int | None] | None = None,
    ) -> np.ndarray:
        """Fills a given array with values from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slices: The slices to fill within the data_array.
            slices: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[tuple(array_slices)] = self.data[tuple(slices)]
        return data_array


# Assign Cyclic Definitions
ContainerProxyArray.default_editable_type = ContainerProxyArray
