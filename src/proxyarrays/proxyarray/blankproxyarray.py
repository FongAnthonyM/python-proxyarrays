"""blankproxyarray.py
A proxy for holding blank data such as NaNs, zeros, or a single number.
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
from collections.abc import Iterable, Sized
from typing import Any, Callable

# Third-Party Packages #
from baseobjects.functions import FunctionRegister, CallableMultiplexer
from dspobjects.operations import nan_array
import numpy as np

# Local Packages #
from .baseproxyarray import BaseProxyArray


# Definitions #
# Classes #
class BlankProxyArray(BaseProxyArray):
    """A proxy for holding blank data such as NaNs, zeros, or a single number.

    This proxy does not store a blank array, rather it generates an array whenever data would be accessed.

    Attributes:
        _shape: The assigned shape that this proxy will be.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        dtype: The data type of that the data will be.
        generate_data: The method for generating data.

    Args:
        shape: The assigned shape that this proxy will be.
        dtype: The data type of the generated data.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        fill_method: The name or the function used to create the blank data.
        fill_kwargs: The keyword arguments for the fill method.
        *args: Arguments for inheritance.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    generation_functions: FunctionRegister = FunctionRegister({
        "nans": nan_array,
        "empty": np.empty,
        "zeros": np.zeros,
        "ones": np.ones,
        "full": np.full,
    })

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        axis: int = 0,
        fill_method: str | Callable = "nans",
        fill_kwargs: dict[str, Any] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # Shape
        self._shape: tuple[int] | None = None
        self.axis: int = 0

        # Data Type
        self.dtype: np.dtype | str = "f4"
        self.fill_kwargs: dict[str, Any] = {}

        # Assign Methods #
        self.generate_data: CallableMultiplexer = CallableMultiplexer(
            register=self.generation_functions,
            instance=self,
            select="nans",
        )

        # Parent Attributes #
        super().__init__(*args, int=init, **kwargs)

        # Construct Object #
        if init:
            self.construct(
                shape=shape,
                axis=axis,
                dtype=dtype,
                fill_method=fill_method,
                fill_kwargs=fill_kwargs,
                **kwargs,
            )

    @property
    def shape(self) -> tuple[int]:
        """The assigned shape that this proxy will be."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int]) -> None:
        self._shape = value

    # Numpy ndarray Methods
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        return self.create_data_range(dtype=dtype)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        shape: tuple[int] | None = None,
        axis: int | None = None,
        dtype: np.dtype | str | None = None,
        fill_method: str | Callable | None = None,
        fill_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            shape: The assigned shape that this proxy will be.
            dtype: The data type of the generated data.
            axis: The axis of the data which this proxy extends for the contained data proxies.
            fill_method: The name or the function used to create the blank data.
            fill_kwargs: The keyword arguments for the fill method.
            **kwargs: Keyword arguments for inheritance.
        """
        if shape is not None:
            self._shape = shape

        if dtype is not None:
            self.dtype = dtype

        if axis is not None:
            self.axis = axis

        if fill_value is not None:
            if isinstance(fill_value, str):
                self.generate_data.select(fill_value)
            else:
                self.generate_data.add_select_function(name=fill_method.__name__, func=fill_method)

        if fill_kwargs is not None:
            self.fill_kwargs.clear()
            self.fill_kwargs.update(fill_kwargs)

        super().construct(**kwargs)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "BlankProxyArray":
        """Create a new copy of this object without proxies.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)
        new_copy._shape = self._shape
        new_copy.axis = self.axis
        new_copy.generate_data.select(self.generate_data.selected)
        return new_copy

    # Getters
    def get_shape(self) -> tuple[int]:
        """Get the shape of this proxy from the contained proxies/objects.

        Returns:
            The shape of this proxy.
        """
        return self.shape

    def get_length(self) -> int:
        """Gets the length of this proxy.

        Returns:
            The length of this proxy.
        """
        return self.shape[self.axis]

    def get_item(self, item: Any) -> Any:
        """Gets an item from within this proxy based on an input item.

        Args:
            item: The object to be used to get a specific item within this proxy.

        Returns:
            An item within this proxy.
        """
        if isinstance(item, slice):
            return self.create_data_slice(item)
        elif isinstance(item, (tuple, list)):
            return self.create_slices_data(item)
        elif isinstance(item, ...):
            return self.create_data_range()

    # Shape
    def validate_shape(self) -> bool:
        """Checks if this proxy has a valid/continuous shape.

        Returns:
            If this proxy has a valid/continuous shape.
        """
        return True

    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the proxy without changing its data.

        Args:
            shape: The shape to change this proxy to.
            **kwargs: Any other kwargs for reshaping.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.shape = shape

    # Create Date
    def create_data_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        dtype: np.dtype | str | None = None,
        proxy: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Creates the data from range style input.

        Args:
            start: The start index to get the data from.
            stop: The stop index to get the data from.
            step: The interval between data to get.
            dtype: The data type to generate.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.
            **kwargs: Keyword arguments for generating data.

        Returns:
            The data requested.
        """
        shape = self.shape
        size = shape[self.axis]

        if dtype is None:
            dtype = self.dtype

        if step is None:
            step = 1

        if start is None:
            start = 0
        elif start < 0:
            start = size + start

        if stop is None:
            stop = size
        elif stop < 0:
            stop = size + stop

        if start < 0 or start >= size or stop < 0 or stop > size:
            raise IndexError("index is out of range")

        size = stop - start
        if size < 0:
            raise IndexError("start index is greater than stop")
        shape[self.axis] = size // step

        if (proxy is None and self.returns_proxy) or proxy:
            new_blank = self.copy()
            new_blank._shape = shape
            return new_blank
        else:
            return self.generate_data(shape=shape, dtype=dtype, **(self.fill_kwargs | kwargs))

    def create_data_slice(self, slice_: slice, dtype: np.dtype | str | None = None, **kwargs: Any) -> np.ndarray:
        """Creates data from a slice.

        Args:
            slice_: The data range to create.
            dtype: The data type to generate.
            **kwargs: Keyword arguments for generating data.

        Returns:
            The requested data.
        """
        return self.create_data_range(slice_.start, slice_.stop, slice_.step, dtype, **kwargs)

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
        if dtype is None:
            dtype = self.dtype

        if slices is None:
            shape = self.shape
        else:
            shape = []
            for index, slice_ in enumerate(slices):
                if isinstance(slice_, int):
                    shape.append(1)
                else:
                    start = 0 if slice_.start is None else slice_.start
                    stop = self.shape[index] if slice_.stop is None else slice_.stop
                    step = 1 if slice_.step is None else slice_.step

                    if start < 0:
                        start = self.shape[index] + start

                    if stop < 0:
                        stop = self.shape[index] + stop

                    if start < 0 or start > self.shape[index] or stop < 0 or stop > self.shape[index]:
                        raise IndexError("index is out of range")

                    size = stop - start
                    if size < 0:
                        raise IndexError("start index is greater than stop")
                    shape.append(size // step)

        return self.generate_data(shape, dtype=dtype, **(self.fill_kwargs | kwargs))

    def get_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int | None = None,
        proxy: bool | None = None,
    ) -> BaseProxyArray | np.ndarray:
        """Gets a range of data along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the data of the range.
            axis: The axis to get the data along.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        return self.create_data_range(start=start, stop=stop, step=step, proxy=proxy)

    def flat_iterator(self) -> Iterable[BaseProxyArray, ...]:
        """Creates an iterator which iterates over the innermost proxies.

        Returns:
            The innermost proxies.
        """
        return (self,)

    # Get Index
    def get_from_index(self, indices: Sized | int, reverse: bool = False, proxy: bool | None = None) -> Any:
        """Get an item recursively from within this proxy using indices.

        Args:
            indices: The indices used to get an item within this proxy.
            reverse: Determines if the indices should be used in the reverse order.
            proxy: Determines if the

        Returns:
            The item recursively from within this proxy.
        """
        if isinstance(indices, int):
            start = indices
        elif len(indices) == 1:
            start = indices[0]
        else:
            raise IndexError("index out of range")

        if (proxy is None and self.returns_proxy) or proxy:
            new_blank = self.copy()
            new_blank._shape[self.axis] = 1
            return new_blank
        else:
            return self.create_data_range(start=start, stop=start + 1)[0]

    # Get Ranges of Data with Slices
    def get_slices_array(self, slices: Iterable[slice | int | None] | None = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested range as an array.
        """
        return self.create_slices_data(slices=slices)

    def fill_slices_array(
        self,
        data_array: np.ndarray,
        array_slices: Iterable[slice] | None = None,
        slices: Iterable[slice | int | None] | None = None,
    ) -> np.ndarray:
        """Fills a given array with blank data.

        Args:
            data_array: The numpy array to fill.
            array_slices: The slices to fill within the data_array.
            slices: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[tuple(array_slices)] = self.create_slices_data(slices=slices)
        return data_array
