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
from collections.abc import Iterable, Sized, Generator
from typing import Any, Callable, Union

# Third-Party Packages #
from baseobjects.functions import FunctionRegister, CallableMultiplexer
from dspobjects.operations import nan_array
import numpy as np

# Local Packages #
from .baseproxyarray import BaseProxyArray, Slice
from .containerproxyarray import ContainerProxyArray


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

    default_return_proxy_type: type = None
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

    @property
    def ndim(self) -> int:
        """The number of dimensions of this array."""
        return len(self._shape)

    # Numpy ndarray Methods
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        return self.generate_slice(dtype=dtype)

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

        if fill_method is not None:
            if isinstance(fill_method, str):
                self.generate_data.select(fill_method)
            else:
                self.generate_data.add_select_function(name=fill_method.__name__, func=fill_method)

        if fill_kwargs is not None:
            self.fill_kwargs.clear()
            self.fill_kwargs.update(fill_kwargs)

        super().construct(**kwargs)

    def create_proxy(self, type_: type[BaseProxyArray], *args: Any, **kwargs: Any) -> BaseProxyArray:
        """Creates a new proxy array with the same attributes as this proxy.

        Args:
            type_: The type of proxy array to create.
            *args: The arguments for creating the proxy array.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        if issubclass(type_, BlankProxyArray):
            call_multiplexer = self.generate_data
            kwargs = {"shape": self.shape, "fill_kwargs": self.fill_kwargs} | kwargs
            new_proxy = super().create_proxy(*args, type_=type_, **kwargs)
            new_proxy.generate_data.add_select_function(name=call_multiplexer.selected, func=call_multiplexer.__func__)
            return new_proxy
        else:
            return super().create_proxy(*args, type_=type_, **kwargs)

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

    def proxy_leaf_copy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a copy proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The copy of this proxy array.
        """
        return super().proxy_leaf_copy(type_=type_, **kwargs)

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
            return self.generate_slice(item)
        elif isinstance(item, (tuple, list)):
            return self.generate_slices(item)
        elif item is Ellipsis:
            return self.generate_slice()

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

    # Proxies
    def flat_iterator(self) -> Iterable[BaseProxyArray, ...]:
        """Creates a generator which iterates over the innermost proxies.

        Returns:
            The innermost proxies.
        """
        return (self,)

    def as_flattened(self, type_: type[BaseProxyArray] | None = None, **kwargs: Any) -> BaseProxyArray:
        """Creates a proxy array which contains flattened (proxy depth one) contents of this object.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The flat copy of this object.
        """
        return self.create_return_proxy(type_=type_, **kwargs)

    # Generate Data
    def generate_slices(
        self,
        slices: Iterable[slice | int | None] | None = None,
        dtype: np.dtype | str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generates data based on slices.

        Args:
            slices: The slices to generate the data from.
            dtype: The data type of the generated data.
            **kwargs: Keyword arguments for creating data.

        Returns:
            The requested data.
        """
        if dtype is None:
            dtype = self.dtype

        return self.generate_data(
            shape=self.shape_from_slices(slices=slices),
            dtype=dtype,
            **(self.fill_kwargs | kwargs),
        )

    def generate_slice(
        self,
        start: int | Slice | None = None,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Union["BaseProxyArray", np.ndarray]:
        """Generates data based on a slice.

        Args:
            start: Either the first super index of the slice or the slice itself to generate the data from.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            slice_: The slice to generate the data from.
            axis: The axis to generate the data along.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested data.
        """
        return self.generate_slices(
            slices=self.generate_full_slices(start=start, stop=stop, step=step, axis=axis),
            dtype=dtype,
        )

    # Get Index
    def get_from_index(self, indices: Sized | int, reverse: bool = False, proxy: bool | None = None) -> Any:
        """Gets data from this object if given an index which can be in serval formats.

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

        return self.slice(start=start, stop=start + 1, proxy=proxy)

    # Get Ranges of Data with Slices
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
        data_array[tuple(array_slices)] = self.generate_slices(slices=slices)
        return data_array

    def slices_array(self, slices: Iterable[slice | int | None] | None = None, dtype: Any = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.
            dtype: The dtype of array to return.

        Returns:
            The requested range as an array.
        """
        return self.generate_slices(slices=slices, dtype=dtype)

    def slices_proxy(self, slices: Iterable[Slice] | None = None) -> BaseProxyArray:
        """Get data as a new proxy using slices to determine the data slice.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested slice as a proxy.
        """
        return self.create_return_proxy(shape=self.shape_from_slices(slices))

    def islices(
        self,
        slices: Iterable[Slice | int | None] | None = None,
        islice: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Generator[Union["ContainerProxyArray", np.ndarray], None, None]:
        """Creates a generator which iterates over slices along an axis.

        Args:
            slices: The ranges of the data to get.
            islice: The ranges to data to iterate over.
            axis: The axis to iterate along.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        if axis is None:
            axis = self.axis

        length = len(self)
        slices = list(slices)
        full_slices = slices + [slice(None)] * (self.ndim - len(slices))
        axis_slice = slices[axis]
        if isinstance(axis_slice, int):
            slice_size = axis_slice
            axis_step = None
        else:
            axis_start = 0 if axis_slice.start is None else axis_slice.start
            axis_stop = self.length if axis_slice.stop is None else axis_slice.stop
            axis_step = axis_slice.step
            slice_size = axis_stop - axis_start

        if islice is None:
            outer_start = 0
            outer_stop = length
            outer_step = slice_size
        else:
            outer_start = 0 if islice.start is None else islice.start
            outer_stop = length if islice.stop is None else islice.stop
            outer_step = slice_size * (1 if islice.step is None else islice.step)

        # Adjust outer stop if there is not enough data to fill last slice
        last_index = (((length - outer_start)//outer_step) * outer_step + outer_start)
        adjusted_stop = last_index if (length - last_index) < slice_size else outer_stop

        for inner_start in range(outer_start, adjusted_stop, outer_step):
            full_slices[axis] = slice(inner_start, inner_start + slice_size, axis_step)
            yield self.generate_slices(slices=full_slices, dtype=dtype)


# Assign Cyclic Definitions
BlankProxyArray.default_return_proxy_type = BlankProxyArray
