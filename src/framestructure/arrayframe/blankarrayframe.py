"""blankarrayframe.py
A frame for holding blank data such as NaNs, zeros, or a single number.
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
from dspobjects.operations import nan_array
import numpy as np

# Local Packages #
from .arrayframeinterface import ArrayFrameInterface


# Definitions #
# Classes #
class BlankArrayFrame(ArrayFrameInterface):
    """A frame for holding blank data such as NaNs, zeros, or a single number.

    This frame does not store a blank array, rather it generates an array whenever data would be accessed.

    Attributes:
        _shape: The assigned shape that this frame will be.
        axis: The main axis of this frame.
        dtype: The data type of that the data will be.
        generate_data: The method for generating data.

    Args:
        shape: The assigned shape that this frame will be.
        dtype: The data type of the generated data.
        init: Determines if this object will construct.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        init: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # Shape
        self._shape: tuple[int] | None = None
        self.axis: int = 0

        # Data Type
        self.dtype: np.dtype | str = "f4"

        # Assign Methods #
        self._generate_method: Callable[[tuple[int], Any], np.ndarray] = self.create_nans.__func__

        # Parent Attributes #
        super().__init__(*args, int=init, **kwargs)

        # Construct Object #
        if init:
            self.construct(shape=shape, dtype=dtype)

    @property
    def shape(self) -> tuple[int]:
        """The assigned shape that this frame will be."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int]) -> None:
        self._shape = value

    @property
    def generate_data(self):
        """The selected method for creating timestamps"""
        return self._generate_method.__get__(self, self.__class__)

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
    def construct(self, shape: tuple[int] | None = None, dtype: np.dtype | str | None = None) -> None:
        """Constructs this object.

        Args:
            shape: The assigned shape that this frame will be.
            dtype: The data type of the generated data.
        """
        if shape is not None:
            self._shape = shape

        if dtype is not None:
            self.dtype = dtype

    # Getters
    def get_shape(self) -> tuple[int]:
        """Get the shape of this frame from the contained frames/objects.

        Returns:
            The shape of this frame.
        """
        return self.shape

    def get_length(self) -> int:
        """Gets the length of this frame.

        Returns:
            The length of this frame.
        """
        return self.shape[self.axis]

    def get_item(self, item: Any) -> Any:
        """Gets an item from within this frame based on an input item.

        Args:
            item: The object to be used to get a specific item within this frame.

        Returns:
            An item within this frame.
        """
        if isinstance(item, slice):
            return self.create_data_slice(item)
        elif isinstance(item, (tuple, list)):
            return self.create_slices_data(item)
        elif isinstance(item, ...):
            return self.create_data_range()

    # Setters
    def set_data_generator(self, generator: str | Callable[[tuple[int], Any], np.ndarray]) -> None:
        """Sets this frame's data generator to either numpy array creator or a function that will create data.

        Args:
            generator: Either the string of the type of data to generate or a function to generate data.
        """
        if isinstance(generator, str):
            generator = generator.lower()
            if generator == "nan":
                self._generate_data = nan_array
            elif generator == "empty":
                self._generate_data = np.empty
            elif generator == "zeros":
                self._generate_data = np.zeros
            elif generator == "ones":
                self._generate_data = np.ones
            elif generator == "full":
                self._generate_data = np.full
        else:
            self._generate_data = generator

    # Shape
    def validate_shape(self) -> bool:
        """Checks if this frame has a valid/continuous shape.

        Returns:
            If this frame has a valid/continuous shape.
        """
        return True

    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the frame without changing its data.

        Args:
            shape: The shape to change this frame to.
            **kwargs: Any other kwargs for reshaping.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.shape = shape

    # Create Data
    def create_nans(
        self,
        shape: int | Iterable | tuple[int],
        dtype: object | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Creates an array of NaNs.

        Args:
            shape: The shape of the array to create.
            dtype: The data type of the array.
            **kwargs: The other numpy keyword arguments for creating an array.

        Returns:
            The array of NaNs.
        """
        return nan_array(shape=shape, dtype=dtype, **kwargs)

    def create_data_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
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

        if (frame is None and self.returns_frame) or frame:
            new_blank = self.copy()
            new_blank._shape = shape
            return new_blank
        else:
            return self.generate_data(shape=shape, dtype=dtype, **kwargs)

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

        return self.generate_data(shape, dtype=dtype, **kwargs)

    def get_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int | None = None,
        frame: bool | None = None,
    ) -> ArrayFrameInterface | np.ndarray:
        """Gets a range of data along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the data of the range.
            axis: The axis to get the data along.
            frame: Determines if returned object is a Frame or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        return self.create_data_range(start=start, stop=stop, step=step, frame=frame)

    # Get Index
    def get_from_index(self, indices: Sized | int, reverse: bool = False, frame: bool | None = None) -> Any:
        """Get an item recursively from within this frame using indices.

        Args:
            indices: The indices used to get an item within this frame.
            reverse: Determines if the indices should be used in the reverse order.
            frame: Determines if the

        Returns:
            The item recursively from within this frame.
        """
        if isinstance(indices, int):
            start = indices
        elif len(indices) == 1:
            start = indices[0]
        else:
            raise IndexError("index out of range")

        if (frame is None and self.returns_frame) or frame:
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
