"""basecontainerfiletimeseries.py
A time series proxy that wraps file object which contains time series.
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
from abc import abstractmethod
from collections.abc import Iterable, Generator
import pathlib
from typing import Any

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
import numpy as np

# Local Packages #
from ..timeseries import ContainerTimeSeries
from ..directorytimeseries import BaseDirectoryTimeSeries


# Definitions #
# Classes #
class BaseContainerFileTimeSeries(ContainerTimeSeries, BaseDirectoryTimeSeries):
    """A time series proxy that wraps file object which contains time series.

    Class Attributes:
        default_remain_open: Determines if new files will remain open on initial opening.
        file_type: The file type that this file time proxy will wrap.

    Attributes:
        file: The file object to wrap.
        remain_open: Determines if new files will remain open on initial opening.

    Args:
        file: The file object to wrap or a path to the file.
        mode: The mode this proxy and file will be in.
        path: The path of the file to wrap.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for constructing the file object.
    """

    default_remain_open: bool = False
    file_type: Any = None

    # Class Methods #
    @classmethod
    @abstractmethod
    def validate_path(cls, path: str | pathlib.Path) -> bool:
        """Validates if the path can be used as Directory TimeProxy proxy.

        Args:
            path: The path to directory/file object that this proxy will wrap.

        Returns:
            If the path is usable.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_file()

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        file: Any = None,
        mode: str | None = "r",
        *,
        path: str | pathlib.Path | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._path: pathlib.Path | None = None
        self._file: Any = None
        self.remain_open: bool = self.default_remain_open
        self.file_kwargs: dict[str, Any] = {}

        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Object Construction #
        if init:
            self.construct(file=file, mode=mode, path=path, **kwargs)

    @property
    def path(self) -> pathlib.Path:
        """The path to the data file."""
        return self._path

    @path.setter
    def path(self, value: str | pathlib.Path) -> None:
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    @property
    def file(self) -> pathlib.Path:
        """The file object."""
        if self._file is None:
            self._file = self.file_type(self._path, mode=self.mode, **self.file_kwargs)
        return self._file

    @file.setter
    def file(self, value: str | pathlib.Path) -> None:
        self.set_file(value)

    @property
    def _data(self) -> Any:
        """The numpy data of this file."""
        return self.get_data()

    @_data.setter
    def _data(self, value) -> None:
        if value is not None:
            self.set_data(value)

    @property
    def time_axis(self) -> Any:
        """The timestamp axis of this file."""
        return self.get_time_axis()

    @time_axis.setter
    def time_axis(self, value: Any) -> None:
        if value is not None:
            self.set_time_axis(value)

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        file: Any = None,
        mode: str | None = None,
        *,
        path: str | pathlib.Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            file: The file object to wrap or a path to the file.
            mode: The mode this proxy and file will be in.
            path: The path of the file to wrap.
            **kwargs: The keyword arguments for constructing the file object.
        """
        # New Assignment
        if path is not None:
            self.path = path

        if file is not None:
            self.set_file(file)

        if kwargs:
            self.file_kwargs.update(kwargs)

        # Parent Construction
        super().construct(mode=mode)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "BaseContainerFileTimeSeries":
        """Create a new copy of this object without data.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)

        new_copy._path = self._path

        new_copy.remain_open = self.remain_open
        new_copy.file_kwargs = self.file_kwargs

        return new_copy

    # Cache and Memory
    def refresh(self) -> None:
        """Refreshes this proxy."""
        # Refreshes
        self.load()

    # File
    @singlekwargdispatch("file")
    def set_file(self, file: Any) -> None:
        """Sets the file for this proxy to wrap.

        Args:
            file: The file object for this proxy to wrap.
        """
        if isinstance(file, self.file_type):
            self._file = file
        else:
            raise TypeError(f"{type(self)} cannot set file with {type(file)}")

    @set_file.register(pathlib.Path)
    @set_file.register(str)
    def __set_file(self, file: pathlib.Path | str) -> None:
        """Sets the file for this proxy to wrap.

        Args:
            file: The path to create the file.
        """
        self.path = file

    def open(self, mode: str | None = None, **kwargs: Any) -> BaseDirectoryTimeSeries:
        """Opens this directory proxy which opens all the contained proxies.

        Args:
            mode: The mode to open all the proxies in.
            **kwargs: The keyword arguments to open all the proxies with.

        Returns:
            This object.
        """
        if mode is None:
            mode = self.mode
        if self._file is None:
            self._file = self.file_type(self._path, mode=self.mode, **self.file_kwargs)
        else:
            self._file.open(mode, **kwargs)
        return self

    def close(self) -> None:
        """Closes this proxy."""
        if self._file is not None:
            self._file.close()

    @abstractmethod
    def load(self) -> None:
        """Loads the file's information into memory.'"""
        pass

    # Getters
    @abstractmethod
    def get_data(self) -> Any:
        """Gets the data.

        Returns:
            The data object.
        """
        pass

    @abstractmethod
    def set_data(self, value: Any) -> None:
        """Sets the data.

        Args:
            value: A data object.
        """
        if self.mode == "r":
            raise IOError("not writable")

    @abstractmethod
    def get_time_axis(self) -> Any:
        """Gets the time axis.

        Returns:
            The time axis object.
        """
        pass

    @abstractmethod
    def set_time_axis(self, value: Any) -> None:
        """Sets the time axis

        Args:
            value: A time axis object.
        """
        if self.mode == "r":
            raise IOError("not writable")

    def should_cache(self) -> bool:
        """Determines data should be cached for methods.

        Returns:
            The decision if data should be cached.
        """
        return self.data.nbytes <= 2000000000

    def islices(
        self,
        slices: Iterable[slice | int | None] | None = None,
        islice: slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Generator[ContainerTimeSeries | np.ndarray, None, None]:
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

        if self.should_cache():
            data = self.data[...]
            for inner_start in range(outer_start, adjusted_stop, outer_step):
                full_slices[axis] = slice(inner_start, inner_start + slice_size, axis_step)

                if dtype is None:
                    data_slice = data[tuple(full_slices)]
                else:
                    data_slice = data[tuple(full_slices)].astype(dtype)

                if proxy:
                    yield self.create_return_proxy(data=data_slice)
                else:
                    yield data_slice
        else:
            for inner_start in range(outer_start, adjusted_stop, outer_step):
                full_slices[axis] = slice(inner_start, inner_start + slice_size, axis_step)
                yield self.slices(slices=full_slices, dtype=dtype, proxy=proxy)