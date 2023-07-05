"""filetimecontainerinterface.py
A time series frame that wraps file object which contains time series.
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
import pathlib
from typing import Any

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch

# Local Packages #
from ..timeseriesframe import TimeSeriesContainer
from ..directorytimeframe import DirectoryTimeFrameInterface


# Definitions #
# Classes #
class FileTimeContainerInterface(TimeSeriesContainer, DirectoryTimeFrameInterface):
    """A time series frame that wraps file object which contains time series.

    Class Attributes:
        file_type: The file type that this file time frame will wrap.

    Attributes:
        file: The file object to wrap.

    Args:
        file: The file object to wrap or a path to the file.
        mode: The mode this frame and file will be in.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for constructing the file object.
    """

    file_type: Any = None

    # Class Methods #
    @classmethod
    @abstractmethod
    def validate_path(cls, path: str | pathlib.Path) -> bool:
        """Validates if the path can be used as Directory Time Frame.

        Args:
            path: The path to directory/file object that this frame will wrap.

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
        mode: str | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # Containers #
        self._path: pathlib.Path | None = None
        self._file: Any = None
        self.file_kwargs: dict[str, Any] = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(file=file, mode=mode, **kwargs)

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
            self._file = self.file_type(self._path, **self.file_kwargs)
        return self._file

    @file.setter
    def file(self, value: str | pathlib.Path) -> None:
        self.set_file(value)

    @property
    def data(self) -> Any:
        """The numpy data of this file."""
        return self.get_data()

    @data.setter
    def data(self, value) -> None:
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
    def construct(self, file: Any = None, mode: str | None = None, **kwargs: Any) -> None:
        """Constructs this object.

        Args:
            file: The file object to wrap or a path to the file.
            mode: The mode this frame and file will be in.
            **kwargs: The keyword arguments for constructing the file object.
        """
        # New Assignment
        if file is not None:
            self.set_file(file, mode=mode, **kwargs)

        # Parent Construction
        super().construct(mode=mode)

    # Cache and Memory
    def refresh(self) -> None:
        """Refreshes this frame."""
        # Refreshes
        self.load()

    # File
    @singlekwargdispatch("file")
    def set_file(self, file: Any, **kwargs: Any) -> None:
        """Sets the file for this frame to wrap.

        Args:
            file: The file object for this frame to wrap.
            **kwargs: The keyword arguments for constructing the file.
        """
        if isinstance(file, self.file_type):
            self._file = file
        else:
            raise TypeError(f"{type(self)} cannot set file with {type(file)}")

    @set_file.register(pathlib.Path)
    @set_file.register(str)
    def _(self, file: pathlib.Path | str, **kwargs: Any) -> None:
        """Sets the file for this frame to wrap.

        Args:
            file: The path to create the file.
            **kwargs: The keyword arguments for constructing the file.
        """
        self.path = file
        self.file_kwargs = kwargs

    def open(self, mode: str | None = None, **kwargs: Any) -> DirectoryTimeFrameInterface:
        """Opens this directory frame which opens all the contained frames.

        Args:
            mode: The mode to open all the frames in.
            **kwargs: The keyword arguments to open all the frames with.

        Returns:
            This object.
        """
        if mode is None:
            mode = self.mode
        self.file.open(mode, **kwargs)
        return self

    def close(self) -> None:
        """Closes this frame."""
        self.file.close()

    @abstractmethod
    def require(self, **kwargs: Any) -> None:
        """Create this file if it does not exist.

        Args:
            **kwargs: Keyword arguments for requiring the directory.
        """
        pass

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

    @abstractmethod
    def get_shape(self) -> tuple[int]:
        """Get the shape of this frame from the contained frames/objects.

        Returns:
            The shape of this frame.
        """
        pass
