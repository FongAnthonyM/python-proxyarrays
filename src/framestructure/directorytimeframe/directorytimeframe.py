""" directorytimeframe.py
A frame for directory/file objects which contain time series data.
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
import pathlib
from typing import Any

# Third-Party Packages #

# Local Packages #
from ..timeseriesframe import TimeSeriesFrame
from .directorytimeframeinterface import DirectoryTimeFrameInterface


# Definitions #
# Classes #
class DirectoryTimeFrame(TimeSeriesFrame, DirectoryTimeFrameInterface):
    """A frame for directory/file objects which contain time series data.

    Class Attributes:
        default_return_frame_type: The default type of frame to return when returning a frame.
        default_frame_type: The default type frame to create from the contents of the directory.

    Attributes:
        _path: The path of the directory to wrap.
        glob_condition: The glob string to use when using the glob method.
        frame_type: The type of frame to create from the contents of the directory.
        frame_paths: The paths to the contained frames.

    Args:
        path: The path for this frame to wrap.
        frames: An iterable holding frames/objects to store in this frame.
        mode: Determines if the contents of this frame are editable or not.
        update: Determines if this frame will start_timestamp updating or not.
        open_: Determines if the frames will remain open after construction.
        **kwargs: The keyword arguments to create contained frames.
        init: Determines if this object will construct.
    """
    default_return_frame_type: type = TimeSeriesFrame
    default_frame_type: type = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path: str | pathlib.Path) -> bool:
        """Validates if the path can be used as Directory Time Frame.

        Args:
            path: The path to directory/file object that this frame will wrap.

        Returns:
            If the path is usable.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_dir()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        path: pathlib.Path | str | None = None,
        frames: Iterable[DirectoryTimeFrameInterface] | None = None,
        mode: str = 'a',
        update: bool = False,
        open_: bool = False,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._path: pathlib.Path | None = None

        self.glob_condition: str = "*"

        self.frame_type: type = self.default_frame_type
        self.frame_paths: set[pathlib.Path] = set()

        # Object Construction #
        if init:
            self.construct(path=path, frames=frames, mode=mode, update=update, open_=open_, **kwargs)

    @property
    def path(self) -> pathlib.Path:
        """The path this frame wraps."""
        return self._path

    @path.setter
    def path(self, value: pathlib.Path | str) -> None:
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    # Context Managers
    def __enter__(self) -> "DirectoryTimeFrame":
        """The context enter which opens the directory.

        Returns:
            This object.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """The context exit which closes the file."""
        self.close()

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        path: pathlib.Path | str | None = None,
        frames: Iterable[DirectoryTimeFrameInterface] | None = None,
        mode: str = 'a',
        update: bool = False,
        open_: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            path: The path for this frame to wrap.
            frames: An iterable holding frames/objects to store in this frame.
            mode: Determines if the contents of this frame are editable or not.
            update: Determines if this frame will start_timestamp updating or not.
            open_: Determines if the frames will remain open after construction.
            **kwargs: The keyword arguments to create contained frames.
        """
        super().construct(frames=frames, mode=mode, update=update)

        if path is not None:
            self.path = path

        if path is not None:
            self.construct_frames(open_=open_, mode=self.mode, **kwargs)

    def construct_frames(self, open_=False, **kwargs) -> None:
        """Constructs the frames for this object.

        Args:
            open_: Determines if the frames will remain open after construction.
            **kwargs: The keyword arguments to create contained frames.
        """
        for path in self.path.glob(self.glob_condition):
            if path not in self.frame_paths:
                if self.frame_creation_condition(path):
                    self.frames.append(self.frame_type(path, open_=open_, **kwargs))
                    self.frame_paths.add(path)
        self.frames.sort(key=lambda frame: frame.start_timestamp)
        self.clear_caches()

    # Frames
    def frame_creation_condition(self, path: str | pathlib.Path) -> bool:
        """Determines if a frame will be constructed.

        Args:
            path: The path to create a frame from.

        Returns:
            If the path can be constructed.
        """
        return self.frame_type.validate_path(path)

    # Path and File System
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
        for frame in self.frames:
            frame.open(mode, **kwargs)
        return self

    def close(self) -> None:
        """Closes this directory frame which closes all the contained frames."""
        for frame in self.frames:
            frame.close()

    def require_path(self) -> None:
        """Creates this directory if it does not exist."""
        if not self.path.is_dir():
            self.path.mkdir()

    def require_frames(self) -> None:
        """Creates the contained frames if they do not exist."""
        for frame in self.frames:
            try:
                frame.require()
            except AttributeError:
                continue

    def require(self) -> None:
        """Create this directory and all the contained frames if they do not exist."""
        self.require_path()
        self.require_frames()
