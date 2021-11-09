#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" directorytimeframe.py
Description:
"""
# Package Header #
from ..__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #

# Local Packages #
from ..timeseriesframe import TimeSeriesFrame


# Definitions #
# Classes #
class DirectoryTimeFrame(TimeSeriesFrame):
    default_return_frame_type = TimeSeriesFrame
    default_frame_type = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_dir()

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, path=None, frames=None, mode='a', update=True, open_=False, init=True, **kwargs):
        super().__init__(init=False)
        self._path = None

        self.glob_condition = "*"

        self.is_updating_all = False
        self.is_updating_last = True

        self.frame_type = self.default_frame_type
        self.frame_names = set()

        if init:
            self.construct(path=path, frames=frames, mode=mode, update=update, open_=open_, **kwargs)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    # Context Managers
    def __enter__(self):
        """The context enter which opens the HDF5 file.

        Returns:
            This object.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The context exit which closes the file."""
        self.close()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, path=None, frames=None, mode=None, update=True, open_=False, **kwargs):
        super().construct(frames=frames, mode=mode, update=update)

        if path is not None:
            self.path = path

        if path is not None:
            self.construct_frames(open_=open_, mode=self.mode, **kwargs)

    def construct_frames(self, open_=False, **kwargs):
        for path in self.path.glob(self.glob_condition):
            if path not in self.frame_names:
                if self.frame_creation_condition(path):
                    self.frames.append(self.frame_type(path, open_=open_, **kwargs))
                    self.frame_names.add(path)
        self.frames.sort(key=lambda frame: frame.start)
        self.clear_caches()

    # Frames
    def frame_creation_condition(self, path):
        return self.frame_type.validate_path(path)

    # Get a Range of Frames
    def get_range(self, start=None, stop=None, step=None, frame=None):
        return super().get_range(start=start, stop=stop, step=step, frame=frame)

    # Path and File System
    def open(self, mode=None, **kwargs):
        if mode is None:
            mode = self.mode
        for frame in self.frames:
            frame.open(mode, **kwargs)
        return self

    def close(self):
        for frame in self.frames:
            frame.close()

    def require_path(self):
        if not self.path.is_dir():
            self.path.mkdir()

    def require_frames(self):
        for frame in self.frames():
            try:
                frame.require()
            except AttributeError:
                continue

    def require(self):
        self.require_path()
        self.require_frames()
