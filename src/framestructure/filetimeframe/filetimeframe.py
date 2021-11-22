#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" filetimeframe.py
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
from abc import abstractmethod
import pathlib

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache_method

# Local Packages #
from ..timeseriesframe import TimeSeriesContainer
from ..directorytimeframe import DirectoryTimeFrameInterface


# Definitions #
# Classes #
class FileTimeFrame(TimeSeriesContainer, DirectoryTimeFrameInterface):
    file_type = None
    default_editable_type = TimeSeriesContainer

    # Class Methods #
    @classmethod
    @abstractmethod
    def validate_path(cls, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_file()

    # Magic Methods
    # Construction/Destruction
    def __init__(self, file=None, frames=None, init=True):
        # Parent Attributes #
        super().__init__(init=False)
        self.is_updating = False

        # New Attributes #
        # Containers #
        self.file = None

        # Object Construction #
        if init:
            self.construct(file=file, frames=frames)

    @property
    def data(self):
        return self.get_data()

    @data.setter
    def data(self, value):
        if value is not None:
            self.set_data(value)

    @property
    def time_axis(self):
        return self.get_time_axis()

    @time_axis.setter
    def time_axis(self, value):
        if value is not None:
            self.set_time_axis(value)

    @property
    def shape(self):
        return self.get_shape()

    @property
    def start(self):
        return self.get_start()

    @property
    def end(self):
        return self.get_end()

    @property
    def sample_rate(self):
        return self.get_sample_rate()

    @sample_rate.setter
    def sample_rate(self, value):
        if value is not None:
            raise AttributeError("can't set attribute")

    @property
    def is_continuous(self):
        return self.get_is_continuous()

    # Instance Methods
    # Constructors/Destructors
    def construct(self, file=None, frames=None, **kwargs):
        # New Assignment
        if file is not None:
            self.set_file(file)

        # Parent Construction
        super().construct(frames=frames, **kwargs)

    # Cache and Memory
    def refresh(self):
        self.load_data()
        self.get_start()
        self.get_end()
        self.get_time_axis()
        self.get_sample_rate()
        self.get_sample_period()
        self.get_is_continuous()

    # File
    def set_file(self, file):
        if isinstance(file, self.file_type):
            self.file = file
        else:
            raise ValueError("file must be a path or str")

    def open(self, mode=None, **kwargs):
        if mode is None:
            mode = self.mode
        self.file.open(mode, **kwargs)
        return self

    def close(self):
        self.file.close()

    @abstractmethod
    def load_data(self):
        pass

    # Getters
    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def get_start(self):
        pass

    @abstractmethod
    def get_end(self):
        pass

    @abstractmethod
    def get_time_axis(self):
        pass

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_sample_rate(self):
        self.get_sample_period.clear_cache()
        return self.file.sample_rate

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_is_continuous(self):
        return self.validate_continuous()

    # Setters
    @abstractmethod
    def set_data(self, value):
        if self.mode == 'r':
            raise IOError("not writable")

    @abstractmethod
    def set_time_axis(self, value):
        if self.mode == 'r':
            raise IOError("not writable")
