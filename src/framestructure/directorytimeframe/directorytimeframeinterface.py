"""directorytimeframeinterface.py
An interface which outlines a directory/file object to be used as time series frame.
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
from collections.abc import Iterable
import datetime
from decimal import Decimal
import pathlib
from typing import Any

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime
from dspobjects.time import Timestamp
import numpy as np

# Local Packages #
from ..arrayframe import ArrayFrameInterface
from ..timeframe import TimeFrameInterface
from ..timeseriesframe import TimeSeriesFrameInterface


# Definitions #
# Classes #
class DirectoryTimeFrameInterface(TimeSeriesFrameInterface):
    """An interface which outlines a directory/file object to be used as time series frame."""

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
        pass

    # Magic Methods
    # Construction/Destruction
    # def __init__(self, data=None, times=True, init=True):
    #     self.axis = 0
    #     self.sample_rate = 0
    #
    #     self.data = None
    #     self.times = None
    #
    #     if init:
    #         self.construct(data=data, times=times)

    # Instance Methods
    # Constructors/Destructors
    # def construct(self, data=None, times=None):
    #     if data is not None:
    #         self.data = data
    #
    #     if times is not None:
    #         self.times = times

    # Numpy ndarray Methods
    @abstractmethod
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        pass

    # Instance Methods #
    # File
    @abstractmethod
    def open(self, mode: str | None = None, **kwargs: Any) -> "DirectoryTimeFrameInterface":
        """Opens this directory frame which opens all the contained frames.

        Args:
            mode: The mode to open all the frames in.
            **kwargs: The keyword arguments to open all the frames with.

        Returns:
            This object.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes this frame."""
        pass

    @abstractmethod
    def require(self, **kwargs: Any) -> None:
        """Create this directory and all the contained frames if they do not exist.

        Args:
            **kwargs: Keyword arguments for requiring the directory.
        """
        pass

    # Getters
    @abstractmethod
    def get_shape(self) -> tuple[int]:
        """Get the shape of this frame from the contained frames/objects.

        Returns:
            The shape of this frame.
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """Gets the length of this frame.

        Returns:
            The length of this frame.
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get the sample rate of this frame from the contained frames/objects.

         If the contained frames/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this frame or the minimum sample rate of the contained frames/objects.
        """
        pass

    @abstractmethod
    def get_sample_rate_decimal(self) -> Decimal:
        """Get the sample rate of this frame from the contained frames/objects.

         If the contained frames/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this frame or the minimum sample rate of the contained frames/objects.
        """
        pass

    @abstractmethod
    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        pass

    @abstractmethod
    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        pass

    @abstractmethod
    def set_precision(self, nano: bool) -> None:
        """Sets if this frame returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this frame returns nanostamps (True) or timestamps (False).
        """
        pass

    @abstractmethod
    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained frames.

        Args:
            tzinfo: The time zone to set.
        """
        pass

    @abstractmethod
    def get_item(self, item: Any) -> Any:
        """Gets an item from within this frame based on an input item.

        Args:
            item: The object to be used to get a specific item within this frame.

        Returns:
            An item within this frame.
        """
        pass

    # Shape
    @abstractmethod
    def validate_shape(self) -> bool:
        """Checks if this frame has a valid/continuous shape.

        Returns:
            If this frame has a valid/continuous shape.
        """
        pass

    @abstractmethod
    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the frame without changing its data."""
        pass

    # Sample Rate
    @abstractmethod
    def validate_sample_rate(self) -> bool:
        """Checks if this frame has a valid/continuous sampling rate.

        Returns:
            If this frame has a valid/continuous sampling rate.
        """
        pass

    @abstractmethod
    def resample(self, sample_rate: float, **kwargs: Any) -> None:
        """Resamples the data to match the given sample rate.

        Args:
            sample_rate: The new sample rate for the data.
            **kwargs: Keyword arguments for the resampling.
        """
        pass

    # Continuous Data
    @abstractmethod
    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this frame is continuous.
        """
        pass

    @abstractmethod
    def make_continuous(self) -> None:
        """Adjusts the data to make it continuous."""
        pass

    # Get Nanostamps
    @abstractmethod
    def get_nanostamps(self) -> np.ndarray:
        """Gets all the nanostamps of this frame.

        Returns:
            A numpy array of the nanostamps of this frame.
        """
        pass

    @abstractmethod
    def get_nanostamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the nanostamp.

        Returns:
            The nanostamp
        """
        pass  # return self.time[super_index]

    @abstractmethod
    def get_nanostamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frame: bool = True,
    ) -> np.ndarray | TimeFrameInterface:
        """Get a range of nanostamps with indices.

        Args:
            start: The start_nanostamp super index.
            stop: The stop super index.
            step: The interval between indices to get nanostamps.
            frame: Determines if the returned object will be a frame.

        Returns:
            The requested range of nanostamps.
        """
        pass  # return self.times[slice(start_nanostamp, stop, step)]

    @abstractmethod
    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanostamps from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        pass

    # Get Timestamps
    @abstractmethod
    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this frame base on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        pass

    @abstractmethod
    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        pass

    @abstractmethod
    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        pass  # return self.time[super_index]

    @abstractmethod
    def get_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frame: bool = True,
    ) -> np.ndarray | TimeFrameInterface:
        """Get a range of timestamps with indices.

        Args:
            start: The start_timestamp super index.
            stop: The stop super index.
            step: The interval between indices to get timestamps.
            frame: Determines if the returned object will be a frame.

        Returns:
            The requested range of timestamps.
        """
        pass  # return self.times[slice(start_timestamp, stop, step)]

    @abstractmethod
    def fill_timestamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with timestamps from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        pass

    # Datetimes [Timestamp]
    @abstractmethod
    def get_datetimes(self) -> tuple[datetime.datetime]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        pass

    # Get Data
    @abstractmethod
    def get_slices_array(self, slices: Iterable[slice | int | None] | None = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested range as an array.
        """
        pass

    @abstractmethod
    def fill_slices_array(
        self,
        data_array: np.ndarray,
        array_slices: Iterable[slice] | None = None,
        slices: Iterable[slice | int | None] | None = None,
    ) -> np.ndarray:
        """Fills a given array with values from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slices: The slices to fill within the data_array.
            slices: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        pass

    @abstractmethod
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
        pass

    # Find Index
    @abstractmethod
    def find_time_index(
        self,
        timestamp: datetime.datetime | float,
        approx: bool = True,
        tails: bool = False,
    ) -> IndexDateTime:
        """Finds the index with given time, can give approximate values.

        Args:
            timestamp: The timestamp to find the index for.
            approx: Determines if an approximate index will be given if the time is not present.
            tails: Determines if the first or last index will be give the requested time is outside the axis.

        Returns:
            The requested closest index and the value at that index.
        """
        pass
