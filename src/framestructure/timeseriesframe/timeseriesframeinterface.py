""" timeseriesframeinterface.py
An interface which outlines the basis for a time series frame.
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
from typing import Any

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime, FoundTimeRange, FoundData
import numpy as np

# Local Packages #
from ..arrayframe import ArrayFrameInterface


# Definitions #
# Classes #
class TimeSeriesFrameInterface(ArrayFrameInterface):
    """An interface which outlines the basis for a time series frame."""
    # Magic Methods #
    # Construction/Destruction
    # def __init__(self, data=None, times=True, init=True):
    #     super().__init__()
    #     self.axis = 0
    #     self.sample_rate = 0
    #
    #     self.data = None
    #     self.times = None
    #
    #     if init:
    #         self.construct(data=data, times=times)

    # Instance Methods #
    # Constructors/Destructors
    # def construct(self, data=None, times=None):
    #     if data is not None:
    #         self.data = data
    #
    #     if times is not None:
    #         self.times = times

    @property
    def is_continuous(self) -> bool:
        """If the data in the frames is continuous."""
        return self.validate_continuous()

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
    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.

        Returns:
            The start_timestamp timestamp of this frame.
        """
        pass

    @abstractmethod
    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The sample rate of this frame.
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

    # Get Timestamps
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
    ) -> np.ndarray:
        """Get a range of timestamps with indices.

        Args:
            start: The start_timestamp super index.
            stop: The stop super index.
            step: The interval between indices to get timestamps.

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
        approx: bool = False,
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

    # Find Data
    def find_timestamp_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundTimeRange:
        """Finds the timestamp range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The timestamp range on the axis and the start_timestamp and stop indices.
        """
        if isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start_index = 0
        else:
            start_index, _, _ = self.find_time_index(timestamp=start, approx=approx, tails=tails)
        
        if stop is None:
            stop_index = self.length    
        else:
            stop_index, _, _ = self.find_time_index(timestamp=stop, approx=approx, tails=tails)

        return FoundTimeRange(self.get_timestamp_range(start_index, stop_index, step), start_index, stop_index)

    def find_data_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FoundData:
        """Finds the data range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The data range on the axis and the start_timestamp and stop indices.
        """
        axis, start_index, stop_index = self.find_timestamp_range(start, stop, step, approx, tails)

        data = self.get_range(start=start_index, stop=stop_index, step=step, frame=False)

        if axis is None:
            return FoundData(None, None, None, None, None, None)
        else:
            return FoundData(data, axis, axis[0], axis[-1], start_index, stop_index)
