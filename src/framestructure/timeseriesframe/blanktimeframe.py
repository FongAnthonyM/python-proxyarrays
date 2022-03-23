""" blanktimeframe.py
A frame for holding blank time series data such as NaNs, zeros, or a single number.
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
import datetime
import math
from typing import Any

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime
import numpy as np

# Local Packages #
from ..arrayframe import BlankArrayFrame
from .timeseriesframeinterface import TimeSeriesFrameInterface


# Definitions #
# Classes #
class BlankTimeFrame(BlankArrayFrame, TimeSeriesFrameInterface):
    """A frame for holding blank time series data such as NaNs, zeros, or a single number.

    This frame does not store a blank array, rather it generates an array whenever data would be accessed.

    Attributes:
        _start: The start timestamp of this frame.
        _true_end: The true end timestamp of this frame.
        _assigned_end: The assigned end timestamp of this frame.
        _sample_rate: The sample rate of this frame.

    Args:
        start: The start time of this frame.
        end: The end time of this frame.
        sample_rate: The sample rate of this frame.
        sample_period: The sample period of this frame.
        shape: The assigned shape that this frame will be.
        dtype: The data type of the generated data.
        init: Determines if this object will construct.
    """
    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        start: datetime.datetime | float | None = None,
        end: datetime.datetime | float | None = None,
        sample_rate: float | None = None,
        sample_period: float | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        init: bool = True,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._start: float | None = None
        self._true_end: float | None = None
        self._assigned_end: float | None = None

        self._sample_rate: float | None = None

        # Construct Object #
        if init:
            self.construct(
                start=start,
                end=end,
                sample_rate=sample_rate,
                sample_period=sample_period,
                shape=shape,
                dtype=dtype,
            )

    @property
    def shape(self) -> tuple[int]:
        """The assigned shape that this frame will be."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int]) -> None:
        self._shape = value
        self.refresh()

    @property
    def start_timestamp(self) -> float:
        """The start timestamp of this frame."""
        return self._start

    @start_timestamp.setter
    def start_timestamp(self, value: datetime.datetime | float) -> None:
        if isinstance(value, datetime.datetime):
            self._start = value.timestamp()
        else:
            self._start = value
        self.refresh()

    @property
    def true_end_timestamp(self) -> float:
        """The true end_timestamp timestamp of this frame."""
        return self._true_end

    @true_end_timestamp.setter
    def true_end_timestamp(self, value: datetime.datetime | float):
        if isinstance(value, datetime.datetime):
            self._true_end = value.timestamp()
        else:
            self._true_end = value

    @property
    def assigned_end_timestamp(self) -> float:
        """The assigned end_timestamp for this frame."""
        return self._assigned_end

    @assigned_end_timestamp.setter
    def assigned_end_timestamp(self, value: datetime.datetime | float):
        if isinstance(value, datetime.datetime):
            self._assigned_end = value
        else:
            self._assigned_end = datetime.datetime.fromtimestamp(value)
        self.refresh()

    @property
    def end_timestamp(self):
        """The end timestamp for this object which is calculated based on the sample rate and start timestamp."""
        return self.true_end_timestamp

    @end_timestamp.setter
    def end_timestamp(self, value):
        self.assigned_end_timestamp = value

    @property
    def start_datetime(self) -> datetime.datetime | None:
        """The start_timestamp datetime of this frame."""
        start = self.start_timestamp
        if start is None:
            return None
        else:
            return datetime.datetime.fromtimestamp(start)

    @property
    def end_datetime(self) -> datetime.datetime | None:
        """The end_timestamp datetime of this frame."""
        end = self.end_timestamp
        if end is None:
            return None
        else:
            return datetime.datetime.fromtimestamp(end)

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this frame."""
        start = self.start_datetime
        if start is None:
            return None
        else:
            return start.date()

    @property
    def sample_rate(self) -> float:
        """The sample rate of this frame."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self._sample_rate = value
        self.refresh()

    @property
    def sample_period(self) -> float:
        """The sample period of this frame."""
        return 1 / self.sample_rate

    @sample_period.setter
    def sample_period(self, value):
        self.sample_rate = 1 / value

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        start: datetime.datetime | float | None = None,
        end: datetime.datetime | float | None = None,
        sample_rate: float | None = None,
        sample_period: float | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
    ) -> None:
        """Construct this object

        Args:
            start: The start time of this frame.
            end: The end time of this frame.
            sample_rate: The sample rate of this frame.
            sample_period: The sample period of this frame.
            shape: The assigned shape that this frame will be.
            dtype: The data type of the generated data.
        """
        if start is not None:
            if isinstance(start, datetime.datetime):
                self._start = start.timestamp()
            else:
                self._start = start

        if end is not None:
            if isinstance(end, datetime.datetime):
                self._assigned_end = end.timestamp()
            else:
                self._assigned_end = end

        if sample_period is not None:
            self._sample_rate = 1 / sample_period

        if sample_rate is not None:
            self._sample_rate = sample_rate

        super().construct(shape=shape, dtype=dtype)

        self.refresh()

    # Updating
    def refresh(self) -> None:
        """Resets the true end timestamp by calling get_length."""
        try:
            self.get_length()
        except AttributeError:
            pass

    # Getters
    def get_length(self) -> int:
        """Gets the length of the data of this frame.

        Sets the true end to the closest whole sample base on the start timestamp.

        Returns:
            The length of this frame.
        """
        start = self.start_timestamp
        end = self.assigned_end_timestamp

        size = math.floor((end - start) * self.sample_rate)
        r = math.remainder((end - start), self.sample_period)
        remain = r if r >= 0 else self.sample_period + r
        self.true_end_timestamp = end - remain
        self.shape[self.axis] = size

        return size

    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.

        Returns:
            The start_timestamp timestamp of this frame.
        """
        return self.start_timestamp

    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        return self.true_end_timestamp

    def get_sample_rate(self) -> float:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The sample rate of this frame.
        """
        return self.sample_rate

    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        return 1 / self.sample_rate

    # Shape
    def validate_shape(self) -> bool:
        """Checks if this frame has a valid/continuous shape.

        Returns:
            If this frame has a valid/continuous shape.
        """
        self.refresh()
        return True

    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the frame without changing its data.

        Args:
            shape: The shape to change this frame to.
            **kwargs: Any other kwargs for reshaping.
        """
        if self.mode == 'r':
            raise IOError("not writable")
        self.shape = shape
        self.refresh()

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this frame has a valid/continuous sampling rate.

        Returns:
            If this frame has a valid/continuous sampling rate.
        """
        self.refresh()
        return True

    def resample(self, sample_rate: float, **kwargs: Any) -> None:
        """Resamples the data to match the given sample rate.

        Args:
            sample_rate: The new sample rate for the data.
            **kwargs: Keyword arguments for the resampling.
        """
        if self.mode == 'r':
            raise IOError("not writable")
        self.sample_rate = sample_rate
        self.refresh()

    # Continuous Data
    def where_discontinuous(self, tolerance: float | None = None):
        """Generates a report on where there are sample discontinuities.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            A report on where there are discontinuities.
        """
        pass

    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this frame is continuous.
        """
        self.refresh()
        return True

    def make_continuous(self) -> None:
        """Adjusts the data to make it continuous."""
        if self.mode == 'r':
            raise IOError("not writable")
        self.refresh()

    # Get Timestamps
    def create_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Creates an array of timestamps from range style input.

        Args:
            start: The start index to get the data from.
            stop: The stop index to get the data from.
            step: The interval between data to get.
            dtype: The data type to generate.

        Returns:
            The requested timestamps.
        """
        samples = self.get_length()
        frame_start = self.start_timestamp

        if dtype is None:
            dtype = "f8"

        if start is None:
            start = 0

        if stop is None:
            stop = samples
        elif stop < 0:
            stop = samples + stop

        if step is None:
            step = 1

        if start >= samples or stop < 0:
            raise IndexError("index is out of range")

        start_timestamp = frame_start + self.sample_rate * start
        stop_timestamp = frame_start + self.sample_rate * stop
        period = self.sample_period * step

        return np.arange(start_timestamp, stop_timestamp, period, dtype=dtype)

    def create_timestamp_slice(self, slice_: slice, dtype: np.dtype | str | None = None) -> np.ndarray:
        """Creates a range of timestamps from a slice.

        Args:
            slice_: The timestamp range to create.
            dtype: The timestamp type to generate.

        Returns:
            The requested data.
        """
        return self.create_timestamp_range(start=slice_.start, stop=slice_.stop, step=slice_.step, dtype=dtype)

    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        return self.create_timestamp_range()

    def get_timestamp(self, super_index: int) -> float:
        """Get a timestamp from this frame with an index.

        Args:
            super_index: The index to get.

        Returns:
            The requested timestamp.
        """
        return self.create_timestamp_range(start=super_index, stop=super_index + 1)[0]

    def get_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Gets a range of timestamps along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.

        Returns:
            The requested range.
        """
        return self.create_timestamp_range(start=start, stop=stop, step=step)

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
        data_array[array_slice] = self.create_timestamp_slice(slice_=slice_)
        return data_array

    def get_datetimes(self) -> tuple[datetime.datetime]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        return tuple(datetime.datetime.fromtimestamp(ts) for ts in self.create_timestamp_range())

    # Find
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
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        samples = self.get_length()
        if timestamp < self.start_timestamp:
            if tails:
                return IndexDateTime(0, datetime.datetime.fromtimestamp(self.start_timestamp), self.start_timestamp)
        elif timestamp > self.end_timestamp:
            if tails:
                return IndexDateTime(samples, datetime.datetime.fromtimestamp(self.end_timestamp), self.end_timestamp)
        else:
            remain, sample = math.modf((timestamp - self.start_timestamp) * self.sample_rate)
            if approx or remain == 0:
                true_timestamp = sample / self.sample_rate + self.start_timestamp
                return IndexDateTime(int(sample), datetime.datetime.fromtimestamp(true_timestamp), true_timestamp)

        return IndexDateTime(None, None, None)
