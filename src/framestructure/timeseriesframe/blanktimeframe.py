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
import decimal
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
        _true_start: The true start timestamp of this frame.
        _assigned_start: The assigned start timestamp of this frame.
        _true_end: The true end timestamp of this frame.
        _assigned_end: The assigned end timestamp of this frame.
        _sample_rate: The sample rate of this frame.
        _assigned_length: The assigned lenght of this frame.
        is_infinite: Determines if this blank frame is infinite.

    Args:
        start: The start time of this frame.
        end: The end time of this frame.
        sample_rate: The sample rate of this frame.
        sample_period: The sample period of this frame.
        shape: The assigned shape that this frame will be.
        dtype: The data type of the generated data.
        is_infinite: Determines if this blank frame is infinite.
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
        is_infinite: bool | None = None,
        init: bool = True,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._true_start: decimal.Decimal | None = None
        self._assigned_start: float | None = None
        self._true_end: decimal.Decimal | None = None
        self._assigned_end: float | None = None
        self._assigned_length: int | None = None
        self.is_infinite: bool = False

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
                is_infinite=is_infinite,
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
    def true_start_timestamp(self) -> float:
        """The true start_timestamp timestamp of this frame."""
        return float(self._true_start)

    @true_start_timestamp.setter
    def true_start_timestamp(self, value: datetime.datetime | float) -> None:
        if isinstance(value, datetime.datetime):
            self._true_start = decimal.Decimal(str(value.timestamp()))
        else:
            self._true_start = decimal.Decimal(str(value))

    @property
    def assigned_start_timestamp(self) -> float:
        """The assigned start_timestamp for this frame."""
        return self._assigned_start

    @assigned_start_timestamp.setter
    def assigned_start_timestamp(self, value: datetime.datetime | float) -> None:
        if isinstance(value, datetime.datetime):
            self._assigned_start = value.timestamp()
        else:
            self._assigned_start = value
        self.refresh()
    
    @property
    def start_timestamp(self) -> float:
        """The start timestamp of this frame."""
        return self.true_start_timestamp

    @start_timestamp.setter
    def start_timestamp(self, value: datetime.datetime | float) -> None:
        self.assigned_start_timestamp = value

    @property
    def true_end_timestamp(self) -> float:
        """The true end_timestamp timestamp of this frame."""
        return float(self._true_end)

    @true_end_timestamp.setter
    def true_end_timestamp(self, value: datetime.datetime | float) -> None:
        if isinstance(value, datetime.datetime):
            self._true_end = decimal.Decimal(str(value.timestamp()))
        else:
            self._true_end = decimal.Decimal(str(value))

    @property
    def assigned_end_timestamp(self) -> float:
        """The assigned end_timestamp for this frame."""
        return self._assigned_end

    @assigned_end_timestamp.setter
    def assigned_end_timestamp(self, value: datetime.datetime | float) -> None:
        if isinstance(value, datetime.datetime):
            self._assigned_end = value.timestamp()
        else:
            self._assigned_end = value
        self.refresh()

    @property
    def end_timestamp(self) -> float:
        """The end timestamp for this object which is calculated based on the sample rate and start timestamp."""
        return self.true_end_timestamp

    @end_timestamp.setter
    def end_timestamp(self, value) -> None:
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
        return round(1 / self.sample_rate, 9)

    @sample_period.setter
    def sample_period(self, value) -> None:
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
        is_infinite: bool | None = False,
    ) -> None:
        """Construct this object

        Args:
            start: The start time of this frame.
            end: The end time of this frame.
            sample_rate: The sample rate of this frame.
            sample_period: The sample period of this frame.
            shape: The assigned shape that this frame will be.
            dtype: The data type of the generated data.
            is_infinite: Determines if this blank frame is infinite.
        """
        if start is not None:
            if isinstance(start, datetime.datetime):
                self._assigned_start = start.timestamp()
            else:
                self._assigned_start = start

        if end is not None:
            if isinstance(end, datetime.datetime):
                self._assigned_end = end.timestamp()
            else:
                self._assigned_end = end

        if shape is not None:
            self._assigned_length = shape[self.axis]

        if sample_period is not None:
            self._sample_rate = round(1 / sample_period, 9)

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if is_infinite is not None:
            self.is_infinite = is_infinite

        super().construct(shape=shape, dtype=dtype)

        self.refresh()

    # Updating
    def refresh(self) -> None:
        """Resets the true end timestamp by calling get_length."""
        try:
            self.get_length()
        except AttributeError:
            pass

    # Getters and Setters
    def get_length(self) -> int:
        """Gets the length of the data of this frame.

        Sets the true start and end to the closest whole sample base on the set attributes.

        Returns:
            The length of this frame.
        """
        start = self.assigned_start_timestamp
        end = self.assigned_end_timestamp
        length = self._assigned_length

        if self.is_infinite:
            self._true_start = None if start is None else decimal.Decimal(str(start))
            self._true_end = None if end is None else decimal.Decimal(str(end))
            return 0

        if length is None:
            length = int((end - start) * self.sample_rate)
        elif self.sample_rate is None:
            self._sample_rate = round(length / (end - start), 9)

        if start is not None:
            start = decimal.Decimal(str(start))
            end = (start * 10 ** 9 + length * decimal.Decimal(str(self.sample_period)) * 10 ** 9) / 10 ** 9
        if end is not None:
            end = decimal.Decimal(str(end))
            start = (end * 10 ** 9 - length * decimal.Decimal(str(self.sample_period)) * 10 ** 9) / 10 ** 9
        else:
            raise ValueError("Either start or end must be assigned.")

        self._true_start = start
        self._true_end = end

        new_shape = list(self._shape)
        new_shape[self.axis] = length
        self._shape = tuple(new_shape)
        return length

    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.

        Returns:
            The start_timestamp timestamp of this frame.
        """
        return self.start_timestamp
    
    def set_start(self, value: datetime.datetime | float) -> None:
        """Sets the assigned start time.

        Args:
            value: The time to set.
        """
        if isinstance(value, datetime.datetime):
            self._assigned_start = value.timestamp()
        else:
            self._assigned_start = value

        self.refresh()

    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        return self.true_end_timestamp
    
    def set_end(self, value: datetime.datetime | float) -> None:
        """Sets the assigned end time.
        
        Args:
            value: The time to set.
        """
        if isinstance(value, datetime.datetime):
            self._assigned_end = value.timestamp()
        else:
            self._assigned_end = value
        self.refresh()
        
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
        step: int = 1,
        dtype: np.dtype | str = "f8",
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
        if not self.is_infinite:
            samples = self.get_length()

        period_ns = decimal.Decimal(str(self.sample_period)) * 10 ** 9 * step
        if self.assigned_start_timestamp is not None:
            ns = self._true_start * 10 ** 9

            if start is None:
                start = 0
            elif start < 0:
                start = samples + start

            if stop is None:
                stop = samples
            elif stop < 0:
                stop = samples + stop

            if start < 0 or stop < 0:
                raise IndexError("index is out of range")
        else:
            ns = self._true_end * 10 ** 9

            if start is None:
                start = -samples
            elif start > 0:
                start = start - samples

            if stop is None:
                stop = 0
            elif stop > 0:
                stop = stop - samples

            if start > 0 or stop > 0:
                raise IndexError("index is out of range")

        start_adjusted = int(ns + start * period_ns)
        stop_adjusted = int(ns + stop * period_ns)

        return np.arange(start_adjusted, stop_adjusted, float(period_ns), dtype=dtype) / 10 ** 9

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
        step: int = 1,
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

    # Create Data
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
        if (frame is None and self.returns_frame) or frame:
            new_blank = self.copy()
            new_blank._shape = shape
            return new_blank
        else:
            return self.create_timestamp_range(start=start, stop=stop, step=step, dtype=dtype)

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
        if slices is None:
            start = None
            stop = None
            step = 1

            shape = slice(None),
        else:
            shape = list(slices)

            slice_ = shape[self.axis]
            start = slice_.start
            stop = slice_.stop
            step = 1 if slice_.step is None else slice_.step

            shape[self.axis] = slice(None)

        return self.create_timestamp_range(start=start, stop=stop, step=step, dtype=dtype)[tuple(shape)]

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
