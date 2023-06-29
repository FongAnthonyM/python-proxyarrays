"""blanktimeframe.py
A frame for generating time information.
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
from decimal import Decimal
import math
from typing import Any

# Third-Party Packages #
from baseobjects.typing import AnyCallable
from dspobjects.dataclasses import IndexDateTime
from dspobjects.time import Timestamp, NANO_SCALE, nanostamp
import numpy as np

# Local Packages #
from ..arrayframe import BlankArrayFrame
from .timeframeinterface import TimeFrameInterface


# Definitions #
# Classes #
class BlankTimeFrame(BlankArrayFrame, TimeFrameInterface):
    """A frame for generating time information.

    This frame does not store a blank array, rather it generates an array whenever data would be accessed.

    Attributes:
        _assigned_length: The assigned length of this frame.
        _true_start: The true start timestamp of this frame.
        _assigned_start: The assigned start timestamp of this frame.
        _true_end: The true end timestamp of this frame.
        _assigned_end: The assigned end timestamp of this frame.
        _sample_rate: The sample rate of this frame.
        _create_method_: The method used to create timestamps for the create data methods.
        _precise: Determines if this frame returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        is_infinite: Determines if this blank frame is infinite.

    Args:
        start: The start time of this frame.
        end: The end time of this frame.
        sample_rate: The sample rate of this frame.
        sample_period: The sample period of this frame.
        shape: The assigned shape that this frame will be.
        dtype: The data type of the generated data.
        precise: Determines if this frame returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        is_infinite: Determines if this blank frame is infinite.
        init: Determines if this object will construct.
    """

    time_axis_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        start: datetime.datetime | float | None = None,
        end: datetime.datetime | float | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        is_infinite: bool | None = None,
        init: bool = True,
    ) -> None:
        # New Attributes #
        self._assigned_length: int | None = None

        self._true_start: np.uint64 | None = None
        self._assigned_start: np.uint64 | None = None

        self._true_end: np.uint64 | None = None
        self._assigned_end: np.uint64 | None = None

        self._sample_rate: Decimal | None = None

        self._create_method_: AnyCallable = self.create_timestamp_range.__func__
        self._precise: bool = False

        self.tzinfo: datetime.tzinfo | None = None

        self.is_infinite: bool = False

        # Parent Attributes #
        super().__init__(init=False)

        # Construct Object #
        if init:
            self.construct(
                start=start,
                end=end,
                sample_rate=sample_rate,
                sample_period=sample_period,
                shape=shape,
                dtype=dtype,
                precise=precise,
                tzinfo=tzinfo,
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
    def precise(self) -> bool:
        """Determines if this frame returns nanostamps (True) or timestamps (False)."""
        return self._precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.set_precision(nano=value)

    @property
    def assigned_start_datetime(self) -> Timestamp | None:
        """The assigned start for this frame."""
        return Timestamp.fromnanostamp(self._assigned_start, self.tzinfo) if self._assigned_start is not None else None

    @property
    def assigned_start_nanostamp(self) -> np.uint64 | None:
        """The assigned start nanosecond timstamp for this frame."""
        return self._assigned_start

    @property
    def assigned_start_timestamp(self) -> float | None:
        """The assigned start timstamp for this frame."""
        return float(self._assigned_start / NANO_SCALE) if self._assigned_start is not None else None

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start timestamp of this frame."""
        return Timestamp.fromnanostamp(self._true_start, self.tzinfo) if self._assigned_start is not None else None

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this frame."""
        return self.start_datetime.date() if self._true_start is not None else None

    @property
    def start_nanostamp(self) -> np.uint64 | None:
        """The start nanosecond timestamp of this frame."""
        return self._true_start

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this frame."""
        return float(self._true_start / NANO_SCALE)

    @property
    def assigned_end_datetime(self) -> Timestamp | None:
        """The assigned end for this frame."""
        return Timestamp.fromnanostamp(self._assigned_end, self.tzinfo) if self._assigned_end is not None else None

    @property
    def assigned_end_nanostamp(self) -> np.uint64 | None:
        """The assigned end nanosecond timstamp for this frame."""
        return self._assigned_end

    @property
    def assigned_end_timestamp(self) -> float | None:
        """The assigned end timstamp for this frame."""
        return float(self._assigned_end / NANO_SCALE) if self._assigned_end is not None else None

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end timestamp of this frame."""
        return Timestamp.fromnanostamp(self._true_end, self.tzinfo) if self._assigned_end is not None else None

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this frame."""
        return self.end_datetime.date() if self._true_end is not None else None

    @property
    def end_nanostamp(self) -> np.uint64 | None:
        """The end nanosecond timestamp of this frame."""
        return self._true_end

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this frame."""
        return float(self._true_end / NANO_SCALE)

    @property
    def sample_rate(self) -> float:
        """The sample rate of this frame."""
        return self.get_sample_rate()

    @sample_rate.setter
    def sample_rate(self, value: float | str | Decimal) -> None:
        if isinstance(value, Decimal):
            self._sample_rate = value
        else:
            self._sample_rate = Decimal(value)
        self.refresh()

    @property
    def sample_rate_decimal(self) -> Decimal:
        """The sample rate as Decimal object."""
        return self.get_sample_rate_decimal()

    @property
    def sample_period(self) -> float:
        """The sample period of this frame."""
        return self.get_sample_period()

    @sample_period.setter
    def sample_period(self, value: float | str | Decimal) -> None:
        if not isinstance(value, Decimal):
            value = Decimal(value)
        self._sample_rate = 1 / value

    @property
    def sample_period_decimal(self) -> Decimal:
        """The sample period as Decimal object"""
        return self.get_sample_period_decimal()

    @property
    def _create_method(self):
        """The selected method for creating timestamps"""
        return self._create_method_.__get__(self, self.__class__)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        start: datetime.datetime | float | None = None,
        end: datetime.datetime | float | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
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
            precise: Determines if this frame returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            is_infinite: Determines if this blank frame is infinite.
        """
        if precise is not None:
            self.set_precision(precise)

        if start is not None:
            if isinstance(start, datetime.datetime) and self.tzinfo is None:
                self.tzinfo = start.tzinfo
            self._assign_start(start)

        if end is not None:
            if isinstance(end, datetime.datetime) and self.tzinfo is None:
                self.tzinfo = end.tzinfo
            self._assign_end(end)

        if tzinfo is not None:
            self.tzinfo = tzinfo

        if shape is not None:
            self._assigned_length = shape[self.axis]

        if sample_period is not None:
            self.sample_period = sample_period

        if sample_rate is not None:
            self._sample_rate = sample_rate if isinstance(sample_rate, Decimal) else Decimal(sample_rate)

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
        start = self._assigned_start
        end = self._assigned_end
        length = self._assigned_length

        if self.is_infinite:
            self._true_start = None if start is None else start
            self._true_end = None if end is None else end
            return 0

        if length is None:
            length = int((end - start) * self._sample_rate / NANO_SCALE)
        elif self._sample_rate is None:
            if end < start:
                raise ValueError("assigned end is before start")
            self._sample_rate = Decimal(length * NANO_SCALE / (end - start))

        if length < 0:
            raise ValueError("assigned length must be positive")
        if start is not None:
            end = start + np.uint64(length) * np.uint64(self.sample_period_decimal * NANO_SCALE)
        if end is not None:
            end = Decimal(str(end))
            start = end - np.uint64(length) * np.uint64(self.sample_period_decimal * NANO_SCALE)
        else:
            raise ValueError("Either start or end must be assigned.")

        self._true_start = start
        self._true_end = end

        new_shape = list(self._shape)
        new_shape[self.axis] = length
        self._shape = tuple(new_shape)
        return length

    def _assign_start(
        self,
        value: datetime.datetime | float | int | np.dtype | None,
        is_nano: bool = False,
    ) -> None:
        """Assigns the start of this frame.

        Args:
            value: The value to assign as the start.
            is_nano: Determines if the input is in nanoseconds.
        """
        if value is None:
            self._assigned_end = None
        else:
            self._assigned_start = nanostamp(value=value, is_nano=is_nano)

    def assign_start(self, value: datetime.datetime | float | int | np.dtype, is_nano: bool = False) -> None:
        """Assigns the start of this frame and refreshes this object.

        Args:
            value: The value to assign as the start.
            is_nano: Determines if the input is in nanoseconds.
        """
        self._assign_start(value=value, is_nano=is_nano)
        self.refresh()

    def _assign_end(
        self,
        value: datetime.datetime | float | int | np.dtype | None,
        is_nano: bool = False,
    ) -> None:
        """Assigns the end of this frame.

        Args:
            value: The value to assign as the end.
            is_nano: Determines if the input is in nanoseconds.
        """
        if value is None:
            self._assigned_end = None
        else:
            self._assigned_end = nanostamp(value=value, is_nano=is_nano)

    def assign_end(self, value: datetime.datetime | float | int | np.dtype, is_nano: bool = False) -> None:
        """Assigns the end of this frame and refreshes this object.

        Args:
            value: The value to assign as the end.
            is_nano: Determines if the input is in nanoseconds.
        """
        self._assign_end(value=value, is_nano=is_nano)
        self.refresh()

    def get_sample_rate(self) -> float:
        """Gets the sample rate of this frame.

        Returns:
            The sample rate of this frame.
        """
        return float(self._sample_rate)

    def get_sample_rate_decimal(self) -> Decimal:
        """Gets the sample rate as Decimal object.

        Returns:
            The sample rate as Decimal object.
        """
        return self._sample_rate

    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        Returns:
            The sample period of this frame.
        """
        return float(1 / self._sample_rate)

    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        return 1 / self._sample_rate

    def set_precision(self, nano: bool) -> None:
        """Sets if this frame returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this frame returns nanostamps (True) or timestamps (False).
        """
        if nano:
            self._create_method_ = self.create_nanostamp_range.__func__
        else:
            self._create_method_ = self.create_timestamp_range.__func__
        self._precise = nano

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained frames.

        Args:
            tzinfo: The time zone to set.
        """
        self.tzinfo = tzinfo

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
        if self.mode == "r":
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

    def resample(self, sample_rate: float | str | Decimal, **kwargs: Any) -> None:
        """Resamples the data to match the given sample rate.

        Args:
            sample_rate: The new sample rate for the data.
            **kwargs: Keyword arguments for the resampling.
        """
        if self.mode == "r":
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
        if self.mode == "r":
            raise IOError("not writable")
        self.refresh()

    # Nanostamps
    def create_nanostamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Creates an array of nanosecond timestamps from range style input.

        Args:
            start: The start index to get the data from.
            stop: The stop index to get the data from.
            step: The interval between data to get.
            dtype: The dtype of the returned array.

        Returns:
            The requested timestamps.
        """
        if not self.is_infinite:
            samples = self.get_length()

        period_ns = self.sample_period_decimal * 10**9 // 1 * step
        if self.assigned_start_timestamp is not None:
            ns = self._true_start

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
            ns = self._true_end

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

        start_adjusted = np.uint64(ns + start * period_ns)
        stop_adjusted = np.uint64(ns + stop * period_ns)

        if dtype is not None:
            return np.arange(start_adjusted, stop_adjusted, np.uint64(period_ns), dtype="u8").astype(dtype)
        else:
            return np.arange(start_adjusted, stop_adjusted, np.uint64(period_ns), dtype="u8")

    def create_nanostamp_slice(self, slice_: slice) -> np.ndarray:
        """Creates a range of nanosecond timestamps from a slice.

        Args:
            slice_: The timestamp range to create.

        Returns:
            The requested data.
        """
        return self.create_nanostamp_range(start=slice_.start, stop=slice_.stop, step=slice_.step)

    def get_nanostamps(self) -> np.ndarray:
        """Gets all the nanosecond timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        return self.create_nanostamp_range()

    def get_nanostamp(self, super_index: int) -> np.uint64:
        """Get a nanosecond timestamp from this frame with an index.

        Args:
            super_index: The index to get.

        Returns:
            The requested nanosecond timestamp.
        """
        return self.create_nanostamp_range(start=super_index, stop=super_index + 1)[0]

    def get_nanostamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        frame: bool = False,
    ) -> np.ndarray | TimeFrameInterface:
        """Gets a range of nanosecond timestamps along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.
            frame: Determines if the returned object will be a frame.

        Returns:
            The requested range.
        """
        ts = self.create_nanostamp_range(start=start, stop=stop, step=step)
        if frame:
            return self.__class__(
                start=ts[0],
                end=ts[-1],
                sample_rate=self._sample_rate,
                precise=True,
            )
        else:
            return ts

    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanosecond timestamps from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[array_slice] = self.create_nanostamp_slice(slice_=slice_)
        return data_array

    # Timestamps
    def create_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
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
        if dtype is not None:
            return (self.create_nanostamp_range(start=start, stop=stop, step=step) / 10**9).astype(dtype)
        else:
            return self.create_nanostamp_range(start=start, stop=stop, step=step) / 10**9

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
        frame: bool = False,
    ) -> np.ndarray | TimeFrameInterface:
        """Gets a range of timestamps along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.
            frame: Determines if the returned object will be a frame.

        Returns:
            The requested range.
        """
        ts = self.create_timestamp_range(start=start, stop=stop, step=step)
        if frame:
            return self.__class__(
                start=ts[0],
                end=ts[-1],
                sample_rate=self._sample_rate,
                precise=True,
            )
        else:
            return ts

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

    # Datetimes [Timestamp]
    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this frame base on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        return Timestamp.fromnanostamp(self.create_nanostamp_range(index, index + 1)[0], tz=self.tzinfo)

    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes as Timestamps of this frame.

        Returns:
            All the times.
        """
        return tuple(Timestamp.fromnanostamp(ts, tz=self.tzinfo) for ts in self.create_nanostamp_range())

    # Find
    def find_time_index(
        self,
        timestamp: datetime.datetime | float | int | np.dtype,
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
        nano_ts = nanostamp(timestamp)

        samples = self.get_length()
        if nano_ts < self._true_start:
            if tails:
                return IndexDateTime(0, self.start_datetime)
        elif nano_ts > self._true_end:
            if tails:
                return IndexDateTime(samples, self.end_datetime)
        else:
            remain, sample = math.modf((nano_ts - self._true_start) * (self._sample_rate / NANO_SCALE))
            if approx or remain == 0:
                true_nano_ts = np.uint64(self._true_start + sample * (self.sample_period_decimal * 10**9 // 1))
                return IndexDateTime(int(sample), Timestamp.fromnanostamp(true_nano_ts))

        return IndexDateTime(None, None)
