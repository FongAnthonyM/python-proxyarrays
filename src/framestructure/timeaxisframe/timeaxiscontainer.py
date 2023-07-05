"""timeaxiscontainer.py
A time axis frame container that wraps an array like object to give it time axis frame functionality.
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
from typing import Any, Callable

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.typing import AnyCallable
from dspobjects.dataclasses import IndexDateTime
from dspobjects.operations import nan_array
from dspobjects.time import Timestamp, nanostamp
import numpy as np

# Local Packages #
from ..arrayframe import ArrayContainer
from ..timeframe import TimeFrameInterface
from .timeaxisframeinterface import TimeAxisFrameInterface


# Definitions #
# Classes #
class TimeAxisContainer(ArrayContainer, TimeAxisFrameInterface):
    """A time axis frame container that wraps an array like object to give it time axis frame functionality.

    Attributes:
        switch_algorithm_size: Determines at what point to change the continuity checking algorithm.
        target_sample_rate: The sample rate that this frame should be.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        _precise: Determines if this frame returns nanostamps (True) or timestamps (False).
        _sample_rate: The sample rate of the data.
        tzinfo: The time zone of the timestamps.
        _data_method: The method for getting the correct data.
        _blank_generator: The method for creating blank data.
        _tail_correction: The method for correcting the tails of the data.
        _nanostamps: The nanosecond timestamps of this frame.
        _timestamps: The timestamps of this frame.

    Args:
        data: The numpy array for this frame to wrap.
        time_axis: The time axis of the data.
        shape: The shape that frame should be and if resized the shape it will default to.
        sample_rate: The sample rate of the data.
        sample_period: The sample period of this frame.
        precise: Determines if this frame returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        mode: Determines if the contents of this frame are editable or not.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for creating a new numpy array.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str = "a",
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # System
        self.switch_algorithm_size = 10000000  # Consider chunking rather than switching

        # Time
        self._precise: bool | None = False
        self.target_sample_rate: float | None = None
        self.time_tolerance: float = 0.000001
        self._sample_rate: Decimal | None = None
        self.tzinfo: datetime.tzinfo | None = None

        # Method Assignment #
        self._data_method: AnyCallable = self._get_timestamps.__func__
        self._blank_generator = nan_array
        self._tail_correction = self.default_tail_correction.__func__

        # Containers #
        self._nanostamps: np.ndarray | None = None
        self._timestamps: np.ndarray | None = None

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                data=data,
                shape=shape,
                sample_rate=sample_rate,
                sample_period=sample_period,
                precise=precise,
                tzinfo=tzinfo,
                mode=mode,
                **kwargs,
            )

    @property
    def precise(self) -> bool:
        """Determines if this frame returns nanostamps (True) or timestamps (False)."""
        return self._precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.set_precision(nano=value)

    @property
    def nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this frame."""
        try:
            return self.get_nanostamps.caching_call()
        except AttributeError:
            return self.get_nanostamps()

    @nanostamps.setter
    def nanostamps(self, value: np.ndarray | None) -> None:
        self._nanostamps = value
        self.get_nanostamps.clear_cache()
        self.get_timestamps.clear_cache()
        self.get_datetimes.clear_cache()

    @property
    def day_nanostamps(self) -> np.ndarray | None:
        """The day nanosecond timestamps of this frame."""
        try:
            return self.get_day_nanostamps.caching_call()
        except AttributeError:
            return self.get_day_nanostamps()

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this frame."""
        try:
            return self.get_timestamps.caching_call()
        except AttributeError:
            return self.get_timestamps()

    @timestamps.setter
    def timestamps(self, value: np.ndarray | None) -> None:
        self._timestamps = value
        self.get_nanostamps.clear_cache()
        self.get_timestamps.clear_cache()
        self.get_datetimes.clear_cache()

    @property
    def data(self) -> np.ndarray:
        """Returns the time stamp type based on the precision."""
        return self._data_method.__get__(self, self.__class__)()

    @data.setter
    def data(self, value: np.ndarray) -> None:
        if self.precise:
            self.nanostamps = value
        else:
            self.timestamps = value

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start datetime of this frame."""
        nanostamps = self.nanostamps
        return Timestamp.fromnanostamp(nanostamps[0], tz=self.tzinfo) if nanostamps is not None else None

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this frame."""
        start = self.start_datetime
        return start.date() if start is not None else None

    @property
    def start_nanostamp(self) -> float | None:
        """The start timestamp of this frame."""
        nanostamps = self.nanostamps
        return nanostamps[0] if nanostamps is not None else None

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this frame."""
        timestamps = self.timestamps
        return timestamps[0] if timestamps is not None else None

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end datetime of this frame."""
        nanostamps = self.nanostamps
        return Timestamp.fromnanostamp(nanostamps[-1], tz=self.tzinfo) if nanostamps is not None else None

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this frame."""
        end = self.end_datetime
        return end.date() if end is not None else None

    @property
    def end_nanostamp(self) -> float | None:
        """The end timestamp of this frame."""
        nanostamps = self.nanostamps
        return nanostamps[-1] if nanostamps is not None else None

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this frame."""
        timestamps = self.timestamps
        return timestamps[-1] if timestamps is not None else None

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

    @property
    def sample_rate_decimal(self) -> Decimal:
        """The sample rate as Decimal object"""
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
    def tail_correction(self) -> AnyCallable | None:
        """The correction method for data to be appended."""
        return self._tail_correction.__get__(self, self.__class__) if self._tail_correction is not None else None

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        shape: tuple[int] | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The numpy array for this frame to wrap.
            shape: The shape that frame should be and if resized the shape it will default to.
            sample_rate: The sample rate of the data.
            sample_period: The sample period of this frame.
            precise: Determines if this frame returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            mode: Determines if the contents of this frame are editable or not.
            **kwargs: Keyword arguments for creating a new numpy array.
        """
        if precise is not None:
            self.precise = precise

        if tzinfo is not None:
            self.tzinfo = tzinfo

        if sample_period is not None:
            self.sample_period = sample_period

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if data is not None and data.dtype == np.uint64:
            self.set_precision(True)

        super().construct(data=data, shape=shape, mode=mode, **kwargs)

    # Getters and Setters
    def get_sample_rate(self) -> float | None:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The sample rate of this frame.
        """
        sample_rate = self._sample_rate
        return float(sample_rate) if sample_rate is not None else None

    def get_sample_rate_decimal(self) -> Decimal | None:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The shape of this frame or the minimum sample rate of the contained frames/objects.
        """
        return self._sample_rate

    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

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
            self._data_method = self._get_nanostamps.__func__
        else:
            self._data_method = self._get_timestamps.__func__
        self._precise = nano

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained frames.

        Args:
            tzinfo: The time zone to set.
        """
        self.tzinfo = tzinfo

    def get_correction(self, name) -> Callable | None:
        name.lower()
        if name == "none":
            return None
        elif name == "tail":
            return self.tail_correction
        elif name == "default tail":
            return self.default_tail_correction
        elif name == "nearest end":
            return self.shift_to_nearest_sample_end
        elif name == "end":
            return self.shift_to_the_end

    def set_tail_correction(self, obj):
        if isinstance(obj, str):
            self._tail_correction = self.get_correction(obj).__func__
        else:
            self._tail_correction = obj

    def set_blank_generator(self, obj: AnyCallable) -> None:
        if isinstance(obj, str):
            obj = obj.lower()
            if obj == "nan":
                self.blank_generator = nan_array
            elif obj == "empty":
                self.blank_generator = np.empty
            elif obj == "zeros":
                self.blank_generator = np.zeros
            elif obj == "ones":
                self.blank_generator = np.ones
            elif obj == "full":
                self.blank_generator = np.full
        else:
            self.blank_generator = obj

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this frame has a valid/continuous sampling rate.

        Returns:
            If this frame has a valid/continuous sampling rate.
        """
        return self.validate_continuous()

    def resample(self, sample_rate: float, **kwargs: Any) -> None:
        """Resamples the data to match the given sample rate.

        Args:
            sample_rate: The new sample rate for the data.
            **kwargs: Keyword arguments for the resampling.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if not self.validate_sample_rate():
            raise ValueError("the data needs to have a uniform sample rate before resampling")

        nanostamps = self.nanostamps
        if nanostamps is not None:
            period_ns = np.uint64(self.sample_period_decimal * 10**9)
            self._nanostamps = np.arange(nanostamps[0], nanostamps[-1], period_ns, dtype="u8")

        timestamps = self.timestamps
        if timestamps is not None:
            self._timestamps = np.arange(timestamps[0], timestamps[-1], self.sample_period, dtype="f8")

        self.get_nanostamps.clear_cache()
        self.get_timestamps.clear_cache()

    # Continuous Data
    def where_discontinuous(self, tolerance: float | None = None) -> list | None:
        """Generates a report on where there are sample discontinuities.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            A report on where there are discontinuities.
        """
        # Todo: Get discontinuity type and make report
        if tolerance is None:
            tolerance = self.time_tolerance

        data = self.nanostamps.astype("int64")
        period_ns = np.int64(self.sample_period_decimal * 10**9)
        tolerance = np.int64(tolerance * 10**9)
        if data.shape[0] > self.switch_algorithm_size:
            discontinuous = []
            for index in range(0, len(data) - 1):
                interval = data[index] - data[index - 1]
                if abs(interval - period_ns) >= tolerance:
                    discontinuous.append(index)
        else:
            discontinuous = list(np.where(np.abs(np.ediff1d(data) - period_ns) > tolerance)[0] + 1)

        if discontinuous:
            return discontinuous
        else:
            return None

    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this frame is continuous.
        """
        if tolerance is None:
            tolerance = self.time_tolerance

        data = self.nanostamps
        period_ns = np.uint64(self.sample_period_decimal * 10**9)
        tolerance = np.uint64(tolerance * 10**9)
        if data.shape[0] > self.switch_algorithm_size:
            for index in range(0, len(data) - 1):
                interval = data[index + 1] - data[index]
                if abs(interval - period_ns) > tolerance:
                    return False
        elif False in np.abs(np.ediff1d(data) - period_ns) <= tolerance:
            return False

        return True

    def make_continuous(self, axis: int | None = None, tolerance: float | None = None) -> None:
        """Adjusts the data to make it continuous.

        Args:
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
        """
        raise NotImplemented

    # Get Nanostamps
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_nanostamps(self) -> np.ndarray | None:
        """Gets the nanostamps of this frame.

        Returns:
            The nanostamps of this frame.
        """
        if self._nanostamps is not None:
            return self._nanostamps
        elif self._timestamps is not None:
            return (self._timestamps * 10**9).astype(np.uint64)
        else:
            return None

    def _get_nanostamps(self) -> np.ndarray | None:
        """An alias method for getting the nanostamps of this frame.

        Returns:
            The nanostamps of this frame.
        """
        return self.nanostamps

    def get_nanostamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the nanostamp.

        Returns:
            The nanostamp
        """
        return self.nanostamps[super_index]

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
        ts = self.nanostamps[slice(start, stop, step)]
        if frame:
            return self.__class__(
                ts,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
            )
        else:
            return ts

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
        data_array[array_slice] = self.nanostamps[slice_]
        return data_array

    # Get Day Nanostamps
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_day_nanostamps(self) -> np.ndarray | None:
        """Gets the day nanostamps of this frame.

        Returns:
            The day nanostamps of this frame.
        """
        return (self.get_nanostamps() // 864e11 * 864e11).astype("u8")

    # Get Timestamps
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_timestamps(self) -> np.ndarray | None:
        """Gets the timestamps of this frame.

        Returns:
            The timestamps of this frame.
        """
        if self._timestamps is not None:
            return self._timestamps
        elif self._nanostamps is not None:
            return self._nanostamps / 10**9
        else:
            return None

    def _get_timestamps(self) -> np.ndarray | None:
        """An alias method for getting the timestamps of this frame.

        Returns:
            The timestamps of this frame.
        """
        return self.timestamps

    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        return self.timestamps[super_index]

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
        ts = self.timestamps[slice(start, stop, step)]
        if frame:
            return self.__class__(
                ts,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
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
        data_array[array_slice] = self.timestamps[slice_]
        return data_array

    # Datetimes [Timestamp]
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        return tuple(Timestamp.fromnanostamp(ts, tz=self.tzinfo) for ts in self.get_nanostamps())

    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this frame base on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        return Timestamp.fromnanostamp(self.nanostamps[index], tz=self.tzinfo)

    def get_datetime_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frame: bool = False,
    ) -> tuple[Timestamp] | TimeFrameInterface:
        """Get a range of datetimes with indices.

        Args:
            start: The start index.
            stop: The stop index.
            step: The interval between indices to get datetimes.
            frame: Determines if the returned object will be a frame.

        Returns:
            The requested range of datetimes.
        """
        ns = self.nanostamps[slice(start, stop, step)]
        if frame:
            return self.__class__(
                ns,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
            )
        else:
            return tuple(Timestamp.fromnanostamp(ts, tz=self.tzinfo) for ts in ns)

    def fill_datetime_array(
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
        data_array[array_slice] = tuple(Timestamp.fromnanostamp(ts, tz=self.tzinfo) for ts in self.nanostamps[slice_])
        return data_array

    # For Other Data
    def shift_to_nearest_sample_end(self, data: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """Shifts data to the nearest valid sample after this frame's data.

        Args:
            data: The data to shift.
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            The shifted data.
        """
        if tolerance is None:
            tolerance = self.time_tolerance

        if data.dtype == np.uint64:
            shift = data[0] - self.nanostamps[-1]
            period = np.uint64(self.sample_period_decimal * 10**9)
            tolerance = np.uint64(tolerance * 10**9)
        else:
            shift = data[0] - self.timestamps[-1]
            period = self.sample_period

        if shift < 0:
            raise ValueError("cannot shift data to an existing range")
        elif abs(math.remainder(shift, period)) > tolerance:
            if shift < period:
                remain = shift - period
            else:
                remain = math.remainder(shift, period)
            data -= remain

        return data

    def shift_to_the_end(self, data: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """Shifts data to the next valid sample after this frame's data.

        Args:
            data: The data to shift.
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            The shifted data.
        """
        if tolerance is None:
            tolerance = self.time_tolerance

        if data.dtype == np.uint64:
            shift = data[0] - self.nanostamps[-1]
            period = np.uint64(self.sample_period_decimal * 10**9)
            tolerance = np.uint64(tolerance * 10**9)
        else:
            shift = data[0] - self.timestamps[-1]
            period = self.sample_period

        if abs(shift - period) > tolerance:
            data -= shift - period

        return data

    def default_tail_correction(self, data: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """Shifts data to the nearest valid sample after this frame's data or to the next valid sample after this frame.

        Args:
            data: The data to shift.
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            The shifted data.
        """
        if data.dtype == np.uint64:
            shift = data[0] - self.nanostamps[-1]
        else:
            shift = data[0] - self.timestamps[-1]

        if shift >= 0:
            data = self.shift_to_nearest_sample_end(data, tolerance)
        else:
            data = self.shift_to_the_end(data, tolerance)

        return data

    # Data
    def shift_times(
        self,
        shift: np.ndarray | float | int,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None:
        """Shifts times by a certain amount.

        Args:
            shift: The amount to shift the times by
            start: The first time point to shift.
            stop: The stop time point to shift.
            step: The interval of the time points to shift.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if self._nanostamps is not None:
            self._nanostamps[start:stop:step] += shift

        if self._timestamps is not None:
            self._timestamps[start:stop:step] += shift

        self.get_nanostamps.clear_cache()
        self.get_timestamps.clear_cache()

    def append(
        self,
        data: np.ndarray,
        axis: int | None = None,
        tolerance: float | None = None,
        correction: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Appends data and timestamps onto the contained data and timestamps

        Args:
            data: The data to append.
            axis: The axis to append the data to.
            tolerance: The allowed deviation a sample can be away from the sample period.
            correction: Determines if time correction will be run on the data and the type if a str.
            **kwargs: The keyword arguments for the time correction.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if not any(data.shape):
            return

        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if correction is None or (isinstance(correction, bool) and correction):
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction and self.data.size != 0:
            data = correction(data, tolerance=tolerance)

        self.data = np.append(self.data, data, axis)

    def append_frame(
        self,
        frame: TimeAxisFrameInterface,
        axis: int | None = None,
        truncate: bool | None = None,
        correction: str | bool | None = None,
    ) -> None:
        """Appends data and timestamps from another frame to this frame.

        Args:
            frame: The frame to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other frame's data will be truncated to fit this frame's shape.
            correction: Determines if time correction will be run on the data and the type if a str.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if truncate is None:
            truncate = self.is_truncate

        if not frame.validate_sample_rate() or frame.sample_rate != self.sample_rate:
            raise ValueError("the frame's sample rate does not match this object's")

        shape = self.shape
        slices = ...
        if not frame.validate_shape or frame.shape != shape:
            if not truncate:
                raise ValueError("the frame's shape does not match this object's")
            else:
                slices = [None] * len(shape)
                for index, size in enumerate(shape):
                    slices[index] = slice(None, size)
                slices[axis] = slice(None, None)
                slices = tuple(slices)

        self.append(frame[slices], axis=axis, correction=correction)

    def add_frames(
        self,
        frames: Iterable[TimeAxisFrameInterface],
        axis: int | None = None,
        truncate: bool | None = None,
    ) -> None:
        """Appends data and timestamps from other frames to this frame.

        Args:
            frames: The frames to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other frames' data will be truncated to fit this frame's shape.
        """
        if self.mode == "r":
            raise IOError("not writable")

        frames = list(frames)

        if self.data is None:
            frame = frames.pop(0)
            if not frame.validate_sample_rate():
                raise ValueError("the frame's sample rate must be valid")
            self.data = frame[...]
            self.time_axis = frame.get_timestamps()

        for frame in frames:
            self.append_frame(frame, axis=axis, truncate=truncate)

    def get_intervals(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> np.ndarray:
        """Get the intervals between each time in the time axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each time in the time axis.
        """
        return np.ediff1d(self.data[slice(start, stop, step)])

    # Find Index
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
        if nano_ts < self.nanostamps[0]:
            if tails:
                return IndexDateTime(0, self.start_datetime)
        elif nano_ts > self.nanostamps[-1]:
            if tails:
                return IndexDateTime(samples, self.end_datetime)
        else:
            index = int(np.searchsorted(self.nanostamps, nano_ts, side="right") - 1)
            true_timestamp = self.nanostamps[index]
            if approx or nano_ts == true_timestamp:
                return IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=self.tzinfo))

        raise IndexError("Timestamp out of range.")

    def find_day_index(
        self,
        timestamp: datetime.date | float | int | np.dtype,
        approx: bool = True,
        tails: bool = False,
    ) -> IndexDateTime:
        """Finds the index with given day, can give approximate values.

        Args:
            timestamp: The timestamp to find the index for.
            approx: Determines if an approximate index will be given if the time is not present.
            tails: Determines if the first or last index will be give the requested time is outside the axis.

        Returns:
            The requested closest index and the value at that index.
        """
        nano_ts = nanostamp(timestamp)

        samples = self.get_length()
        if nano_ts < self.day_nanostamps[0]:
            if tails:
                return IndexDateTime(0, Timestamp.fromnanostamp(self.day_nanostamps[0], tz=self.tzinfo))
        elif nano_ts > self.day_nanostamps[-1]:
            if tails:
                return IndexDateTime(
                    samples,
                    Timestamp.fromnanostamp(self.day_nanostamps[-1], tz=self.tzinfo),
                )
        else:
            index = int(np.searchsorted(self.day_nanostamps, nano_ts, side="right") - 1)
            true_timestamp = self.day_nanostamps[index]
            if approx or nano_ts == true_timestamp:
                return IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=self.tzinfo))

        raise IndexError("Timestamp out of range.")
