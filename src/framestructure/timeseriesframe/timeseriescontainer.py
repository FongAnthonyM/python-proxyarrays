"""timeseriescontainer.py
A time series frame container that wraps an array like object to give it time series frame functionality.
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
from baseobjects.typing import AnyCallable
from dspobjects import Resample
from dspobjects.dataclasses import IndexDateTime
from dspobjects.operations import nan_array
from dspobjects.time import Timestamp
import numpy as np
from scipy import interpolate

# Local Packages #
from ..arrayframe import ArrayContainer
from ..timeframe import TimeFrameInterface
from ..timeaxisframe import TimeAxisContainer, TimeAxisFrameInterface
from .timeseriesframeinterface import TimeSeriesFrameInterface


# Todo: Make an interpolator object
# Todo: Make implement data mapping to reduce memory
# Definitions #
# Classes #
class TimeSeriesContainer(ArrayContainer, TimeSeriesFrameInterface):
    """A time series frame container that wraps an array like object to give it time series frame functionality.

    Attributes:
        switch_algorithm_size: Determines at what point to change the continuity checking algorithm.
        target_sample_rate: The sample rate that this frame should be.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        fill_type: The type that will fill discontinuous data.
        interpolate_type: The interpolation type for realigning data for correct times.
        interpolate_fill_value: The fill type for the missing values.
        _resampler: The object that will be used for resampling the data.
        blank_generator: The method for creating blank data.
        tail_correction: The method for correcting the tails of the data.
        time_axis: The timestamps of each sample of the data.

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
        time_axis: TimeAxisFrameInterface | np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str = "a",
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        # System
        self.switch_algorithm_size = 10000000  # Consider chunking rather than switching

        # Time
        self.target_sample_rate: float | None = None
        self.time_tolerance: float = 0.000001

        # Interpolate
        self.interpolate_type: str = "linear"
        self.interpolate_fill_value: str = "extrapolate"

        # Object Assignment
        self._resampler: Resample | None = None

        # Method Assignment
        self.blank_generator = nan_array
        self._tail_correction = self.default_tail_correction.__func__

        # Containers
        self.time_axis: TimeAxisFrameInterface | None = None

        # Object Construction #
        if init:
            self.construct(
                data=data,
                time_axis=time_axis,
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
        return self.time_axis.precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.time_axis.set_precision(nano=value)

    @property
    def nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this frame."""
        return self.get_nanostamps()

    @nanostamps.setter
    def nanostamps(self, value: np.ndarray | None) -> None:
        self.time_axis.nanostamps = value

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this frame."""
        return self.get_timestamps()

    @timestamps.setter
    def timestamps(self, value: np.ndarray | None) -> None:
        self.timestamps = value

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start datetime of this frame."""
        return self.time_axis.start_datetime

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this frame."""
        return self.time_axis.start_date

    @property
    def start_nanostamp(self) -> float | None:
        """The start timestamp of this frame."""
        return self.time_axis.start_nanostamp

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this frame."""
        return self.time_axis.start_timestamp

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end datetime of this frame."""
        return self.time_axis.end_datetime

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this frame."""
        return self.time_axis.end_date

    @property
    def end_nanostamp(self) -> float | None:
        """The end timestamp of this frame."""
        return self.time_axis.end_nanostamp

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this frame."""
        return self.time_axis.end_timestamp

    @property
    def sample_rate(self) -> float:
        """The sample rate of this frame."""
        return self.time_axis.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float | str | Decimal) -> None:
        self.time_axis.sample_rate = value

    @property
    def sample_rate_decimal(self) -> Decimal:
        """The sample rate as Decimal object"""
        return self.time_axis.sample_rate.decimal

    @property
    def sample_period(self) -> float:
        """The sample period of this frame."""
        return self.time_axis.sample_period

    @sample_period.setter
    def sample_period(self, value: float | str | Decimal) -> None:
        self.time_axis.sample_period = value

    @property
    def sample_period_decimal(self) -> Decimal:
        """The sample period as Decimal object"""
        return self.time_axis.sample_period_decimal

    @property
    def resampler(self) -> Resample:
        """The object that can resample the data in this frame container."""
        if self._resampler is None:
            self.construct_resampler()
        return self._resampler

    @resampler.setter
    def resampler(self, value: Resample) -> None:
        self._resampler = value

    @property
    def tail_correction(self) -> AnyCallable | None:
        """The correction method for data to be appended."""
        return self._tail_correction.__get__(self, self.__class__) if self._tail_correction is not None else None

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        time_axis: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
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
            time_axis: The time axis of the data.
            shape: The shape that frame should be and if resized the shape it will default to.
            sample_rate: The sample rate of the data.
            sample_period: The sample period of this frame.
            precise: Determines if this frame returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            mode: Determines if the contents of this frame are editable or not.
            **kwargs: Keyword arguments for creating a new numpy array.
        """
        is_axis_type = isinstance(time_axis, TimeAxisFrameInterface)
        if time_axis is not None and not is_axis_type:
            self.time_axis = TimeAxisContainer(
                data=time_axis,
                sample_rate=sample_rate,
                sample_period=sample_period,
                precise=precise,
                tzinfo=tzinfo,
                mode=mode,
            )
        else:
            if is_axis_type:
                self.time_axis = time_axis

            if sample_period is not None:
                self.time_axis.sample_period = sample_period

            if sample_rate is not None:
                self.time_axis.sample_rate = sample_rate

            if precise is not None:
                self.time_axis.set_precision(precise)

            if tzinfo is not None:
                self.time_axis.set_tzinfo(tzinfo)

        super().construct(data=data, shape=shape, mode=mode, **kwargs)

    def construct_resampler(self) -> Resample:
        """Constructs the resampler for this frame.

        Returns:
            The resampler.
        """
        self.resampler = Resample(old_fs=self.sample_rate, new_fs=self.target_sample_rate, axis=self.axis)
        return self.resampler

    # Getters and Setters
    def get_sample_rate(self) -> float:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The sample rate of this frame.
        """
        return self.time_axis.get_sample_rate()

    def get_sample_rate_decimal(self) -> Decimal:
        """Get the sample rate of this frame from the contained frames/objects.

        Returns:
            The shape of this frame or the minimum sample rate of the contained frames/objects.
        """
        return self.time_axis.get_sample_rate_decimal()

    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        return self.time_axis.get_sample_period()

    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        return self.time_axis.get_sample_period_decimal()

    def set_precision(self, nano: bool) -> None:
        """Sets if this frame returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this frame returns nanostamps (True) or timestamps (False).
        """
        self.time_axis.set_precision(nano)

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained frames.

        Args:
            tzinfo: The time zone to set.
        """
        self.time_axis.set_tzinfo(tzinfo)

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
            self._tail_correction = self.get_correction(obj)
        else:
            self._tail_correction = obj

    def set_blank_generator(self, obj):
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

    def get_time_intervals(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Get the intervals between each sample of the axis.

        Args:
            start: The start index to get the intervals.
            stop: The last index to get the intervals.
            step: The step of the indices to the intervals.

        Returns:
            The intervals between each datum of the axis.
        """
        return self.time_axis.get_intervals(start=start, stop=stop, step=step)

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this frame has a valid/continuous sampling rate.

        Returns:
            If this frame has a valid/continuous sampling rate.
        """
        return self.time_axis.validate_continuous()

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

        # Todo: Make Resample for multiple frames (maybe edit resampler from an outer layer)
        self.data = self.resampler(data=self.data[...], new_fs=self.sample_rate, **kwargs)
        self.time_axis.resample(sample_rate=sample_rate)

    # Continuous Data
    def where_discontinuous(self, tolerance: float | None = None):
        """Generates a report on where there are sample discontinuities.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            A report on where there are discontinuities.
        """
        return self.time_axis.where_discontinuous(tolerance)

    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this frame is continuous.
        """
        return self.time_axis.validate_continuous(tolerance)

    def make_continuous(self, axis: int | None = None, tolerance: float | None = None) -> None:
        """Adjusts the data to make it continuous.

        Args:
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
        """
        self.time_correction_interpolate(axis=axis, tolerance=tolerance)
        self.fill_time_correction(axis=axis, tolerance=tolerance)

    # Time Correction
    def time_correction_interpolate(
        self,
        axis: int | None = None,
        interp_type: str | None = None,
        fill_value: str | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Corrects the data if it is time miss aligned by interpolating the data.

        Args:
            axis: The axis to apply the time correction.
            interp_type: The interpolation type for the interpolation.
            fill_value: The fill type for the missing values on the edge of data.
            tolerance: The allowed deviation a sample can be away from the sample period.
            **kwargs: The keyword arguments for the interpolator.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if interp_type is None:
            interp_type = self.interpolate_type

        if fill_value is None:
            fill_value = self.interpolate_fill_value

        discontinuities = self.where_discontinuous(tolerance=tolerance)
        while discontinuities:
            discontinuity = discontinuities.pop(0)
            timestamp = self.time_axis[discontinuity]
            previous = discontinuity - 1
            previous_timestamp = self.time_axis[previous]

            if (timestamp - previous_timestamp) < (2 * self.sample_period):
                consecutive = [previous, discontinuity]
                start = previous_timestamp + self.sample_period
            else:
                consecutive = [discontinuity]
                nearest = round((timestamp - previous_timestamp) * self.sample_rate)
                start = previous_timestamp + self.sample_period * nearest

            if discontinuities:
                for next_d in discontinuities:
                    if (self.time_axis[next_d] - self.time_axis[next_d - 1]) < (2 * self.sample_period):
                        consecutive.append(discontinuities.pop(0))
                    else:
                        consecutive.append(next_d - 1)
                        break
            else:
                consecutive.append(len(self.time_axis) - 1)

            new_size = consecutive[-1] + 1 - consecutive[0]
            end = start + self.sample_period * (new_size - 1)
            new_times = np.arange(start, end, self.sample_period)
            if new_size > 1:
                times = self.time_axis[consecutive[0] : consecutive[-1] + 1]
                data = self.get_range(consecutive[0], consecutive[-1] + 1)
                interpolator = interpolate.interp1d(times, data, interp_type, axis, fill_value=fill_value, **kwargs)
                self.set_range(interpolator(new_times), start=discontinuity)
            else:
                self.time_axis[discontinuity] = start

    def fill_time_correction(self, axis: int | None = None, tolerance: float | None = None, **kwargs: Any) -> None:
        """Fill empty sections of the data with blank values.

        Args:
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
            **kwargs: The keyword arguments for the blank data generator.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        discontinuities = self.where_discontinuous(tolerance=tolerance)

        if discontinuities:
            offsets = np.empty((0, 2), dtype="i")
            gap_discontinuities = []
            previous_discontinuity = 0
            for discontinuity in discontinuities:
                timestamp = self.time_axis[discontinuity]
                previous = discontinuity - 1
                previous_timestamp = self.time_axis[previous]
                if (timestamp - previous_timestamp) >= (2 * self.sample_period):
                    real = discontinuity - previous_discontinuity
                    blank = round((timestamp - previous_timestamp) * self.sample_rate) - 1
                    offsets = np.append(offsets, [[real, blank]], axis=0)
                    gap_discontinuities.append(discontinuities)
                    previous_discontinuity = discontinuity
            offsets = np.append(offsets, [[self.time_axis - discontinuities[-1], 0]], axis=0)

            new_size = np.sum(offsets)
            new_shape = list(self.data.shape)
            new_shape[axis] = new_size
            old_data = self.data
            old_times = self.time_axis
            self.data = self.blank_generator(shape=new_shape, **kwargs)
            self.time_axis = np.empty((new_size,), dtype="f8")
            old_start = 0
            new_start = 0
            for discontinuity, offset in zip(gap_discontinuities, offsets):
                previous = discontinuity - 1
                new_mid = new_start + offset[0]
                new_end = new_mid + offset[1]
                mid_timestamp = old_times[previous] + self.sample_period
                end_timestamp = offset[1] * self.sample_period

                slice_ = slice(start=old_start, stop=old_start + offset[0])
                slices = [slice(None, None)] * len(old_data.shape)
                slices[axis] = slice_

                self.set_range(old_data[tuple(slices)], start=new_start)

                self.time_axis[new_start:new_mid] = old_times[slice_]
                self.time_axis[new_mid:new_end] = np.arange(mid_timestamp, end_timestamp, self.sample_period)

                old_start = discontinuity
                new_start += sum(offset)

    # Get Nanostamps
    def get_nanostamps(self) -> np.ndarray | None:
        """Gets all the nanostamps of this frame.

        Returns:
            A numpy array of the nanostamps of this frame.
        """
        return self.time_axis.get_nanostamps()

    def get_nanostamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the nanostamp.

        Returns:
            The nanostamp
        """
        return self.time_axis.get_nanostamp(super_index)

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
        return self.time_axis.get_nanostamp_range(start, stop, step, frame)

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
        return self.time_axis.fill_nanostamps_array(data_array, array_slice, slice_)

    # Get Timestamps
    def get_timestamps(self) -> np.ndarray | None:
        """Gets all the timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        return self.time_axis.get_timestamps()

    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        return self.time_axis.get_timestamp(super_index)

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
        return self.time_axis.get_timestamp_range(start, stop, step, frame)

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
        return self.time_axis.fill_timestamps_array(data_array, array_slice, slice_)

    # Datetimes [Timestamp]
    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this frame base on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        return self.time_axis.get_datetime(index=index)

    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        return self.time_axis.get_datetimes()

    # Other Data Methods
    def interpolate_shift_other(
        self,
        y: np.ndarray,
        x: np.ndarray,
        shift: np.ndarray | float | int,
        interp_type: str | None = None,
        axis: int = 0,
        fill_value: str | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolates given data and returns the data that has shifted along the x axis.

        Args:
            y: The data to interpolate.
            x: The x axis of the data to interpolate.
            shift: The amount to shift the x axis by.
            interp_type: The interpolation type for the interpolation.
            axis: The axis to apply the interpolation.
            fill_value: The fill type for the missing values on the edge of data.
            **kwargs: The keyword arguments for the interpolator.

        Returns:
            The interpolated values.
        """
        if interp_type is None:
            interp_type = self.interpolate_type

        if fill_value is None:
            fill_value = self.interpolate_fill_value

        interpolator = interpolate.interp1d(x, y, interp_type, axis, fill_value=fill_value, **kwargs)
        new_x = x + shift
        new_y = interpolator(new_x)

        return new_x, new_y

    def shift_to_nearest_sample_end(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        axis: int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shifts data to the nearest valid sample after this frame's data.

        Args:
            data: The data to shift.
            time_axis: The timestamp axis of the data.
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
            **kwargs: The keyword arguments for the interpolator.

        Returns:
            The shifted data.
        """
        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if time_axis.dtype == np.uint64:
            shift = time_axis[0] - self.nanostamps[-1]
            period = np.uint64(self.sample_period_decimal * 10**9)
            tolerance = np.uint64(tolerance * 10**9)
        else:
            shift = time_axis[0] - self.timestamps[-1]
            period = self.sample_period

        if shift < 0:
            raise ValueError("cannot shift data to an existing range")
        elif abs(math.remainder(shift, period)) > tolerance:
            if shift < period:
                remain = shift - period
            else:
                remain = math.remainder(shift, period)
            small_shift = -remain
            data, time_axis = self.interpolate_shift_other(data, time_axis, small_shift, axis=axis, **kwargs)

        return data, time_axis

    def shift_to_the_end(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        axis: int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shifts data to the next valid sample after this frame's data, if its time is beyond a valid sample.

        Args:
            data: The data to shift.
            time_axis: The timestamp axis of the data.
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
            **kwargs: The keyword arguments for the interpolator.

        Returns:
            The shifted data.
        """
        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if time_axis.dtype == np.uint64:
            shift = time_axis[0] - self.nanostamps[-1]
            period = np.uint64(self.sample_period_decimal * 10**9)
            tolerance = np.uint64(tolerance * 10**9)
        else:
            shift = time_axis[0] - self.timestamps[-1]
            period = self.sample_period

        if abs(shift - period) > tolerance:
            data, time_axis = self.interpolate_shift_other(data, time_axis, period - shift, axis=axis, **kwargs)

        return data, time_axis

    def default_tail_correction(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        axis: int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shifts data to the next valid sample after this frame's data, if its time is beyond a valid sample.

        Args:
            data: The data to shift.
            time_axis: The timestamp axis of the data.
            axis: The axis to apply the time correction.
            tolerance: The allowed deviation a sample can be away from the sample period.
            **kwargs: The keyword arguments for the interpolator.

        Returns:
            The shifted data.
        """
        if time_axis.dtype == np.uint64:
            shift = time_axis[0] - self.nanostamps[-1]
        else:
            shift = time_axis[0] - self.timestamps[-1]

        if shift >= 0:
            data, time_axis = self.shift_to_nearest_sample_end(data, time_axis, axis, tolerance, **kwargs)
        else:
            data, time_axis = self.shift_to_the_end(data, time_axis, axis, tolerance, **kwargs)

        return data, time_axis

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

        self.time_axis.shift_times(shift, start, stop, step)

    def append(
        self,
        data: np.ndarray,
        time_axis: np.ndarray | None = None,
        axis: int | None = None,
        tolerance: float | None = None,
        correction: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Appends data and timestamps onto the contained data and timestamps

        Args:
            data: The data to append.
            time_axis: The timestamps of the data.
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
            data, time_axis = correction(data, time_axis, axis=axis, tolerance=tolerance, **kwargs)

        self.data = np.append(self.data, data, axis)
        self.time_axis = self.time_axis.append(time_axis)

    def append_frame(
        self,
        frame: TimeSeriesFrameInterface,
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

        self.append(frame[slices], frame.time_axis[...], axis, correction=correction)

    def add_frames(
        self,
        frames: Iterable[TimeSeriesFrameInterface],
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
        return np.ediff1d(self.time_axis[slice(start, stop, step)])

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
        return self.time_axis.find_time_index(timestamp=timestamp, approx=approx, tails=tails)

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
        return self.time_axis.find_day_index(timestamp=timestamp, approx=approx, tails=tails)


# Assign Cyclic Definitions
TimeSeriesContainer.time_series_type = TimeSeriesContainer
