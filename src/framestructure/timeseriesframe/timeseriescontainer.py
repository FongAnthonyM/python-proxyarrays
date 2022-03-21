""" timeseriescontainer.py
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
import math
from typing import Any, Callable

# Third-Party Packages #
from dspobjects import Resample
from dspobjects.dataclasses import IndexValue, IndexDateTime
import numpy as np
from scipy import interpolate

# Local Packages #
from ..arrayframe import ArrayContainer
from .timeseriesframeinterface import TimeSeriesFrameInterface


# Todo: Make an interpolator object
# Todo: Make a time_axis object
# Todo: Make implement data mapping to reduce memory
# Todo: Fix correction.
# Definitions #
# Classes #
class TimeSeriesContainer(ArrayContainer, TimeSeriesFrameInterface):
    """A time series frame container that wraps an array like object to give it time series frame functionality.

    Attributes:
        switch_algorithm_size: Determines at what point to change the continuity checking algorithm.
        _date: An assigned date if this object does not have a start_timestamp timestamp yet.
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
        mode: Determines if the contents of this frame are editable or not.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for creating a new numpy array.
    """
    # Static Methods #
    @staticmethod
    def create_nan_array(shape=None, **kwargs):
        a = np.empty(shape=shape, **kwargs)
        a.fill(np.nan)
        return a

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        data: np.ndarray | None = None,
        time_axis: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        sample_rate: float | None = None,
        mode: str = 'a',
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        # Descriptors #
        # System
        self.switch_algorithm_size = 10000000  # Consider chunking rather than switching

        # Time
        self._date: datetime.date | None = None
        self.target_sample_rate: float | None = None
        self.time_tolerance: float = 0.000001
        self.sample_rate: float | None = None

        # Interpolate
        self.interpolate_type: str = "linear"
        self.interpolate_fill_value: str = "extrapolate"

        # Object Assignment #
        self._resampler: Resample | None = None

        # Method Assignment #
        self.blank_generator = self.create_nan_array
        self.tail_correction = self.default_tail_correction

        # Containers #
        self.time_axis: np.ndarray | None = None

        # Object Construction #
        if init:
            self.construct(
                data=data,
                time_axis=time_axis,
                shape=shape,
                sample_rate=sample_rate,
                mode=mode,
                **kwargs,
            )

    @property
    def start_timestamp(self) -> float | None:
        """The start_timestamp timestamp of this frame."""
        return self.get_start_timestamp()

    @property
    def end_timestamp(self) -> float | None:
        """The end_timestamp timestamp of this frame."""
        return self.get_end_timestamp()

    @property
    def start_datetime(self) -> datetime.datetime | None:
        """The start_timestamp datetime of this frame."""
        return datetime.datetime.fromtimestamp(self.get_start_timestamp())

    @property
    def end_datetime(self) -> datetime.datetime | None:
        """The end_timestamp datetime of this frame."""
        return datetime.datetime.fromtimestamp(self.get_end_timestamp())

    @property
    def start_date(self) -> datetime.date | None:
        """The start_timestamp date of the data in this frame."""
        start = self.start_datetime
        if start is None:
            return self._date
        else:
            return start.date()

    @property
    def sample_period(self) -> float:
        """The period between samples."""
        return self.get_sample_period()

    @property
    def resampler(self) -> Resample:
        """The object that can resample the data in this frame container."""
        if self._resampler is None:
            self.construct_resampler()
        return self._resampler

    @resampler.setter
    def resampler(self, value: Resample) -> None:
        self._resampler = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        data: np.ndarray | None = None,
        time_axis: np.ndarray | None = None,
        shape: Iterable[int] | None = None,
        sample_rate: float | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The numpy array for this frame to wrap.
            time_axis: The time axis of the data.
            shape: The shape that frame should be and if resized the shape it will default to.
            sample_rate: The sample rate of the data.
            mode: Determines if the contents of this frame are editable or not.
            **kwargs: Keyword arguments for creating a new numpy array.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if time_axis is not None:
            self.time_axis = time_axis

        super().construct(data=data, shape=shape, mode=mode, **kwargs)

    def construct_resampler(self) -> Resample:
        """Constructs the resampler for this frame.

        Returns:
            The resampler.
        """
        self.resampler = Resample(old_fs=self.sample_rate, new_fs=self.target_sample_rate, axis=self.axis)
        return self.resampler

    # Getters
    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.

        Returns:
            The start_timestamp timestamp of this frame.
        """
        return self.time_axis[0]

    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        return self.time_axis[-1]

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
            self.tail_correction = self.get_correction(obj)
        else:
            self.tail_correction = obj

    def set_blank_generator(self, obj):
        if isinstance(obj, str):
            obj = obj.lower()
            if obj == "nan":
                self.blank_generator = self.create_nan_array
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
        if self.mode == 'r':
            raise IOError("not writable")

        if not self.validate_sample_rate():
            raise ValueError("the data needs to have a uniform sample rate before resampling")

        # Todo: Make Resample for multiple frames (maybe edit resampler from an outer layer)
        self.data = self.resampler(data=self.data[...], new_fs=self.sample_rate, **kwargs)
        self.time_axis = np.arange(self.time_axis[0], self.time_axis[-1], self.sample_period, dtype="f8")

    # Continuous Data
    def where_discontinuous(self, tolerance: float | None = None):
        """Generates a report on where there are sample discontinuities.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            A report on where there are discontinuities.
        """
        # Todo: Get discontinuity type and make report
        if tolerance is None:
            tolerance = self.time_tolerance

        if self.time_axis.shape[0] > self.switch_algorithm_size:
            discontinuous = []
            for index in range(0, len(self.time_axis) - 1):
                interval = self.time_axis[index] - self.time_axis[index - 1]
                if abs(interval - self.sample_period) >= tolerance:
                    discontinuous.append(index)
        else:
            discontinuous = list(
                np.where(np.abs(np.ediff1d(self.time_axis) - self.sample_period) > tolerance)[0] + 1)

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

        if self.time_axis.shape[0] > self.switch_algorithm_size:
            for index in range(0, len(self.time_axis) - 1):
                interval = self.time_axis[index + 1] - self.time_axis[index]
                if abs(interval - self.sample_period) > tolerance:
                    return False
        elif False in np.abs(np.ediff1d(self.time_axis) - self.sample_period) <= tolerance:
            return False

        return True

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
        if self.mode == 'r':
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
                times = self.time_axis[consecutive[0]: consecutive[-1] + 1]
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
        if self.mode == 'r':
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

    # Get Timestamps
    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this frame.

        Returns:
            A numpy array of the timestamps of this frame.
        """
        return self.time_axis.copy()

    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained frame with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        return self.time_axis[super_index]

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
        return self.time_axis[slice(start, stop, step)]

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
        data_array[array_slice] = self.time_axis[slice_]
        return data_array

    def get_datetimes(self) -> tuple[datetime.datetime]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        return tuple(datetime.datetime.fromtimestamp(ts) for ts in self.time_axis)

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

        shift = time_axis[0] - self.time_axis[-1]
        if shift < 0:
            raise ValueError("cannot shift data to an existing range")
        elif shift - self.sample_period > tolerance:
            if round(shift * self.sample_rate) < 1:
                remain = shift - self.sample_period
            else:
                remain = math.remainder(shift, self.sample_period)
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

        shift = time_axis[0] - self.time_axis[-1]
        if abs(shift - self.sample_period) > tolerance:
            small_shift = - math.remainder(shift, self.sample_period)
            large_shift = shift - self.sample_period + small_shift

            data, time_axis = self.interpolate_shift_other(data, time_axis, small_shift, axis=axis, **kwargs)
            time_axis -= large_shift

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
        shift = time_axis[0] - self.time_axis[-1]
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
        if self.mode == 'r':
            raise IOError("not writable")

        self.time_axis[start:stop:step] += shift

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
        if self.mode == 'r':
            raise IOError("not writable")

        if axis is None:
            axis = self.axis

        if tolerance is None:
            tolerance = self.time_tolerance

        if correction is None or (isinstance(correction, bool) and correction):
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction:
            data, time_axis = correction(data, time_axis, axis=axis, tolerance=tolerance, **kwargs)

        self.data = np.append(self.data, data, axis)
        self.time_axis = np.append(self.time_axis, time_axis, 0)

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
        if self.mode == 'r':
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

        self.append(frame[slices], frame.get_timestamps(), axis, correction=correction)

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
        if self.mode == 'r':
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
        if timestamp < self.time_axis[0]:
            if tails:
                return IndexDateTime(0, datetime.datetime.fromtimestamp(self.start_timestamp), self.start_timestamp)
        elif timestamp > self.time_axis[-1]:
            if tails:
                return IndexDateTime(samples, datetime.datetime.fromtimestamp(self.end_timestamp), self.end_timestamp)
        else:
            index = int(np.searchsorted(self.time_axis, timestamp, side="right") - 1)
            true_timestamp = self.time_axis[index]
            if approx or timestamp == true_timestamp:
                return IndexDateTime(index, datetime.datetime.fromtimestamp(true_timestamp), true_timestamp)

        return IndexDateTime(None, None, None)
