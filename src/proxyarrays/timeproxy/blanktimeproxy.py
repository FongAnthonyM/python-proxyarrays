"""blanktimeproxy.py
A proxy for generating time information.
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
from collections.abc import Iterable, Generator
import datetime
from decimal import Decimal
import math
from typing import Any, Callable

# Third-Party Packages #
from baseobjects import singlekwargdispatch
from baseobjects.typing import AnyCallable
from baseobjects.functions import MethodMultiplexer
from dspobjects.dataclasses import IndexDateTime
from dspobjects.time import Timestamp, NANO_SCALE, nanostamp
import numpy as np

# Local Packages #
from ..proxyarray.baseproxyarray import Slice, BaseProxyArray
from ..proxyarray import BlankProxyArray
from .basetimeproxy import BaseTimeProxy


# Definitions #
# Classes #
class BlankTimeProxy(BlankProxyArray, BaseTimeProxy):
    """A proxy for generating time information.

    This proxy does not store a blank array, rather it generates an array whenever data would be accessed.

    Attributes:
        _assigned_length: The assigned length of this proxy.
        _true_start: The true start timestamp of this proxy.
        _assigned_start: The assigned start timestamp of this proxy.
        _true_end: The true end timestamp of this proxy.
        _assigned_end: The assigned end timestamp of this proxy.
        _sample_rate: The sample rate of this proxy.
        _precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
        generate_time: The method to create time information.
        tzinfo: The time zone of the timestamps.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        is_infinite: Determines if this blank proxy is infinite.

    Args:
        start: The start time of this proxy.
        end: The end time of this proxy.
        sample_rate: The sample rate of this proxy.
        sample_period: The sample period of this proxy.
        shape: The assigned shape that this proxy will be.
        dtype: The data type of the generated data.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        is_infinite: Determines if this blank proxy is infinite.
        fill_method: The name or the function used to create the blank data.
        fill_kwargs: The keyword arguments for the fill method.
        *args: Arguments for inheritance.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    time_axis_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        start: datetime.datetime | float | np.uint64 | None = None,
        end: datetime.datetime | float | np.uint64 | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        axis: int = 0,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        is_infinite: bool | None = None,
        fill_method: str | Callable | None = None,
        fill_kwargs: dict[str, Any] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._assigned_length: int | None = None

        self._true_start: np.uint64 | None = None
        self._assigned_start: np.uint64 | None = None

        self._true_end: np.uint64 | None = None
        self._assigned_end: np.uint64 | None = None

        self._sample_rate: Decimal | None = None
        self._precise: bool = False

        self.generate_time: MethodMultiplexer = MethodMultiplexer(instance=self, select="generate_timestamp_slice")

        self.time_tolerance: float = 0.000001
        self.tzinfo: datetime.tzinfo | None = None

        self.is_infinite: bool = False

        # Parent Attributes #
        super().__init__(*args, init=False)

        # Construct Object #
        if init:
            self.construct(
                start=start,
                end=end,
                sample_rate=sample_rate,
                sample_period=sample_period,
                shape=shape,
                dtype=dtype,
                axis=axis,
                precise=precise,
                tzinfo=tzinfo,
                is_infinite=is_infinite,
                fill_method=fill_method,
                fill_kwargs=fill_kwargs,
                **kwargs,
            )

    @property
    def shape(self) -> tuple[int]:
        """The assigned shape that this proxy will be."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int]) -> None:
        self._shape = value
        self.refresh()

    @property
    def ndim(self) -> int:
        """The number of dimensions of this array."""
        return len(self._shape)

    @property
    def precise(self) -> bool:
        """Determines if this proxy returns nanostamps (True) or timestamps (False)."""
        return self._precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.set_precision(nano=value)

    @property
    def assigned_start_datetime(self) -> Timestamp | None:
        """The assigned start for this proxy."""
        return Timestamp.fromnanostamp(self._assigned_start, self.tzinfo) if self._assigned_start is not None else None

    @property
    def assigned_start_nanostamp(self) -> np.uint64 | None:
        """The assigned start nanosecond timstamp for this proxy."""
        return self._assigned_start

    @property
    def assigned_start_timestamp(self) -> float | None:
        """The assigned start timstamp for this proxy."""
        return float(self._assigned_start / NANO_SCALE) if self._assigned_start is not None else None

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start timestamp of this proxy."""
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return Timestamp.fromnanostamp(self._true_start, tz) if self._assigned_start is not None else None

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this proxy."""
        return self.start_datetime.date() if self._true_start is not None else None

    @property
    def start_nanostamp(self) -> np.uint64 | None:
        """The start nanosecond timestamp of this proxy."""
        return self._true_start

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this proxy."""
        return float(self._true_start / NANO_SCALE)

    @property
    def assigned_end_datetime(self) -> Timestamp | None:
        """The assigned end for this proxy."""
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return Timestamp.fromnanostamp(self._assigned_end, tz) if self._assigned_end is not None else None

    @property
    def assigned_end_nanostamp(self) -> np.uint64 | None:
        """The assigned end nanosecond timstamp for this proxy."""
        return self._assigned_end

    @property
    def assigned_end_timestamp(self) -> float | None:
        """The assigned end timstamp for this proxy."""
        return float(self._assigned_end / NANO_SCALE) if self._assigned_end is not None else None

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end timestamp of this proxy."""
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return Timestamp.fromnanostamp(self._true_end, tz) if self._assigned_end is not None else None

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this proxy."""
        return self.end_datetime.date() if self._true_end is not None else None

    @property
    def end_nanostamp(self) -> np.uint64 | None:
        """The end nanosecond timestamp of this proxy."""
        return self._true_end

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this proxy."""
        return float(self._true_end / NANO_SCALE)

    @property
    def sample_rate(self) -> float:
        """The sample rate of this proxy."""
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
        """The sample period of this proxy."""
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

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        start: datetime.datetime | float | np.uint64 | None = None,
        end: datetime.datetime | float | np.uint64 | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        shape: tuple[int] | None = None,
        dtype: np.dtype | str | None = None,
        axis: int | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        is_infinite: bool | None = False,
        fill_method: str | Callable | None = None,
        fill_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct this object

        Args:
            start: The start time of this proxy.
            end: The end time of this proxy.
            sample_rate: The sample rate of this proxy.
            sample_period: The sample period of this proxy.
            shape: The assigned shape that this proxy will be.
            dtype: The data type of the generated data.
            axis: The axis of the data which this proxy extends for the contained data proxies.
            precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            is_infinite: Determines if this blank proxy is infinite.
            fill_method: The name or the function used to create the blank data.
            fill_kwargs: The keyword arguments for the fill method.
            **kwargs: Keyword arguments for generating data.
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

        if sample_period is not None:
            self.sample_period = sample_period

        if sample_rate is not None:
            self._sample_rate = sample_rate if isinstance(sample_rate, Decimal) else Decimal(sample_rate)

        if is_infinite is not None:
            self.is_infinite = is_infinite

        super().construct(
            shape=shape,
            dtype=dtype,
            axis=axis,
            fill_method=fill_method,
            fill_kwargs=fill_kwargs,
            **kwargs,
        )

        if shape is not None and (a_len := shape[self.axis]) > 0:
            self._assigned_length = a_len

        self.refresh()

    def empty_copy(self, *args: Any, **kwargs: Any) -> "BlankTimeProxy":
        """Create a new copy of this object without proxies.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)

        new_copy._assigned_length = self._assigned_length

        new_copy._true_start = self._true_start
        new_copy._assigned_start = self._assigned_start

        new_copy._true_end = self._true_end
        new_copy._assigned_end = self._assigned_end

        new_copy._sample_rate = self._sample_rate
        new_copy._precise = self._precise

        new_copy.generate_time.select(self.generate_time.selected)

        new_copy.tzinfo = self.tzinfo

        new_copy.is_infinite = self.is_infinite
        return new_copy

    def create_proxy(self, type_: type[BaseTimeProxy], *args, **kwargs: Any) -> BaseTimeProxy:
        """Creates a new proxy array with the same attributes as this proxy.

        Args:
            type_: The type of proxy array to create.
            *args: The arguments for creating the proxy array.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        if issubclass(type_, BlankTimeProxy):
            kwargs = {"precise": self._precise, "tzinfo": self.tzinfo, "sample_rate": self._sample_rate} | kwargs
        return super().create_proxy(*args, type_=type_, **kwargs)

    # Updating
    def refresh(self) -> None:
        """Resets the true end timestamp by calling get_length."""
        try:
            self.get_length()
        except AttributeError:
            pass

    # Getters and Setters
    def get_length(self) -> int:
        """Gets the length of the data of this proxy.

        Sets the true start and end to the closest whole sample based on the set attributes.

        Returns:
            The length of this proxy.
        """
        start = self._assigned_start
        end = self._assigned_end
        length = self._assigned_length

        if self.is_infinite:
            self._true_start = None if start is None else start
            self._true_end = None if end is None else end
            return 0

        if length is None:
            if start is None or end is None or self._sample_rate is None:
                raise ValueError("start, end, and sample_rate must be assigned if length is unknown")
            length = int((end - start) * self._sample_rate / NANO_SCALE)
        elif self._sample_rate is None:
            if end < start:
                raise ValueError("assigned end is before start")
            self._sample_rate = Decimal(length * NANO_SCALE / (end - start))

        if length < 0:
            raise ValueError("assigned length must be positive")
        elif start is not None:
            end = start + np.uint64(length * (self.sample_period_decimal * NANO_SCALE))
        elif end is not None:
            start = end - np.uint64(length * (self.sample_period_decimal * NANO_SCALE))
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
        """Assigns the start of this proxy.

        Args:
            value: The value to assign as the start.
            is_nano: Determines if the input is in nanoseconds.
        """
        if value is None or isinstance(value, np.uint64):
            self._assigned_start = value
        else:
            self._assigned_start = nanostamp(value=value, is_nano=is_nano)

    def assign_start(self, value: datetime.datetime | float | int | np.dtype, is_nano: bool = False) -> None:
        """Assigns the start of this proxy and refreshes this object.

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
        """Assigns the end of this proxy.

        Args:
            value: The value to assign as the end.
            is_nano: Determines if the input is in nanoseconds.
        """
        if value is None or isinstance(value, np.uint64):
            self._assigned_end = value
        else:
            self._assigned_end = nanostamp(value=value, is_nano=is_nano)

    def assign_end(self, value: datetime.datetime | float | int | np.dtype, is_nano: bool = False) -> None:
        """Assigns the end of this proxy and refreshes this object.

        Args:
            value: The value to assign as the end.
            is_nano: Determines if the input is in nanoseconds.
        """
        self._assign_end(value=value, is_nano=is_nano)
        self.refresh()

    def get_sample_rate(self) -> float:
        """Gets the sample rate of this proxy.

        Returns:
            The sample rate of this proxy.
        """
        return float(self._sample_rate)

    def get_sample_rate_decimal(self) -> Decimal:
        """Gets the sample rate as Decimal object.

        Returns:
            The sample rate as Decimal object.
        """
        return self._sample_rate

    def get_sample_period(self) -> float:
        """Get the sample period of this proxy.

        Returns:
            The sample period of this proxy.
        """
        return float(1 / self._sample_rate)

    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this proxy.
        """
        return 1 / self._sample_rate

    def set_precision(self, nano: bool) -> None:
        """Sets if this proxy returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        self.generate_time.select("generate_nanostamp_slice" if nano else "generate_timestamp_slice")
        self._precise = nano

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained proxies.

        Args:
            tzinfo: The time zone to set.
        """
        self.tzinfo = tzinfo

    # Shape
    def validate_shape(self) -> bool:
        """Checks if this proxy has a valid/continuous shape.

        Returns:
            If this proxy has a valid/continuous shape.
        """
        self.refresh()
        return True

    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the proxy without changing its data.

        Args:
            shape: The shape to change this proxy to.
            **kwargs: Any other kwargs for reshaping.
        """
        if self.mode == "r":
            raise IOError("not writable")
        self.shape = shape
        self.refresh()

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this proxy has a valid/continuous sampling rate.

        Returns:
            If this proxy has a valid/continuous sampling rate.
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
            If this proxy is continuous.
        """
        self.refresh()
        return True

    def make_continuous(self) -> None:
        """Adjusts the data to make it continuous."""
        if self.mode == "r":
            raise IOError("not writable")
        self.refresh()

    def slices_proxy(self, slices: Iterable[Slice] | None = None) -> BaseProxyArray:
        """Get data as a new proxy using slices to determine the data slice.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested slice as a proxy.
        """
        start_index = slices[self.axis].start
        start = self.get_nanostamp(0 if start_index is None else start_index)
        return self.create_return_proxy(start=start, shape=self.shape_from_slices(slices))

    # Iterate Slices
    def index_islice_time(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[slice, None, None]:
        """Creates a generator which yields index slices for nanostamp slices based on time.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields slices.
        """
        # Start/Stop
        start_time = self._true_start if start is None else  nanostamp(start)
        stop_time = self._true_end if stop is None else nanostamp(stop)

        # Step
        if step is None:
            start_index, stop_index, _ = self.find_time_index_slice(start, stop, approx=True, tails=True)
            return (s for s in (slice(start_index[0], stop_index[0]),))
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Create Slices
        diff = (stop_time - start_time)
        adjustment = 0 if (diff % istep) == 0 else 1
        starts = np.array(range(0, int(diff // istep) + adjustment)) * step + start_time
        stops = starts + step
        slices = (self.find_time_index_slice(np.uint64(s), np.uint64(e), tails=True) for s, e in zip(starts, stops))
        return (slice(s[0], e[0]) for s, e, _ in slices)

    def index_islice_deltatime(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[slice, None, None]:
        """Creates a generator which yields index slices for nanostamp slices.

        Args:
            start: The start index to begin slicing.
            stop: The last index to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields slices.
        """
        # Start/Stop
        start_index = 0 if start is None else start
        stop_index = self.get_length() if stop is None else stop
        start_time = self.nanostamps[start_index]
        stop_time = self.nanostamps[stop_index - 1]

        # Step
        if step is None:
            return (s for s in (slice(start_index, stop_index),))
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Create Slices
        diff = (stop_time - start_time)
        adjustment = 0 if (diff % istep) == 0 else 1
        starts = np.array(range(0, int(diff // istep) + adjustment)) * step + start_time
        stops = starts + step
        slices = (self.find_time_index_slice(s, e, tails=True) for s, e in zip(starts, stops))
        return (slice(s[0], e[0]) for s, e, _ in slices)

    # Nanostamps
    def _generate_nanostamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Generates an array of nanosecond timestamps.

        Args:
            start: The start index of the slice to generate.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            dtype: The dtype of the returned array.

        Returns:
            The requested nanosecond timestamps.
        """
        if step is None:
            step = 1

        if dtype is None:
            dtype = "u8"

        if not self.is_infinite:
            samples = self.get_length()
        elif start is None and self._assigned_start is None:
            raise ValueError(f"Start must be specified with an infinite {self.__class__}.")
        elif stop is None and self._assigned_end is None:
            raise ValueError(f"Stop must be specified with an infinite {self.__class__}.")

        period_ns = self.sample_period_decimal * 10**9 / 1 * step
        if self._assigned_start is not None:
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

        return (np.array(range(stop - start)) * period_ns + np.uint64(ns + start * period_ns)).astype(dtype)

    @singlekwargdispatch("start")
    def generate_nanostamp_slice(
        self,
        start: int | Slice | None = None,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Generates an array of nanosecond timestamps.

        Args:
            start: Either the first super index of the slice or the slice itself to generate the data from.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            slice_: The slice to generate the data from.
            dtype: The dtype of the returned array.
            
        Returns:
            All the slices of this proxy array.
        """
        if start is None and (slice_ is None or isinstance(slice_, Slice)):
            if slice_ is not None:
                start = slice_.start
                stop = slice_.stop
                step = slice_.stop
            return self._generate_nanostamp_slice(
                start=start,
                stop=stop,
                step=step,
                dtype=dtype,
            )
        else:
            raise TypeError(f"A {type(start)} cannot be used to slice a {self.__class__}.")

    @generate_nanostamp_slice.register(Slice)
    def _generate_nanostamp_slice_slice(
        self,
        start: Slice,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Generates an array of nanosecond timestamps.

        Args:
            start: The slice to generate the data from.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            slice_: The slice to generate the data from.
            dtype: The dtype of the returned array.
            
        Returns:
            All the slices of this proxy array.
        """
        return self._generate_nanostamp_slice(start=start.start, stop=start.stop, step=start.step, dtype=dtype)

    @generate_nanostamp_slice.register(int)
    def _generate_full_slices_int(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        dtype: np.dtype | str | None = None,
    ) -> tuple[Slice, ...]:
        """Generates an array of nanosecond timestamps.

        Args:
            start: The start index of the slice to generate.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            slice_: The slice to generate the data from.
            dtype: The dtype of the returned array.

        Returns:
            The requested nanosecond timestamps.
        """
        return self._generate_nanostamp_slice(start=start, stop=stop, step=step, dtype=dtype)

    def get_nanostamps(self) -> np.ndarray:
        """Gets all the nanosecond timestamps of this proxy.

        Returns:
            A numpy array of the timestamps of this proxy.
        """
        return self.generate_nanostamp_slice()

    def get_nanostamp(self, super_index: int) -> np.uint64:
        """Get a nanosecond timestamp from this proxy with an index.

        Args:
            super_index: The index to get.

        Returns:
            The requested nanosecond timestamp.
        """
        return self.generate_nanostamp_slice(start=super_index, stop=super_index + 1)[0]

    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanosecond timestamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[array_slice] = self.generate_nanostamp_slice(slice_=slice_)
        return data_array

    def nanostamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = False,
    ) -> np.ndarray | BaseTimeProxy:
        """Gets a slice of nanosecond timestamps.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range.
        """
        ts = self.generate_nanostamp_slice(start=start, stop=stop, step=step)
        if proxy:
            return self.create_return_proxy(
                start=ts[0],
                end=ts[-1],
                shape=ts.shape,
                precise=True,
            )
        else:
            return ts

    def nanostamp_islice(
        self,
        start: int | None,
        stop: int | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[BaseTimeProxy, None, None]:
        """Creates a generator which yields nanostamps slices.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields nanostamp slices.
        """
        inner_slices = self.index_islice_deltatime(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
        )
        return (self.nanostamp_slice(s.start, s.stop, proxy=True) for s in inner_slices)

    def nanostamp_islice_time(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        istep: int = 1,
    ) -> Generator[BaseTimeProxy, None, None]:
        """Creates a generator which yields nanostamps slices based on times.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields nanostamp slices.
        """
        inner_slices = self.index_islice_time(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
        )

        return (self.nanostamp_slice(s.start, s.stop, proxy=True) for s in inner_slices)

    # Timestamps
    def generate_timestamp_slice(
        self,
        start: int | Slice | None = None,
        stop: int | None = None,
        step: int = 1,
        slice_: Slice | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        """Generates an array of nanosecond timestamps.

        Args:
            start: Either the first super index of the slice or the slice itself to generate the data from.
            stop: The length of the slice to generate.
            step: The interval of the slice to generate.
            slice_: The slice to generate the data from.
            dtype: The dtype of the returned array.

        Returns:
            All the slices of this proxy array.
        """
        ts = self.generate_nanostamp_slice(start=start, stop=stop, step=step, slice_=slice_) / 10**9
        if dtype is not None:
            return ts.astype(dtype)
        else:
            return ts

    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this proxy.

        Returns:
            A numpy array of the timestamps of this proxy.
        """
        return self.generate_timestamp_slice()

    def get_timestamp(self, super_index: int) -> float:
        """Get a timestamp from this proxy with an index.

        Args:
            super_index: The index to get.

        Returns:
            The requested timestamp.
        """
        return self.generate_timestamp_slice(start=super_index, stop=super_index + 1)[0]

    def fill_timestamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with timestamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[array_slice] = self.create_timestamp_slice(slice_=slice_)
        return data_array

    def timestamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        proxy: bool = False,
    ) -> np.ndarray | BaseTimeProxy:
        """Gets a slice of timestamps along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range.
        """
        ts = self.generate_timestamp_slice(start=start, stop=stop, step=step)
        if proxy:
            return self.create_return_proxy(
                start=ts[0],
                end=ts[-1],
                precise=True,
            )
        else:
            return ts

    # Datetimes [Timestamp]
    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this proxy based on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return Timestamp.fromnanostamp(self.generate_nanostamp_slice(index, index + 1)[0], tz=tz)

    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes as Timestamps of this proxy.

        Returns:
            All the times.
        """
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return tuple(Timestamp.fromnanostamp(ts, tz=tz) for ts in self.generate_nanostamp_slice())

    # Find Time Index
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
                true_nano_ts = np.uint64(self._true_start + sample * int(self.sample_period_decimal * int(10**9)))
                return IndexDateTime(int(sample), Timestamp.fromnanostamp(true_nano_ts))

        return IndexDateTime(None, None)

    def find_time_index_slice(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> tuple[IndexDateTime, IndexDateTime, int | float | datetime.timedelta | None]:
        """Finds the indices for a slice inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the slice.
            stop: The last time to find for the slice.
            step: The step between elements in the slice.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The slice indices.
        """
        samples = self.get_length()
        first_ns = self._true_start
        last_ns = self._true_end

        if start is None:
            start_index = IndexDateTime(0, self.start_datetime)
        else:
            start_ns = nanostamp(start)
            start_index = None
            if start_ns < first_ns:
                if tails:
                    start_index = IndexDateTime(0, self.start_datetime)
            elif start_ns > last_ns:
                if tails:
                    start_index = IndexDateTime(samples, self.end_datetime)
            else:
                remain, sample = math.modf((start_ns - first_ns) * (self._sample_rate / NANO_SCALE))
                sample = int(sample)
                if approx or remain == 0:
                    true_nano_ts = np.uint64(self._true_start + sample * (self.sample_period_decimal * 10 ** 9 // 1))
                    start_index = IndexDateTime(int(sample), Timestamp.fromnanostamp(true_nano_ts))

        if stop is None:
            stop_index = IndexDateTime(samples, self.end_datetime)
        else:
            stop_ns = nanostamp(stop)
            stop_index = None
            if stop_ns < first_ns:
                if tails:
                    stop_index = IndexDateTime(0, self.start_datetime)
            elif stop_ns > last_ns:
                if tails:
                    stop_index = IndexDateTime(samples, self.end_datetime)
            else:
                remain, sample = math.modf((stop_ns - first_ns) * (self._sample_rate / NANO_SCALE))
                sample = int(sample)
                if approx or remain == 0:
                    true_nano_ts = np.uint64(self._true_start + sample * (self.sample_period_decimal * 10 ** 9 // 1))
                    stop_index = IndexDateTime(int(sample + 1), Timestamp.fromnanostamp(true_nano_ts))

        if start_index is None:
            raise IndexError("Start out of range.")
        if stop_index is None:
            raise IndexError("Stop out of range.")

        return start_index, stop_index, step
