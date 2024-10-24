"""containertimeaxis.py
A time axis proxy container that wraps an array like object to give it time axis proxy functionality.
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
from typing import Any, Callable, Union
from warnings import warn

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.typing import AnyCallable
from baseobjects.functions import CallableMultiplexer, MethodMultiplexer
from dspobjects.dataclasses import IndexDateTime
from dspobjects.operations import nan_array
from dspobjects.time import Timestamp, nanostamp
import numpy as np

# Local Packages #
from ..proxyarray import BaseProxyArray, ContainerProxyArray
from ..timeproxy import BaseTimeProxy
from .basetimeaxis import BaseTimeAxis


# Definitions #
# Classes #
class ContainerTimeAxis(ContainerProxyArray, BaseTimeAxis):
    """A time axis proxy container that wraps an array like object to give it time axis proxy functionality.

    Attributes:
        switch_algorithm_size: Determines at what point to change the continuity checking algorithm.
        target_sample_rate: The sample rate that this proxy should be.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        _precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
        _sample_rate: The sample rate of the data.
        tzinfo: The time zone of the timestamps.
        get_data: The method for getting the correct data.
        tail_correction: The method for correcting the tails of the data.
        blank_generator: The method for creating blank data.
        _nanostamps: The nanosecond timestamps of this proxy.
        _timestamps: The timestamps of this proxy.

    Args:
        data: The numpy array for this proxy to wrap.
        time_axis: The time axis of the data.
        shape: The shape that proxy should be and if resized the shape it will default to.
        sample_rate: The sample rate of the data.
        sample_period: The sample period of this proxy.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        mode: Determines if the contents of this proxy are editable or not.
        *args: Arguments for inheritance.
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
        axis: int = 0,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str = "a",
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # System
        self.switch_algorithm_size = 10000000  # Consider chunking rather than switching

        # TimeProxy
        self._precise: bool | None = False
        self.target_sample_rate: float | None = None
        self.time_tolerance: float = 0.000001
        self._sample_rate: Decimal | None = None
        self.tzinfo: datetime.tzinfo | None = None

        # Method Assignment #
        self.get_data: MethodMultiplexer = MethodMultiplexer(instance=self, select="_get_nanostamps")
        self.tail_correction: MethodMultiplexer = MethodMultiplexer(instance=self, select="default_tail_correction")
        self.blank_generator: CallableMultiplexer = CallableMultiplexer(
            register=self.blank_generation_functions,
            instance=self,
            select="nan_array",
        )

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
                axis=axis,
                precise=precise,
                tzinfo=tzinfo,
                mode=mode,
                **kwargs,
            )

    @property
    def precise(self) -> bool:
        """Determines if this proxy returns nanostamps (True) or timestamps (False)."""
        return self._precise

    @precise.setter
    def precise(self, value: bool) -> None:
        self.set_precision(nano=value)

    @property
    def nanostamps(self) -> np.ndarray | None:
        """The nanosecond timestamps of this proxy."""
        return self.get_nanostamps()

    @nanostamps.setter
    def nanostamps(self, value: np.ndarray | None) -> None:
        self._nanostamps = value

    @property
    def day_nanostamps(self) -> np.ndarray | None:
        """The day nanosecond timestamps of this proxy."""
        return self.get_day_nanostamps()

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this proxy."""
        return self.get_timestamps()

    @timestamps.setter
    def timestamps(self, value: np.ndarray | None) -> None:
        self._timestamps = value

    @property
    def data(self) -> np.ndarray:
        """Returns the time stamp type based on the precision."""
        return self.get_data()

    @data.setter
    def data(self, value: np.ndarray) -> None:
        if self.precise:
            self.nanostamps = value
        else:
            self.timestamps = value

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start datetime of this proxy."""
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        nanostamps = self.nanostamps
        return Timestamp.fromnanostamp(nanostamps[0], tz=tz) if nanostamps is not None and any(nanostamps) else None

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this proxy."""
        start = self.start_datetime
        return start.date() if start is not None else None

    @property
    def start_nanostamp(self) -> float | None:
        """The start timestamp of this proxy."""
        nanostamps = self.nanostamps
        return nanostamps[0] if nanostamps is not None else None

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this proxy."""
        timestamps = self.timestamps
        return timestamps[0] if timestamps is not None else None

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end datetime of this proxy."""
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        nanostamps = self.nanostamps
        return Timestamp.fromnanostamp(nanostamps[-1], tz=tz) if nanostamps is not None and any(nanostamps) else None

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this proxy."""
        end = self.end_datetime
        return end.date() if end is not None else None

    @property
    def end_nanostamp(self) -> float | None:
        """The end timestamp of this proxy."""
        nanostamps = self.nanostamps
        return nanostamps[-1] if nanostamps is not None else None

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this proxy."""
        timestamps = self.timestamps
        return timestamps[-1] if timestamps is not None else None

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

    @property
    def sample_rate_decimal(self) -> Decimal:
        """The sample rate as Decimal object"""
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
        data: np.ndarray | None = None,
        shape: tuple[int] | None = None,
        sample_rate: float | str | Decimal | None = None,
        sample_period: float | str | Decimal | None = None,
        axis: int | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            data: The numpy array for this proxy to wrap.
            shape: The shape that proxy should be and if resized the shape it will default to.
            sample_rate: The sample rate of the data.
            sample_period: The sample period of this proxy.
            axis: The axis of the data which this proxy extends for the contained data proxies.
            precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            mode: Determines if the contents of this proxy are editable or not.
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

        super().construct(data=data, shape=shape, axis=axis, mode=mode, **kwargs)

    def create_proxy(self, type_: type[BaseTimeProxy], *args: Any, **kwargs: Any) -> BaseTimeProxy:
        """Creates a new proxy array with the same attributes as this proxy.

        Args:
            type_: The type of proxy array to create.
            *args: The arguments for creating the proxy array.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        if issubclass(type_, ContainerTimeAxis):
            kwargs = {"sample_rate": self.sample_rate_decimal, "precise": self._precise, "tzinfo": self.tzinfo} | kwargs
        return super().create_proxy(*args, type_=type_, **kwargs)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "ContainerProxyArray":
        """Create a new copy of this object without data.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)

        new_copy.switch_algorithm_size = self.switch_algorithm_size
        new_copy._precise = self._precise
        new_copy.target_sample_rate = self.target_sample_rate
        new_copy.time_tolerance = self.time_tolerance
        new_copy._sample_rate = self._sample_rate
        new_copy.tzinfo = self.tzinfo

        new_copy.get_data.select(self.get_data.selected)
        new_copy.tail_correction.select(self.tail_correction.selected)
        new_copy.blank_generator.select(self.blank_generator.selected)
        return new_copy

    # Getters and Setters
    def get_sample_rate(self) -> float | None:
        """Get the sample rate of this proxy from the contained proxies/objects.

        Returns:
            The sample rate of this proxy.
        """
        sample_rate = self._sample_rate
        return float(sample_rate) if sample_rate is not None else None

    def get_sample_rate_decimal(self) -> Decimal | None:
        """Get the sample rate of this proxy from the contained proxies/objects.

        Returns:
            The shape of this proxy or the minimum sample rate of the contained proxies/objects.
        """
        return self._sample_rate

    def get_sample_period(self) -> float:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

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

    def get_original_precision(self) -> bool:
        """Gets the presision of the timestamps from the orignial file.

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        return self._data.dtype == np.uint64

    def set_precision(self, nano: bool) -> None:
        """Sets if this proxy returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        self.get_data.select("_get_nanostamps" if nano else "_get_timestamps")
        self._precise = nano

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained proxies.

        Args:
            tzinfo: The time zone to set.
        """
        self.tzinfo = tzinfo

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this proxy has a valid/continuous sampling rate.

        Returns:
            If this proxy has a valid/continuous sampling rate.
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
            If this proxy is continuous.
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

    # Iterate Slices
    def index_islice_time(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
        approx: bool = True,
        tails: bool = True,
    ) -> Generator[slice, None, None]:
        """Creates a generator which yields index slices for nanostamp slices based on time.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The generator which yields slices.
        """
        # Start/Stop
        start_time = self.nanostamps[0] if start is None else nanostamp(start)
        stop_time = self.nanostamps[-1] if stop is None else nanostamp(stop)

        # Step
        if step is None:
            start_index, stop_index, _ = self.find_time_index_slice(start, stop, approx=approx, tails=tails)
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
        slices = np.zeros((len(starts), 2), dtype=int)
        slices[:, 0] = np.searchsorted(self.nanostamps, starts)
        slices[:, 1] = np.searchsorted(self.nanostamps, starts + step)
        return (slice(int(s), int(e)) for s, e in slices)

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
        slices = np.zeros((len(starts), 2), dtype=int)
        slices[:, 0] = np.searchsorted(self.nanostamps, starts)
        slices[:, 1] = np.searchsorted(self.nanostamps, starts + step)
        return (slice(int(s), int(e)) for s, e in slices)

    # Get Nanostamps
    def get_nanostamps(self) -> np.ndarray | None:
        """Gets the nanostamps of this proxy.

        Returns:
            The nanostamps of this proxy.
        """
        if self._data is None:
            return None
        elif self.get_original_precision():
            return self._data
        else:
            return (self._data * 10**9).astype(np.uint64)

    def _get_nanostamps(self) -> np.ndarray | None:
        """An alias method for getting the nanostamps of this proxy.

        Returns:
            The nanostamps of this proxy.
        """
        return self.nanostamps

    def get_nanostamp(self, super_index: int) -> float:
        """Get a time from a contained proxy with a super index.

        Args:
            super_index: The index to get the nanostamp.

        Returns:
            The nanostamp
        """
        return self.nanostamps[super_index]

    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanostamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        data_array[array_slice] = self.nanostamps[slice_]
        return data_array

    def nanostamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = True,
    ) -> np.ndarray | BaseTimeProxy:
        """Get a range of nanostamps with indices.

        Args:
            start: The start_nanostamp super index.
            stop: The stop super index.
            step: The interval between indices to get nanostamps.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range of nanostamps.
        """
        ts = self.nanostamps[slice(start, stop, step)]
        if proxy:
            return self.create_return_proxy_leaf(data=ts)
        else:
            return ts

    def nanostamp_islice(
        self,
        start: int | None,
        stop: int | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int | Decimal = 1,
    ) -> Generator["ContainerTimeAxis", None, None]:
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
        approx: bool = True,
        tails: bool = True,
    ) -> Generator["ContainerTimeAxis", None, None]:
        """Creates a generator which yields nanostamps slices based on times.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The generator which yields nanostamp slices.
        """
        inner_slices = self.index_islice_time(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
            approx=approx,
            tails=tails,
        )

        return (self.nanostamp_slice(s.start, s.stop, proxy=True) for s in inner_slices)

    # Get Day Nanostamps
    def get_day_nanostamps(self) -> np.ndarray | None:
        """Gets the day nanostamps of this proxy.

        Returns:
            The day nanostamps of this proxy.
        """
        return (self.get_nanostamps() // 864e11 * 864e11).astype("u8")

    # Get Timestamps
    def get_timestamps(self) -> np.ndarray | None:
        """Gets the timestamps of this proxy.

        Returns:
            The timestamps of this proxy.
        """
        if self._data is None:
            return None
        elif not self.get_original_precision():
            return self._data
        else:
            return self._data / 10**9

    def _get_timestamps(self) -> np.ndarray | None:
        """An alias method for getting the timestamps of this proxy.

        Returns:
            The timestamps of this proxy.
        """
        return self.timestamps

    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained proxy with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        return self.timestamps[super_index]

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
        data_array[array_slice] = self.timestamps[slice_]
        return data_array

    def timestamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = True,
    ) -> np.ndarray | BaseTimeProxy:
        """Get a range of timestamps with indices.

        Args:
            start: The start_timestamp super index.
            stop: The stop super index.
            step: The interval between indices to get timestamps.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range of timestamps.
        """
        ts = self.timestamps[slice(start, stop, step)]
        if proxy:
            return self.__class__(
                ts,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
            )
        else:
            return ts

    # Datetimes [Timestamp]
    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes of this proxy.

        Returns:
            All the times as a tuple of datetimes.
        """
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return tuple(Timestamp.fromnanostamp(ts, tz=tz) for ts in self.get_nanostamps())

    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this proxy based on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        return Timestamp.fromnanostamp(self.nanostamps[index], tz=tz)

    def fill_datetime_array(
        self,
        data_array: np.ndarray,
        array_slice: slice | None = None,
        slice_: slice | None = None,
    ) -> np.ndarray:
        """Fills a given array with nanostamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        data_array[array_slice] = tuple(Timestamp.fromnanostamp(ts, tz=tz) for ts in self.nanostamps[slice_])
        return data_array

    def datetime_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = False,
    ) -> tuple[Timestamp] | BaseTimeProxy:
        """Get a range of datetimes with indices.

        Args:
            start: The start index.
            stop: The stop index.
            step: The interval between indices to get datetimes.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range of datetimes.
        """
        ns = self.nanostamps[slice(start, stop, step)]
        if proxy:
            return self.__class__(
                ns,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
            )
        else:
            tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
            return tuple(Timestamp.fromnanostamp(ts, tz=tz) for ts in ns)

    # For Other Data
    def shift_to_nearest_sample_end(self, data: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """Shifts data to the nearest valid sample after this proxy's data.

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
        """Shifts data to the next valid sample after this proxy's data.

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
        """Shifts data to the nearest valid sample after this proxy's data or to the next valid sample after this proxy.

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

        if isinstance(correction, bool) and correction:
            correction = self.tail_correction
        elif isinstance(correction, str):
            correction = self.get_correction(correction)

        if correction and self.data.size != 0:
            data = correction(data, tolerance=tolerance)

        self.data = np.append(self.data, data, axis)

    def append_proxy(
        self,
        proxy: BaseTimeAxis,
        axis: int | None = None,
        truncate: bool | None = None,
        correction: str | bool | None = None,
    ) -> None:
        """Appends data and timestamps from another proxy to this proxy.

        Args:
            proxy: The proxy to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other proxy's data will be truncated to fit this proxy's shape.
            correction: Determines if time correction will be run on the data and the type if a str.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if truncate is None:
            truncate = self.is_truncate

        if not proxy.validate_sample_rate() or proxy.sample_rate != self.sample_rate:
            raise ValueError("the proxy's sample rate does not match this object's")

        shape = self.shape
        slices = ...
        if not proxy.validate_shape or proxy.shape != shape:
            if not truncate:
                raise ValueError("the proxy's shape does not match this object's")
            else:
                slices = [None] * len(shape)
                for index, size in enumerate(shape):
                    slices[index] = slice(None, size)
                slices[axis] = slice(None, None)
                slices = tuple(slices)

        self.append(proxy[slices], axis=axis, correction=correction)

    def add_proxies(
        self,
        proxies: Iterable[BaseTimeAxis],
        axis: int | None = None,
        truncate: bool | None = None,
    ) -> None:
        """Appends data and timestamps from other proxies to this proxy.

        Args:
            proxies: The proxies to append data from.
            axis: The axis to append the data along.
            truncate: Determines if the other proxies' data will be truncated to fit this proxy's shape.
        """
        if self.mode == "r":
            raise IOError("not writable")

        proxies = list(proxies)

        if self.data is None:
            proxy = proxies.pop(0)
            if not proxy.validate_sample_rate():
                raise ValueError("the proxy's sample rate must be valid")
            self.data = proxy[...]
            self.time_axis = proxy.get_timestamps()

        for proxy in proxies:
            self.append_proxy(proxy, axis=axis, truncate=truncate)

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
                tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
                return IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=tz))

        raise IndexError("Timestamp out of range.")
    
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
        first_ns = self.nanostamps[0]
        last_ns = self.nanostamps[-1]

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
                index = int(np.searchsorted(self.nanostamps, start_ns, side="right") - 1)
                true_timestamp = self.nanostamps[index]
                if approx or start_ns == true_timestamp:
                    tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
                    start_index = IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=tz))

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
                index = int(np.searchsorted(self.nanostamps, stop_ns))
                true_timestamp = self.nanostamps[index - 1 if index != 0 else 0]
                if approx or stop_ns == true_timestamp:
                    tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
                    stop_index = IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=tz))

        if start_index is None:
            raise IndexError("Start out of range.")
        if stop_index is None:
            raise IndexError("Stop out of range.")

        return start_index, stop_index, step

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
        tz = datetime.timezone.utc if self.tzinfo is None else self.tzinfo
        nano_ts = nanostamp(timestamp)

        samples = self.get_length()
        if nano_ts < self.day_nanostamps[0]:
            if tails:
                return IndexDateTime(0, Timestamp.fromnanostamp(self.day_nanostamps[0], tz=tz))
        elif nano_ts > self.day_nanostamps[-1]:
            if tails:
                return IndexDateTime(
                    samples,
                    Timestamp.fromnanostamp(self.day_nanostamps[-1], tz=tz),
                )
        else:
            index = int(np.searchsorted(self.day_nanostamps, nano_ts, side="right") - 1)
            true_timestamp = self.day_nanostamps[index]
            if approx or nano_ts == true_timestamp:
                return IndexDateTime(index, Timestamp.fromnanostamp(true_timestamp, tz=tz))

        raise IndexError("Timestamp out of range.")


# Assign Cyclic Definitions
ContainerTimeAxis.default_return_proxy_leaf = ContainerTimeAxis
ContainerTimeAxis.default_return_proxy_type = ContainerTimeAxis
