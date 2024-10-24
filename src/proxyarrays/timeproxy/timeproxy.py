"""timeproxy.py
A proxy for holding time information.
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
from collections.abc import Generator, Iterable
from decimal import Decimal
import datetime
from typing import Any, Union
from warnings import warn

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from baseobjects.functions import MethodMultiplexer
from baseobjects.typing import AnyCallable
from dspobjects.dataclasses import IndexValue, IndexDateTime
from dspobjects.operations import nan_array
from dspobjects.time import Timestamp, nanostamp
import numpy as np

# Local Packages #
from ..proxyarray import ProxyArray, BaseProxyArray
from .basetimeproxy import BaseTimeProxy
from .blanktimeproxy import BlankTimeProxy


# Definitions #
# Classes #
class TimeProxy(ProxyArray, BaseTimeProxy):
    """A ProxyArray that has been expanded to handle time data.

    Class Attributes:
        default_fill_type: The default type to fill discontinuous data.

    Attributes:
        target_sample_rate: The sample rate that this proxy should be.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        tzinfo: The time zone of the timestamps.
        fill_type: The type that will fill discontinuous data.

    Args:
        proxies: An iterable holding proxies/objects to store in this proxy.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
        tzinfo: The time zone of the timestamps.
        mode: Determines if the contents of this proxy are editable or not.
        update: Determines if this proxy will start updating or not.
        *args: Arguments for inheritance.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    default_fill_type = BlankTimeProxy
    time_axis_type = None

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        proxies: Iterable[BaseTimeProxy] | None = None,
        axis: int = 0,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str = "a",
        update: bool = True,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._precise: bool = True
        self.target_sample_rate: float | None = None
        self.time_tolerance: float = 0.000001
        self.tzinfo: datetime.tzinfo | None = None

        self.get_data: MethodMultiplexer = MethodMultiplexer(instance=self, select="_get_nanostamps")

        self.fill_type: type = self.default_fill_type

        # Parent Attributes #
        super().__init__(*args, init=False)

        # Object Construction #
        if init:
            self.construct(
                proxies=proxies,
                axis=axis,
                precise=precise,
                tzinfo=tzinfo,
                mode=mode,
                update=update,
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
        try:
            return self.get_nanostamps.caching_call()
        except AttributeError:
            return self.get_nanostamps()

    @property
    def timestamps(self) -> np.ndarray | None:
        """The timestamps of this proxy."""
        try:
            return self.get_timestamps.caching_call()
        except AttributeError:
            return self.get_timestamps()

    @property
    def data(self) -> np.ndarray:
        """Returns the time stamp type based on the precision."""
        return self.get_data()

    @property
    def start_datetimes(self) -> tuple[Timestamp | None]:
        """The start datetimes of this proxy."""
        try:
            return self.get_start_datetimes.caching_call()
        except AttributeError:
            return self.get_start_datetimes()

    @property
    def start_datetime(self) -> Timestamp | None:
        """The start datetime of this proxy."""
        return self.proxies[0].start_datetime if self.proxies else None

    @property
    def start_date(self) -> datetime.date | None:
        """The start date of the data in this proxy."""
        start = self.start_datetime
        return start.date() if start is not None else None

    @property
    def start_nanostamps(self) -> tuple[Timestamp | None]:
        """The start nanostamps of this proxy."""
        try:
            return self.get_start_nanostamps.caching_call()
        except AttributeError:
            return self.get_start_nanostamps()

    @property
    def start_nanostamp(self) -> float | None:
        """The start timestamp of this proxy."""
        return self.proxies[0].start_nanostamp if self.proxies else None

    @property
    def start_timestamps(self) -> tuple[Timestamp | None]:
        """The start timestamps of this proxy."""
        try:
            return self.get_start_timestamps.caching_call()
        except AttributeError:
            return self.get_start_timestamps()

    @property
    def start_timestamp(self) -> float | None:
        """The start timestamp of this proxy."""
        return self.proxies[0].start_timestamp if self.proxies else None

    @property
    def end_datetimes(self) -> tuple[Timestamp | None]:
        """The end datetimes of this proxy."""
        try:
            return self.get_end_datetimes.caching_call()
        except AttributeError:
            return self.get_end_datetimes()

    @property
    def end_datetime(self) -> Timestamp | None:
        """The end datetime of this proxy."""
        return self.proxies[-1].end_datetime if self.proxies else None

    @property
    def end_date(self) -> datetime.date | None:
        """The end date of the data in this proxy."""
        end = self.end_datetime
        return end.date() if end is not None else None

    @property
    def end_nanostamps(self) -> tuple[Timestamp | None]:
        """The end nanostamps of this proxy."""
        try:
            return self.get_end_nanostamps.caching_call()
        except AttributeError:
            return self.get_end_nanostamps()

    @property
    def end_nanostamp(self) -> float | None:
        """The end timestamp of this proxy."""
        return self.proxies[-1].end_nanostamp if self.proxies else None

    @property
    def end_timestamps(self) -> tuple[Timestamp | None]:
        """The end timestamps of this proxy."""
        try:
            return self.get_end_timestamps.caching_call()
        except AttributeError:
            return self.get_end_timestamps()

    @property
    def end_timestamp(self) -> float | None:
        """The end timestamp of this proxy."""
        return self.proxies[-1].end_timestamp if self.proxies else None

    @property
    def sample_rates(self) -> tuple[float]:
        """The sample rates of the contained proxies."""
        try:
            return self.get_sample_rates.caching_call()
        except AttributeError:
            return self.get_sample_rates()

    @property
    def sample_rates_decimal(self) -> tuple[Decimal]:
        """The sample rates as Decimal objects."""
        try:
            return self.get_sample_rates_decimal.caching_call()
        except AttributeError:
            return self.get_sample_rates_decimal()

    @property
    def sample_rate(self) -> float:
        """The sample rate of this proxy."""
        try:
            return self.get_sample_rate.caching_call()
        except AttributeError:
            return self.get_sample_rate()

    @property
    def sample_rate_decimal(self) -> Decimal:
        """The sample rate as Decimal object"""
        try:
            return self.get_sample_rate_decimal.caching_call()
        except AttributeError:
            return self.get_sample_rate_decimal()

    @property
    def sample_periods(self) -> tuple[float]:
        """The sample period of the contained proxies."""
        try:
            return self.get_sample_periods.caching_call()
        except AttributeError:
            return self.get_sample_periods()

    @property
    def sample_periods_decimal(self) -> tuple[Decimal]:
        """The sample rates as Decimal objects."""
        try:
            return self.get_sample_periods_decimal.caching_call()
        except AttributeError:
            return self.get_sample_periods_decimal()

    @property
    def sample_period(self) -> float:
        """The sample period of this proxy."""
        try:
            return self.get_sample_period.caching_call()
        except AttributeError:
            return self.get_sample_period()

    @property
    def sample_period_decimal(self) -> Decimal:
        """The sample period as Decimal object"""
        try:
            return self.get_sample_period_decimal.caching_call()
        except AttributeError:
            return self.get_sample_period_decimal()

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        proxies: Iterable[BaseProxyArray] = None,
        axis: int | None = None,
        precise: bool | None = None,
        tzinfo: datetime.tzinfo | None = None,
        mode: str | None = None,
        update: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            proxies: An iterable holding proxies/objects to store in this proxy.
            axis: The axis of the data which this proxy extends for the contained data proxies.
            precise: Determines if this proxy returns nanostamps (True) or timestamps (False).
            tzinfo: The time zone of the timestamps.
            mode: Determines if the contents of this proxy are editable or not.
            update: Determines if this proxy will start updating or not.
            **kwargs: Keyword arguments for inheritance.
        """
        super().construct(proxies=proxies, axis=axis, mode=mode, update=update)

        if precise is not None:
            self.set_precision(precise)

        if tzinfo is not None:
            self.set_tzinfo(tzinfo)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "TimeProxy":
        """Create a new copy of this object without proxies.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)

        new_copy._precise = self._precise
        new_copy.target_sample_rate = self.target_sample_rate
        new_copy.time_tolerance = self.time_tolerance
        new_copy.tzinfo = self.tzinfo

        new_copy.get_data.select(self.get_data.selected)

        new_copy.fill_type = self.fill_type
        return new_copy

    # Sorting
    def proxy_sort_key(self, proxy: Any) -> Any:
        """The key to be used in sorting with the proxy as the sort basis.

        Args:
            proxy: The proxy to sort.
        """
        return proxy.start_timestamp

    # Cache and Memory
    def refresh(self) -> None:
        """Resets this proxy's caches and fills them with updated values."""
        super().refresh()
        self.get_start_timestamps()
        self.get_end_timestamps()
        self.get_sample_rates()
        self.get_sample_rate()
        self.get_sample_period()

    # Getters and Setters
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_start_datetimes(self) -> tuple[Timestamp | None]:
        """Get the start datetimes of all contained proxies.

        Returns:
            All the start datetimes.
        """
        starts = [None] * len(self.proxies)
        for index, proxy in enumerate(self.proxies):
            starts[index] = proxy.start_datetime
        return tuple(starts)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_start_nanostamps(self) -> np.ndarray:
        """Get the start nanostamps of all contained proxies.

        Returns:
            All the start nanostamps.
        """
        starts = nan_array(len(self.proxies), dtype="u8")
        for index, proxy in enumerate(self.proxies):
            start = proxy.start_nanostamp
            if start is not None:
                starts[index] = start
        return starts

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_start_timestamps(self) -> np.ndarray:
        """Get the start timestamps of all contained proxies.

        Returns:
            All the start timestamps.
        """
        starts = nan_array(len(self.proxies), dtype="f8")
        for index, proxy in enumerate(self.proxies):
            start = proxy.start_timestamp
            if start is not None:
                starts[index] = start
        return starts

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_end_datetimes(self) -> tuple[Timestamp | None]:
        """Get the end datetimes of all contained proxies.

        Returns:
            All the end datetimes.
        """
        ends = [None] * len(self.proxies)
        for index, proxy in enumerate(self.proxies):
            ends[index] = proxy.end_datetime
        return tuple(ends)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_end_nanostamps(self) -> np.ndarray:
        """Get the end nanostamps of all contained proxies.

        Returns:
            All the end nanostamps.
        """
        ends = nan_array(len(self.proxies), dtype="u8")
        for index, proxy in enumerate(self.proxies):
            end = proxy.end_nanostamp
            if end is not None:
                ends[index] = end
        return ends

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_end_timestamps(self) -> np.ndarray:
        """Get the end timestamps of all contained proxies.

        Returns:
            All the end timestamps.
        """
        ends = nan_array(len(self.proxies), dtype="f8")
        for index, proxy in enumerate(self.proxies):
            end = proxy.end_timestamp
            if end is not None:
                ends[index] = end
        return ends

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_rates(self) -> tuple[float]:
        """Get the sample rates of all contained proxies.

        Returns:
            The sample rates of all contained proxies.
        """
        return tuple(proxy.get_sample_rate() for proxy in self.proxies)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_rates_decimal(self) -> tuple[Decimal]:
        """Get the sample rates of all contained proxies.

        Returns:
            The sample rates of all contained proxies.
        """
        return tuple(proxy.get_sample_rate_decimal() for proxy in self.proxies)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_rate(self) -> float:
        """Get the sample rate of this proxy from the contained proxies/objects.

         If the contained proxies/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this proxy or the minimum sample rate of the contained proxies/objects.
        """
        sample_rates = list(self.get_sample_rates())
        if self.validate_sample_rate():
            return sample_rates[0]
        else:
            warn(f"The TimeAxisProxy '{self}' does not have a valid sample rate, returning minimum sample rate.")
            return min(sample_rates)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_rate_decimal(self) -> Decimal:
        """Get the sample rate of this proxy from the contained proxies/objects.

         If the contained proxies/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this proxy or the minimum sample rate of the contained proxies/objects.
        """
        sample_rates = list(self.get_sample_rates_decimal())
        if self.validate_sample_rate():
            return sample_rates[0]
        else:
            warn(f"The TimeAxisProxy '{self}' does not have a valid sample rate, returning minimum sample rate.")
            return self.sample_rates[np.nanargmin(np.asarray(self.sample_rates))]

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_periods(self) -> tuple[float]:
        """Get the sample periods of all contained proxies.

        Returns:
            The sample periods of all contained proxies.
        """
        return tuple(proxy.get_sample_period() for proxy in self.proxies)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_periods_decimal(self) -> tuple[Decimal]:
        """Get the sample periods of all contained proxies.

        Returns:
            The sample periods of all contained proxies.
        """
        return tuple(proxy.get_sample_period_decimal() for proxy in self.proxies)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_period(self) -> float:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this proxy.
        """
        sample_periods = list(self.get_sample_periods())
        if self.validate_sample_rate():
            return sample_periods[0]
        else:
            warn(f"The TimeAxisProxy '{self}' does not have a valid sample period, returning maximum sample period.")
            return max(sample_periods)

    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this proxy.
        """
        sample_periods = list(self.get_sample_periods_decimal())
        if self.validate_sample_rate():
            return sample_periods[0]
        else:
            warn(f"The TimeAxisProxy '{self}' does not have a valid sample period, returning maximum sample period.")
            return max(sample_periods)

    def set_precision(self, nano: bool) -> None:
        """Sets if this proxy returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        for proxy in self.proxies:
            proxy.set_precision(nano)

        self.get_data.select("_get_nanostamps" if nano else "_get_timestamps")
        self._precise = nano

    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained proxies.

        Args:
            tzinfo: The time zone to set.
        """
        for proxy in self.proxies:
            proxy.set_tzinfo(tzinfo)

        self.tzinfo = tzinfo

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this proxy has a valid/continuous sample rate.

        Returns:
            If this data proxy has a valid/continuous sample rate.
        """
        sample_rates = np.asarray(self.sample_rates)
        return (sample_rates == sample_rates[0]).all()

    def resample(self, sample_rate: float | None = None, **kwargs: Any) -> None:
        """Resample the proxy and contained proxies.

        Args:
            sample_rate: The new sample rate to change to.
            **kwargs: Any additional kwargs need to change the shape of contained proxies/objects.
        """
        if self.mode == "r":
            raise IOError("not writable")

        if sample_rate is None:
            sample_rate = self.target_sample_rate

        for index, proxy in enumerate(self.proxies):
            if not proxy.validate_sample_rate() or proxy.sample_rate != sample_rate:
                self.proxies[index].resample(sample_rate=sample_rate, **kwargs)

    # Continuous Data
    def where_discontinuous(self, tolerance: float | None = None):
        """Generates a report on where there are sample discontinuities.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            A report on where there are discontinuities.
        """
        # Todo: Make a report object.
        if tolerance is None:
            tolerance = self.time_tolerance

        discontinuities = []
        for index, proxy in enumerate(self.proxies):
            # Check Each proxy
            discontinuities.append(proxy.where_discontinuous(tolerance=tolerance))

            # Check Inbetween proxies
            if index + 1 < len(self.proxies):
                first = proxy.end_timestamp
                second = self.proxies[index + 1].start_timestamp

                if abs((second - first) - self.sample_period) > tolerance:
                    discontinuities.append(index + 1)
                else:
                    discontinuities.append(None)
        return discontinuities

    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this proxy is continuous.
        """
        if tolerance is None:
            tolerance = self.time_tolerance

        for index, proxy in enumerate(self.proxies):
            # Check Each proxy
            if not proxy.validate_continuous():
                return False

            # Check Inbetween proxies
            if index + 1 < len(self.proxies):
                first = proxy.end_timestamp
                second = self.proxies[index + 1].start_timestamp

                if abs((second - first) - self.sample_period) > tolerance:
                    return False

        return True

    def make_continuous(self) -> None:
        """Rearranges the data and interpolates to fill missing data to make this proxy continuous."""
        # Todo: Make actually functional.
        raise NotImplemented
        # if self.mode == 'r':
        #     raise IOError("not writable")
        #
        # fill_proxies = []
        # if self.validate_sample_rate():
        #     sample_rate = self.sample_rate
        #     sample_period = self.sample_period
        # else:
        #     sample_rate = self.target_sample_rate
        #     sample_period = 1 / sample_rate
        #
        # if self.validate_shape():
        #     shape = self.shape
        # else:
        #     shape = self.target_shape
        #
        # for index, proxy in enumerate(self.proxies):
        #     # Make Each proxy Continuous
        #     if not proxy.validate_continuous():
        #         proxy.make_continuous()
        #
        #     # Make Continuous Between proxies
        #     if index + 1 < len(self.proxies):
        #         first = proxy.end_timestamp
        #         second = self.proxies[index + 1].start_timestamp
        #
        #         if (second - first) - sample_period > self.time_tolerance:
        #             start_timestamp = first + sample_period
        #             end_timestamp = second + sample_period
        #             fill_proxies.append(self.fill_type(start_timestamp=start_timestamp, end_timestamp=end_timestamp, sample_rate=sample_rate, shape=shape))
        #
        # if fill_proxies:
        #     self.proxies += fill_proxies
        #     self.sort_proxies()
        #     self.refresh()

    # Time

    # Get Nanostamps
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_nanostamps(self) -> np.ndarray | None:
        """Gets all the nanostamps of this proxy.

        Returns:
            All the nanostamps
        """
        if not self.proxies:
            self.get_nanostamps.clear_cache()
            return None

        # Create nan numpy array
        nanostamps = nan_array(self.get_length(), dtype="u8")

        return self.fill_nanostamps_array(data_array=nanostamps)

    def _get_nanostamps(self) -> np.ndarray | None:
        """An alias method for getting the nanostamps of this proxy.

        Returns:
            The nanostamps of this proxy.
        """
        return self.nanostamps

    def get_nanostamp(self, super_index: int) -> float:
        """Get a nanostamp from within this proxy with a super index.

        Args:
            super_index: The super index to find.

        Returns:
            The requested nanostamp.
        """
        proxy_index, _, inner_index = self.find_inner_proxy_index(super_index=super_index)
        return self.proxies[proxy_index].get_nanostamp(super_index=inner_index)

    def fill_nanostamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice = slice(None),
        slice_: slice = slice(None),
    ) -> np.ndarray:
        """Fills a given array with nanostamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        # Get indices range
        da_shape = data_array.shape
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=slice_.start, stop=slice_.stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        inner_start = range_proxy_indices.start.inner_index
        inner_stop = range_proxy_indices.stop.inner_index
        slice_ = slice(inner_start, inner_stop, slice_.step)

        # Get start_nanostamp and stop array locations
        array_start = 0 if array_slice.start is None else array_slice.start
        array_stop = da_shape[self.axis] if array_slice.stop is None else array_slice.stop

        # Contained proxy/object fill kwargs
        fill_kwargs = {
            "data_array": data_array,
            "array_slice": array_slice,
            "slice_": slice_,
        }

        # Get Data
        if start_proxy == stop_proxy:
            self.proxies[start_proxy].fill_nanostamps_array(**fill_kwargs)
        else:
            # First proxy
            proxy = self.proxies[start_proxy]
            d_size = len(proxy) - inner_start
            a_stop = array_start + d_size
            fill_kwargs["array_slice"] = slice(array_start, a_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(inner_start, None, slice_.step)
            proxy.fill_nanostamps_array(**fill_kwargs)

            # Middle proxies
            fill_kwargs["slice_"] = slice(None, None, slice_.step)
            for proxy in self.proxies[start_proxy + 1: stop_proxy]:
                d_size = len(proxy)
                a_start = a_stop
                a_stop = a_start + d_size
                fill_kwargs["array_slice"] = slice(a_start, a_stop, array_slice.step)
                proxy.fill_nanostamps_array(**fill_kwargs)

            # Last proxy
            a_start = a_stop
            fill_kwargs["array_slice"] = slice(a_start, array_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(None, inner_stop, slice_.step)
            self.proxies[stop_proxy].fill_nanostamps_array(**fill_kwargs)

        return data_array

    def nanostamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool | None = None,
    ) -> Union["BaseTimeProxy", np.ndarray]:
        """Get a slice of nanostamps with indices.

        Args:
            start: The start nanostamp super index.
            stop: The stop super index.
            step: The interval between indices to get nanostamps.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested slice of nanostamps.
        """
        # Create nan numpy array
        start = 0 if start is None else start
        stop = self.length if stop is None else stop
        step = 1 if step is None else step
        length = (stop - start) // step

        nanostamps = nan_array(length, dtype="u8")

        if nanostamps.shape[0] > 0:
            ts = self.fill_nanostamps_array(data_array=nanostamps, slice_=slice(start, stop, step))
        else:
            ts = nanostamps

        if (proxy is None and self.returns_proxy) or proxy:
            return self.time_axis_type(
                ts,
                sample_rate=self.sample_rate_decimal,
                precise=True,
                tzinfo=self.tzinfo,
                mode=self.mode,
            )
        else:
            return ts

    # Nanostamps iterate slice
    def nanostamp_islice_proxy(
        self,
        start_proxy: int,
        stop_proxy: int,
        start_time: datetime.datetime | float | int | np.dtype,
        stop_time: datetime.datetime | float | int | np.dtype,
        step: int | float | datetime.timedelta | Decimal,
        istep: int | Decimal,
    ) -> Generator[BaseTimeProxy, None, None]:
        """Creates a generator which yields nanostamp slices based on proxy indices and times.

        Args:
            start_proxy: The index of the first proxy to start slicing into.
            stop_proxy: The index of the last proxy to stop slicing, inclusive.
            start_time: The time to start slicing into.
            stop_time: The time to stop slicing into, exclusive.
            step: The step of the nanostamp slices to get.
            istep: The step of slices to provide.

        Returns:
            The generator which yields nanostamp slices.
        """
        # Setup Step
        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Iterate through all slices
        current_start = nanostamp(start_time)
        current_stop = current_start + step
        gap = None  # Gap data is the data between proxies
        for proxy_index in range(start_proxy, stop_proxy):
            # Proxy Information
            proxy = self.proxies[proxy_index]
            proxy_start = proxy.start_nanostamp
            proxy_end = proxy.end_nanostamp

            # Yield Gap data if there is break in data between proxies
            if proxy_start > current_stop:
                if gap is None:
                    # On first iteration create gap data
                    gap = self.create_return_proxy_node()
                elif len(gap.proxies) > 0:
                    # Yield Gap data if there is any
                    yield gap.proxies.pop(0) if len(gap.proxies) == 1 else gap
                    gap = self.create_return_proxy_node()
                current_start = proxy_start - (int(proxy_start - current_start) % istep)
                current_stop = current_start + step

            # Handle Gap data between proxies
            if gap is None:
                # On first iteration create gap data
                gap = self.create_return_proxy_node()
            else:
                # Get remainder data from proxy based on start offset into proxy
                p, _, _ = proxy.find_nanostamp_slice(
                    start=np.uint64(current_start),
                    stop=np.uint64(current_stop),
                    tails=True,
                )

                if p is not None and len(p) > 0:
                    gap.proxies.append(p)

                if proxy_end <= current_stop:
                    # Continue to next proxy if there is not enough data in the proxy
                    continue
                else:
                    # Yield completed gap data
                    yield gap.proxies.pop(0) if len(gap.proxies) == 1 else gap
                    gap = self.create_return_proxy_node()
                    current_start = current_start + istep
                    current_stop = current_start + step

            # Iterate over inner proxy
            iter_ = proxy.nanostamp_islice_time(start=np.uint64(current_start), stop=None, step=step, istep=istep)

            try:
                new = next(iter_)
            except StopIteration:
                new = None

            while True:
                last = new
                try:
                    new = next(iter_)
                except StopIteration:
                    break
                else:
                    yield last
                    current_start += istep
                    current_stop = current_start + step

            # Add last slice to the gap data
            gap.proxies.append(last)

        # Last proxy iteration
        proxy = self.proxies[stop_proxy]
        proxy_start = proxy.start_nanostamp
        proxy_end = proxy.end_nanostamp

        # Yield Gap data if there is break in data between proxies
        if proxy_start > current_stop:
            if len(gap.proxies) > 0:
                # Yield Gap data if there is any
                yield gap.proxies.pop(0) if len(gap.proxies) == 1 else gap
                gap = self.create_return_proxy_node()
            current_start = proxy_start + istep - (int(proxy_start - current_start) % istep)
            current_stop = current_start + step

        # Handle Gap data between proxies
        # Get remainder data from proxy based on start offset into proxy
        p, _, _ = proxy.find_nanostamp_slice(
            start=np.uint64(current_start),
            stop=np.uint64(current_stop),
            tails=True,
        )

        if p is not None and len(p) > 0:
            gap.proxies.append(p)

        # Yield completed gap data
        if proxy_end >= current_stop:
            yield gap.proxies.pop(0) if len(gap.proxies) == 1 else gap
            current_start = current_start + istep
            current_stop = current_start + step

        # Iterate over inner proxy
        if proxy_end > current_stop and not (start := np.uint64(current_start)) > (stop := nanostamp(stop_time)):
            iter_ = proxy.nanostamp_islice_time(
                start=start,
                stop=stop,
                step=step,
                istep=istep,
            )

            for item in iter_:
                yield item

    def nanostamp_islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | float | datetime.timedelta | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[BaseTimeProxy, None, None]:
        """Creates a generator which yields nanostamp slices based on indices.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields nanostamp slices.
        """
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=start, stop=stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        start_time = self.proxies[start_proxy].get_nanostamp(range_proxy_indices.start.inner_index)
        stop_time = self.proxies[stop_proxy].get_nanostamp(range_proxy_indices.stop.inner_index)

        # Step
        if step is None:
            return (s for s in (self.nanostamp_slice(start, stop, proxy=True),))
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Get Data
        if start_proxy == stop_proxy:
            return self.proxies[start_proxy].nanostamp_islice_time(start=start_time, stop=stop_time, step=step,
                                                                   istep=istep)
        else:
            return self.nanostamp_islice_proxy(start_proxy=start_proxy, stop_proxy=stop_proxy, start_time=start,
                                               stop_time=stop, step=step, istep=istep)

    def nanostamp_islice_time(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | Decimal | None = None,
        istep: int = 1,
        approx: bool = True,
        tails: bool = False,
    ) -> Generator[BaseTimeProxy, None, None]:
        """Creates a generator which yields nanostamp slices based on times.

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
        stop = nanostamp(stop)
        start_index, stop_index, _ = self.find_time_index_slice(start=start, stop=stop, approx=approx, tails=tails)
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=start_index[0], stop=stop_index[0])

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        adjusted_stop = nanostamp(stop_index.datetime) + np.uint64(1)
        if adjusted_stop < stop:
            stop = adjusted_stop

        # Step
        if step is None:
            return (s for s in (self.nanostamp_slice(start_index[0], stop_index[0], proxy=True),))
        elif isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Get Data
        if start_proxy == stop_proxy:
            return self.proxies[start_proxy].nanostamp_islice_time(start=start, stop=stop, step=step, istep=istep)
        else:
            return self.nanostamp_islice_proxy(start_proxy=start_proxy, stop_proxy=stop_proxy, start_time=start,
                                               stop_time=stop, step=step, istep=istep)

    # Get Timestamps
    @timed_keyless_cache(call_method="clearing_call", local=True)
    def get_timestamps(self) -> np.ndarray | None:
        """Gets all the timestamps of this proxy.

        Returns:
            All the timestamps
        """
        if not self.proxies:
            self.get_timestamps.clear_cache()
            return None

        # Create nan numpy array
        timestamps = nan_array(self.get_length())

        return self.fill_timestamps_array(data_array=timestamps)

    def _get_timestamps(self) -> np.ndarray | None:
        """An alias method for getting the timestamps of this proxy.

        Returns:
            The timestamps of this proxy.
        """
        return self.timestamps

    def get_timestamp(self, super_index: int) -> float:
        """Get a timestamp from within this proxy with a super index.

        Args:
            super_index: The super index to find.

        Returns:
            The requested timestamp.
        """
        proxy_index, _, inner_index = self.find_inner_proxy_index(super_index)
        return self.proxies[proxy_index].get_timestamp(inner_index)

    def fill_timestamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice = slice(None),
        slice_: slice = slice(None),
    ) -> np.ndarray:
        """Fills a given array with timestamps from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        # Get indices range
        da_shape = data_array.shape
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=slice_.start, stop=slice_.stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        inner_start = range_proxy_indices.start.inner_index
        inner_stop = range_proxy_indices.stop.inner_index
        slice_ = slice(inner_start, inner_stop, slice_.step)

        # Get start_timestamp and stop array locations
        array_start = 0 if array_slice.start is None else array_slice.start
        array_stop = da_shape[self.axis] if array_slice.stop is None else array_slice.stop

        # Contained proxy/object fill kwargs
        fill_kwargs = {
            "data_array": data_array,
            "array_slice": array_slice,
            "slice_": slice_,
        }

        # Get Data
        if start_proxy == stop_proxy:
            self.proxies[start_proxy].fill_timestamps_array(**fill_kwargs)
        else:
            # First proxy
            proxy = self.proxies[start_proxy]
            d_size = len(proxy) - inner_start
            a_stop = array_start + d_size
            fill_kwargs["array_slice"] = slice(array_start, a_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(inner_start, None, slice_.step)
            proxy.fill_timestamps_array(**fill_kwargs)

            # Middle proxies
            fill_kwargs["slice_"] = slice(None, None, slice_.step)
            for proxy in self.proxies[start_proxy + 1: stop_proxy]:
                d_size = len(proxy)
                a_start = a_stop
                a_stop = a_start + d_size
                fill_kwargs["array_slice"] = slice(a_start, a_stop, array_slice.step)
                proxy.fill_timestamps_array(**fill_kwargs)

            # Last proxy
            a_start = a_stop
            fill_kwargs["array_slice"] = slice(a_start, array_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(None, inner_stop, slice_.step)
            self.proxies[stop_proxy].fill_timestamps_array(**fill_kwargs)

        return data_array

    def timestamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool | None = None,
    ) -> np.ndarray | BaseTimeProxy:
        """Gets a slice of timestamps along an axis.

        Args:
            start: The first super index of the slice to get.
            stop: The length of the slice to get.
            step: The interval to get the timestamps of the slice.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested slice.
        """
        # Create nan numpy array
        start = 0 if start is None else start
        stop = self.length if stop is None else stop
        step = 1 if step is None else step
        length = (stop - start) // step

        timestamps = nan_array(length)

        ts = self.fill_timestamps_array(data_array=timestamps, slice_=slice(start, stop, step))
        if (proxy is None and self.returns_proxy) or proxy:
            return self.time_axis_type(
                ts,
                sample_rate=self.sample_rate_decimal,
                precise=False,
                tzinfo=self.tzinfo,
                mode=self.mode,
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
        proxy_index, _, inner_index = self.find_inner_proxy_index(super_index=index)
        return self.proxies[proxy_index].get_datetime(inner_index)

    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes of this proxy.

        Returns:
            All the times as a tuple of datetimes.
        """
        datetimes = []
        for proxy in self.proxies:
            datetimes.extend(proxy.get_datetimes())
        return tuple(datetimes)

    # Find Proxy
    def find_proxy(self, timestamp: datetime.datetime | float, tails: bool = False) -> IndexValue:
        """Finds a proxy with a given timestamp in its range

        Args:
            timestamp: The time to find within the proxies.
            tails: Determines if the flanking proxies will be given if the timestamp is out of the range.

        Returns:
            The requested proxy.
        """
        # Setup
        nano_ts = nanostamp(timestamp)

        index = None
        times = self.get_end_nanostamps()

        if nano_ts < self.start_nanostamp:
            if tails:
                index = 0
        elif nano_ts > self.end_nanostamp:
            if tails:
                index = times.shape[0] - 1
        else:
            index = int(np.searchsorted(times, nano_ts, side="left"))

        if index is None:
            raise IndexError("proxy not found. Timestamp out of range")

        proxy = self.proxies[index]

        return IndexValue(index, proxy)

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
        index, proxy = self.find_proxy(timestamp, tails)
        super_index = None
        true_datetime = None

        if index is not None:
            location, true_datetime = proxy.find_time_index(
                timestamp=timestamp,
                approx=approx,
                tails=tails,
            )
            super_index = self.proxy_start_indices[index] + location

        return IndexDateTime(super_index, true_datetime)

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
        if start is None:
            start_index = IndexDateTime(0, self.start_nanostamp)
        else:
            index, proxy = self.find_proxy(start, tails)
            if index is not None:
                location, true_start_datetime = proxy.find_time_index(
                    timestamp=start,
                    approx=approx,
                    tails=tails,
                )
                start_super_index = self.proxy_start_indices[index] + location
                start_index = IndexDateTime(start_super_index, true_start_datetime)
            else:
                start_index = IndexDateTime(None, None)

        if stop is None:
            stop_index = IndexDateTime(self.length, self.end_nanostamp)
        else:
            index, proxy = self.find_proxy(stop, tails=True)
            if index is not None:
                _, inner_stop_index, _, = proxy.find_time_index_slice(
                    stop=stop,
                    approx=approx,
                    tails=tails,
                )
                start_super_index = self.proxy_start_indices[index] + inner_stop_index[0]
                stop_index = IndexDateTime(start_super_index, inner_stop_index[1])
            else:
                stop_index = IndexDateTime(None, None)

        return start_index, stop_index, step

    def where_missing(self) -> tuple[int, ...]:
        """Gets the indices of where gaps in time between proxies."""
        n_proxies = len(self.proxies)
        if n_proxies <= 1:
            return ()
        else:
            starts = np.fromiter(self.start_nanostamps, dtype=np.uint64)
            max_starts = np.fromiter(
                iter=(p.end_nanostamp + (p.sample_period + self.time_tolerance) * 10**9 for p in self.proxies),
                dtype=np.uint64,
            )
            return tuple(np.where(starts[1:] > max_starts[:-1])[0] + 1)

    def insert_missing(self, type_: type[BaseTimeProxy] | None = None, recursive: bool = False, **kwargs: Any) -> None:
        """Inserts blanks proxies in missing proxy segments.

        Args:
            type_: The type of proxy to insert into the missing regions of the timeseries.
            recursive: Determines if blank proxies will be inserted into the contained proxies.
            **kwargs: The keyword arguments for creating the blank proxies.
        """
        if recursive:
            for proxy in self.proxies:
                proxy.insert_missing(type_=type_, recursive=recursive, **kwargs)

        f_type = self.fill_type if type_ is None else None
        for i in reversed(self.where_missing()):
            previous_proxy = self.proxies[i - 1]
            previous_period = previous_proxy.sample_period
            next_proxy = self.proxies[i]
            shape = list(previous_proxy.shape)
            shape[previous_proxy.axis] = 0
            b_kwargs = {
                "start": previous_proxy.end_timestamp + previous_period,
                "end": next_proxy.start_timestamp - previous_period,
                "sample_rate": previous_proxy.sample_rate,
                "shape": shape,
                "axis": previous_proxy.axis,
                "precise": previous_proxy.precise,
                "tzinfo": previous_proxy.tzinfo,
            }
            self.proxies.insert(i, f_type(**(b_kwargs | kwargs)))
        self.clear_caches()


# Assign Cyclic Definitions
TimeProxy.default_return_proxy_type = TimeProxy
