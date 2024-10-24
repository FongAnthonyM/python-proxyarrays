"""timeseriesproxy.py
A TimeProxy that has been expanded to handle time series data.
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
from collections.abc import Generator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

# Third-Party Packages #
from dspobjects.time import nanostamp
import numpy as np

# Local Packages #
from ..timeproxy import TimeProxy
from ..timeaxis import ContainerTimeAxis
from .basetimeseries import BaseTimeSeries
from .blanktimeseries import BlankTimeSeries
from .containertimeseries import ContainerTimeSeries


# Definitions #
# Classes #
class TimeSeriesProxy(TimeProxy, BaseTimeSeries):
    """A TimeProxy that has been expanded to handle time series data."""

    default_fill_type = BlankTimeSeries
    time_axis_type = ContainerTimeAxis
    time_series_type = ContainerTimeSeries

    # Instance Methods #
    # Constructors/Destructors
    def proxy_leaf_copy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a copy proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The copy of this proxy array.
        """
        proxy_copy = super().proxy_leaf_copy(type_=type_, **kwargs)
        proxy_copy.time_axis = self.time_axis_type(
            data=self.get_nanostamps(),
            samperate=self.sample_rate,
            axis=self.axis,
            precision=True,
            mode=self.mode,
        )
        return proxy_copy

    def dataless_proxy_leaf_copy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a dataless proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The copy of this proxy array.
        """
        empty_copy = self.create_proxy(type_=self.return_proxy_leaf if type_ is None else type_, **kwargs)
        empty_copy.time_axis = self.time_axis_type(
            data=self.get_nanostamps(),
            samperate=self.sample_rate,
            axis=self.axis,
            precision=True,
            mode=self.mode,
        )
        return empty_copy

    # Data iterate slice
    def find_data_islice_proxy(
        self,
        start_proxy: int,
        stop_proxy: int,
        start_time: datetime | float | int | np.dtype,
        stop_time: datetime | float | int | np.dtype,
        step: int | float | timedelta | Decimal,
        istep: int | Decimal,
    ) -> Generator[BaseTimeSeries, None, None]:
        """Creates a generator which yields data slices based on proxy indices and times.

        Args:
            start_proxy: The index of the first proxy to start slicing into.
            stop_proxy: The index of the last proxy to stop slicing, inclusive.
            start_time: The time to start slicing into.
            stop_time: The time to stop slicing into, exclusive.
            step: The step of the nanostamp slices to get.
            istep: The step of slices to provide.

        Returns:
            The generator which yields data slices.
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
                p, _, _, _, _, _ = proxy.find_data_slice(
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
            iter_ = proxy.find_data_islice_time(
                start=np.uint64(current_start),
                stop=None,
                step=step,
                istep=istep,
            )

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
        p, _, _, _, _, _ = proxy.find_data_slice(
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
            iter_ = proxy.find_data_islice_time(
                start=start,
                stop=stop,
                step=step,
                istep=istep,
            )

            for item in iter_:
                yield item

    def find_data_islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | float | timedelta | None = None,
        istep: int | Decimal = 1,
    ) -> Generator[BaseTimeSeries, None, None]:
        """Creates a generator which yields data slices based on indices.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.

        Returns:
            The generator which yields data slices.
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
            return self.proxies[start_proxy].find_data_islice_time(
                start=start_time,
                stop=stop_time,
                step=step,
                istep=istep,
            )
        else:
            return self.find_data_islice_proxy(
                start_proxy=start_proxy,
                stop_proxy=stop_proxy,
                start_time=start,
                stop_time=stop,
                step=step,
                istep=istep,
            )

    def find_data_islice_time(
        self,
        start: datetime | float | int | np.dtype | None = None,
        stop: datetime | float | int | np.dtype | None = None,
        step: int | float | timedelta | Decimal | None = None,
        istep: int = 1,
        approx: bool = True,
        tails: bool = False,
    ) -> Generator[BaseTimeSeries, None, None]:
        """Creates a generator which yields data slices based on times.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The generator which yields data slices.
        """
        if stop is not None:
            stop = nanostamp(stop)
        start_index, stop_index, _ = self.find_time_index_slice(start=start, stop=stop, approx=approx, tails=True)
        if start_index[0] == self.length or stop_index[0] == 0:
            return (self.create_return_proxy_node() for i in (1,))

        range_proxy_indices = self.find_inner_proxy_indices_slice(start=start_index[0], stop=stop_index[0])

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        adjusted_stop = nanostamp(stop_index.datetime) + np.uint64(1)
        if stop is None or adjusted_stop < stop:
            stop = adjusted_stop

        # Step
        if step is None:
            return (s for s in (self.nanostamp_slice(start_index[0], stop_index[0], proxy=True),))
        elif isinstance(step, timedelta):
            step = step.total_seconds()

        if not isinstance(step, Decimal):
            step = Decimal(step) * 10 ** 9
        if not isinstance(istep, Decimal):
            istep = step * istep

        # Get Data
        if start_proxy == stop_proxy:
            return self.proxies[start_proxy].find_data_islice_time(
                start=start,
                stop=stop,
                step=step,
                istep=istep,
            )
        else:
            return self.find_data_islice_proxy(
                start_proxy=start_proxy,
                stop_proxy=stop_proxy,
                start_time=start,
                stop_time=stop,
                step=step,
                istep=istep,
            )


# Assign Cyclic Definitions
TimeSeriesProxy.default_return_proxy_node = TimeSeriesProxy
TimeSeriesProxy.default_return_proxy_leaf = ContainerTimeSeries
TimeSeriesProxy.default_return_proxy_type = TimeSeriesProxy
