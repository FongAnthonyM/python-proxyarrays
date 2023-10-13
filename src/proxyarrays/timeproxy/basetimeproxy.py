"""basetimeproxy.py
An interface which outlines the basis for a time proxy.
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
from abc import abstractmethod
from collections.abc import Iterable, Iterator
import datetime
from decimal import Decimal
from typing import Any, NamedTuple, Union

# Third-Party Packages #
from dspobjects.dataclasses import IndexDateTime
from dspobjects.time import Timestamp
import numpy as np

# Local Packages #
from ..proxyarray.baseproxyarray import Slice
from ..proxyarray import BaseProxyArray


# Definitions #
# Classes #
class BaseTimeProxy(BaseProxyArray):
    """An interface which outlines the basis for a time proxy."""

    # Magic Methods #
    # Construction/Destruction
    @property
    def is_continuous(self) -> bool:
        """If the data in the proxies is continuous."""
        return self.validate_continuous()

    # Numpy ndarray Methods
    @abstractmethod
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        pass

    # Instance Methods #
    # Getters
    @abstractmethod
    def get_shape(self) -> tuple[int]:
        """Get the shape of this proxy from the contained proxies/objects.

        Returns:
            The shape of this proxy.
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """Gets the length of this proxy.

        Returns:
            The length of this proxy.
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get the sample rate of this proxy from the contained proxies/objects.

         If the contained proxies/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this proxy or the minimum sample rate of the contained proxies/objects.
        """
        pass

    @abstractmethod
    def get_sample_rate_decimal(self) -> Decimal:
        """Get the sample rate of this proxy from the contained proxies/objects.

         If the contained proxies/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this proxy or the minimum sample rate of the contained proxies/objects.
        """
        pass

    @abstractmethod
    def get_sample_period(self) -> float:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this proxy.
        """
        pass

    @abstractmethod
    def get_sample_period_decimal(self) -> Decimal:
        """Get the sample period of this proxy.

        If the contained proxies/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this proxy.
        """
        pass

    @abstractmethod
    def set_precision(self, nano: bool) -> None:
        """Sets if this proxy returns nanostamps (True) or timestamps (False).

        Args:
            nano: Determines if this proxy returns nanostamps (True) or timestamps (False).
        """
        pass

    @abstractmethod
    def set_tzinfo(self, tzinfo: datetime.tzinfo | None = None) -> None:
        """Sets the time zone of the contained proxies.

        Args:
            tzinfo: The time zone to set.
        """
        pass

    @abstractmethod
    def get_item(self, item: Any) -> Any:
        """Gets an item from within this proxy based on an input item.

        Args:
            item: The object to be used to get a specific item within this proxy.

        Returns:
            An item within this proxy.
        """
        pass

    # Shape
    @abstractmethod
    def validate_shape(self) -> bool:
        """Checks if this proxy has a valid/continuous shape.

        Returns:
            If this proxy has a valid/continuous shape.
        """
        pass

    @abstractmethod
    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the proxy without changing its data."""
        pass

    # Sample Rate
    @abstractmethod
    def validate_sample_rate(self) -> bool:
        """Checks if this proxy has a valid/continuous sampling rate.

        Returns:
            If this proxy has a valid/continuous sampling rate.
        """
        pass

    @abstractmethod
    def resample(self, sample_rate: float, **kwargs: Any) -> None:
        """Resamples the data to match the given sample rate.

        Args:
            sample_rate: The new sample rate for the data.
            **kwargs: Keyword arguments for the resampling.
        """
        pass

    # Continuous Data
    @abstractmethod
    def validate_continuous(self, tolerance: float | None = None) -> bool:
        """Checks if the time between each sample matches the sample period.

        Args:
            tolerance: The allowed deviation a sample can be away from the sample period.

        Returns:
            If this proxy is continuous.
        """
        pass

    @abstractmethod
    def make_continuous(self) -> None:
        """Adjusts the data to make it continuous."""
        pass

    # Proxy
    @abstractmethod
    def get_from_index(
        self,
        indices: Iterator | Iterable | int,
        reverse: bool = False,
        proxy: bool = True,
    ) -> Any:
        """Get an item recursively from within this proxy using indices.

        Args:
            indices: The indices used to get an item within this proxy.
            reverse: Determines if the indices should be used in the reverse order.
            proxy: Determines if the

        Returns:
            The item recursively from within this proxy.
        """
        pass

    # Get Nanostamps
    @abstractmethod
    def get_nanostamps(self) -> np.ndarray:
        """Gets all the nanostamps of this proxy.

        Returns:
            A numpy array of the nanostamps of this proxy.
        """
        pass

    @abstractmethod
    def get_nanostamp(self, super_index: int) -> float:
        """Get a time from a contained proxy with a super index.

        Args:
            super_index: The index to get the nanostamp.

        Returns:
            The nanostamp
        """
        pass  # return self.time[super_index]

    @abstractmethod
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
        pass

    @abstractmethod
    def nanostamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = True,
    ) -> Union["BaseTimeProxy", np.ndarray]:
        """Get a slice of nanostamps with indices.

        Args:
            start: The start super index.
            stop: The stop super index.
            step: The interval between indices to get nanostamps.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range of nanostamps.
        """
        pass  # return self.times[slice(start_nanostamp, stop, step)]

    # Get Timestamps
    @abstractmethod
    def get_timestamps(self) -> np.ndarray:
        """Gets all the timestamps of this proxy.

        Returns:
            A numpy array of the timestamps of this proxy.
        """
        pass

    @abstractmethod
    def get_timestamp(self, super_index: int) -> float:
        """Get a time from a contained proxy with a super index.

        Args:
            super_index: The index to get the timestamp.

        Returns:
            The timestamp
        """
        pass  # return self.time[super_index]

    @abstractmethod
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
        pass

    @abstractmethod
    def timestamp_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        proxy: bool = True,
    ) -> Union["BaseTimeProxy", np.ndarray]:
        """Get a slice of timestamps with indices.

        Args:
            start: The start_timestamp super index.
            stop: The stop super index.
            step: The interval between indices to get timestamps.
            proxy: Determines if the returned object will be a proxy.

        Returns:
            The requested range of timestamps.
        """
        pass  # return self.times[slice(start_timestamp, stop, step)]

    # Datetimes [Timestamp]
    @abstractmethod
    def get_datetime(self, index: int) -> Timestamp:
        """A datetime from this proxy based on the index.

        Args:
            index: The index of the datetime to get.

        Returns:
            All the times as a tuple of datetimes.
        """
        pass

    @abstractmethod
    def get_datetimes(self) -> tuple[Timestamp]:
        """Gets all the datetimes of this proxy.

        Returns:
            All the times as a tuple of datetimes.
        """
        pass

    def datetime_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "BaseTimeProxy":
        """Get a slice of datetimes with indices.

        Args:
            start: The start index.
            stop: The stop index.
            step: The interval between indices to get datetimes.

        Returns:
            The requested range of datetimes.
        """
        return self.nanostamp_slice(start, stop, step, proxy=True)

    # Get Data
    @abstractmethod
    def fill_slices_array(
        self,
        data_array: np.ndarray,
        array_slices: Iterable[slice] | None = None,
        slices: Iterable[slice | int | None] | None = None,
    ) -> np.ndarray:
        """Fills a given array with values from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slices: The slices to fill within the data_array.
            slices: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        pass

    @abstractmethod
    def slices_array(self, slices: Iterable[slice | int | None] | None = None, dtype: Any = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.
            dtype: The dtype of array to return.

        Returns:
            The requested range as an array.
        """
        pass

    @abstractmethod
    def slices_proxy(self, slices: Iterable[Slice] | None = None) -> "BaseProxyArray":
        """Get data as a new proxy using slices to determine the data slice.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested range as a proxy.
        """
        pass

    @abstractmethod
    def islices(
        self,
        slices: Iterable[slice | int | None] | None = None,
        islice: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Union["BaseProxyArray", np.ndarray]:
        """Creates a generator which iterates over slices along an axis.

        Args:
            slices: The ranges of the data to get.
            islice: The range to data to iterate over.
            axis: The axis to iterate along.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Yields:
            The requested slices.
        """
        pass

    # Find Time Index
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def find_nanostamp_slice(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> "FoundTimeRange":
        """Finds the nanostamp range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The nanostamp range on the axis and the start_nanostamp and stop indices.
        """
        if isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        start_index, stop_index, _ = self.find_time_index_slice(start=start, stop=stop, approx=approx, tails=tails)

        return FoundTimeRange(
            self.nanostamp_slice(start_index[0], stop_index[0], step, proxy=True),
            start_index[0],
            stop_index[0],
        )

    def find_timestamp_slice(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> "FoundTimeRange":
        """Finds the timestamp range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The timestamp range on the axis and the start_timestamp and stop indices.
        """
        if isinstance(step, datetime.timedelta):
            step = step.total_seconds()

        if start is None:
            start_index = 0
        else:
            start_index, _ = self.find_time_index(timestamp=start, approx=approx, tails=tails)

        if stop is None:
            stop_index = self.get_length()
        else:
            stop_index, _ = self.find_time_index(timestamp=stop, approx=approx, tails=tails)

        return FoundTimeRange(
            self.timestamp_slice(start_index, stop_index, step, proxy=True),
            start_index,
            stop_index,
        )

    def find_datetime_slice(
        self,
        start: datetime.datetime | float | int | np.dtype | None = None,
        stop: datetime.datetime | float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> "FoundTimeRange":
        """Finds the datetime range on the axis inbetween two times, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The datetime range on the axis and the start and stop indices.
        """
        return self.find_nanostamp_slice(start, stop, step, approx, tails)

    def find_time_nanoseconds(
        self,
        start: float | int | np.dtype | None = None,
        stop: float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> "FoundTimeRange":
        """Finds the datetime range on the axis inbetween two second offsets, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The datetime range on the axis and the start and stop indices.
        """
        if start is not None:
            if start >= 0:
                start = self.start_nanostamp + np.int64(start)
            else:
                start = self.end_nanostamp + np.int64(start)

        if stop is not None:
            if stop >= 0:
                start = self.start_nanostamp + np.int64(stop)
            else:
                start = self.end_nanostamp + np.int64(stop)

        return self.find_nanostamp_slice(self.start_nanotimestamp + start, self.end_nanotimestamp + stop, step, approx,
                                         tails)

    def find_time_seconds(
        self,
        start: float | int | np.dtype | None = None,
        stop: float | int | np.dtype | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = True,
        tails: bool = False,
    ) -> "FoundTimeRange":
        """Finds the datetime range on the axis inbetween two second offsets, can give approximate values.

        Args:
            start: The first time to find for the range.
            stop: The last time to find for the range.
            step: The step between elements in the range.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The datetime range on the axis and the start and stop indices.
        """
        if start is not None:
            if start >= 0:
                start = self.start_timestamp + start
            else:
                start = self.end_timestamp + start

        if stop is not None:
            if stop >= 0:
                start = self.start_timestamp + stop
            else:
                start = self.end_timestamp + stop

        return self.find_timestamp_slice(self.start_timestamp + start, self.end_timestamp + stop, step, approx, tails)


class FoundTimeRange(NamedTuple):
    """A name tuple for returning a range of times with its start and end."""

    data: BaseTimeProxy | None
    start: int | None
    end: int | None
