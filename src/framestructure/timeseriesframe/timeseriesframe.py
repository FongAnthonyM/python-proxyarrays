""" timeseriesframe.py

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
import datetime  # Todo: Consider Pandas Timestamp for nanosecond resolution
from typing import Any
from warnings import warn

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache
from dspobjects.dataclasses import IndexValue, IndexDateTime
import numpy as np

# Local Packages #
from ..arrayframe import ArrayFrame
from .timeseriesframeinterface import TimeSeriesFrameInterface
from .blanktimeframe import BlankTimeFrame


# Definitions #
# Classes #
class TimeSeriesFrame(ArrayFrame, TimeSeriesFrameInterface):
    """An ArrayFrome that has been expanded to handle time series data.

    Class Attributes:
        default_fill_type: The default type to fill discontinuous data.

    Attributes:
        _date: An assigned date if this object does not have a start_timestamp timestamp yet.
        target_sample_rate: The sample rate that this frame should be.
        time_tolerance: The allowed deviation a sample can be away from the sample period.
        fill_type: The type that will fill discontinuous data.

    Args:
        frames: An iterable holding frames/objects to store in this frame.
        mode: Determines if the contents of this frame are editable or not.
        update: Determines if this frame will start_timestamp updating or not.
        init: Determines if this object will construct.
    """
    default_fill_type = BlankTimeFrame

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        frames: Iterable[TimeSeriesFrameInterface] | None = None,
        mode: str = 'a',
        update: bool = False,
        init: bool = True
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        self._date: datetime.date | None = None

        self.target_sample_rate: float | None = None

        self.time_tolerance: float = 0.000001
        self.fill_type: type = self.default_fill_type

        # Object Construction #
        if init:
            self.construct(frames=frames, mode=mode, update=update)

    @property
    def start_timestamp(self) -> float | None:
        """The start_timestamp timestamp of this frame."""
        try:
            return self.get_start_timestamp.caching_call()
        except AttributeError:
            return self.get_start_timestamp()

    @property
    def end_timestamp(self) -> float | None:
        """The end_timestamp timestamp of this frame."""
        try:
            return self.get_end_timestamp.caching_call()
        except AttributeError:
            return self.get_end_timestamp()
        
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
        """The start_timestamp date of the data in this frame."""
        start = self.start_datetime
        if start is None:
            return self._date
        else:
            return start.date()

    @property
    def sample_rates(self) -> tuple[float]:
        """The sample rates of the contained sample rates."""
        try:
            return self.get_sample_rates.caching_call()
        except AttributeError:
            return self.get_sample_rates()

    @property
    def sample_rate(self) -> float:
        """The sample rate of this frame."""
        try:
            return self.get_sample_rate.caching_call()
        except AttributeError:
            return self.get_sample_rate()

    @property
    def sample_period(self) -> float:
        """The period between samples."""
        try:
            return self.get_sample_period.caching_call()
        except AttributeError:
            return self.get_sample_period()

    # Instance Methods
    # Shorting
    def frame_sort_key(self, frame: Any) -> Any:
        """The key to be used in sorting with the frame as the sort basis.

        Args:
            frame: The frame to sort.
        """
        return frame.start_timestamp

    # Cache and Memory
    def refresh(self) -> None:
        """Resets this frame's caches and fills them with updated values."""
        super().refresh()
        self.get_start_timestamp()
        self.get_end_timestamp()
        self.get_sample_rates()
        self.get_sample_rate()
        self.get_sample_period()
        self.get_is_continuous()

    # Getters
    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_start_timestamp(self) -> float | None:
        """Gets the start_timestamp timestamp of this frame.
        
        Returns:
            The start_timestamp timestamp of this frame.
        """
        if self.frames:
            return self.frames[0].start_timestamp
        else:
            self.get_start_timestamp.clear_cache()
            return None

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_end_timestamp(self) -> float | None:
        """Gets the end_timestamp timestamp of this frame.

        Returns:
            The end_timestamp timestamp of this frame.
        """
        if self.frames:
            return self.frames[-1].end_timestamp
        else:
            self.get_start_timestamp.clear_cache()
            return None

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_start_timestamps(self) -> np.ndarray:
        """Get the start_timestamp timestamps of all contained frames.
        
        Returns:
            All the start_timestamp timestamps.
        """
        starts = np.empty(len(self.frames))
        starts.fill(np.nan)
        for index, frame in enumerate(self.frames):
            start = frame.get_start_timestamp()
            if start is not None:
                starts[index] = start
        return starts

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_end_timestamps(self) -> np.ndarray:
        """Get the end_timestamp timestamps of all contained frames.

        Returns:
            All the end_timestamp timestamps.
        """
        ends = np.empty(len(self.frames))
        ends.fill(np.nan)
        for index, frame in enumerate(self.frames):
            end = frame.get_end_timestamp()
            if end is not None:
                ends[index] = end
        return ends

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_start_datetimes(self) -> tuple[datetime.datetime | None]:
        """Get the start_timestamp datetimes of all contained frames.

        Returns:
            All the start_timestamp datetimes.
        """
        starts = [] * len(self.frames)
        for index, frame in enumerate(self.frames):
            starts[index] = frame.start_datetime
        return tuple(starts)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_end_datetimes(self) -> tuple[datetime.datetime | None]:
        """Get the end_timestamp datetimes of all contained frames.

        Returns:
            All the end_timestamp datetimes.
        """
        ends = [] * len(self.frames)
        for index, frame in enumerate(self.frames):
            ends[index] = frame.end_datetime
        return tuple(ends)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_sample_rates(self) -> tuple[float]:
        """Get the sample rates of all contained frames.

        Returns:
            The sample rates of all contained frames.
        """
        return tuple(frame.get_sample_rate() for frame in self.frames)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_sample_rate(self) -> float:
        """Get the sample rate of this frame from the contained frames/objects.

         If the contained frames/object are different this will raise a warning and return the minimum sample rate.

        Returns:
            The shape of this frame or the minimum sample rate of the contained frames/objects.
        """
        sample_rates = list(self.get_sample_rates())
        if self.validate_sample_rate():
            return sample_rates[0]
        else:
            warn(f"The TimeSeriesFrame '{self}' does not have a valid shape, returning minimum shape.")
            return min(sample_rates)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", collective=False)
    def get_sample_period(self) -> float:
        """Get the sample period of this frame.

        If the contained frames/object are different this will raise a warning and return the maximum period.

        Returns:
            The sample period of this frame.
        """
        sample_rate = self.get_sample_rate()
        if not isinstance(sample_rate, bool):
            return 1 / sample_rate
        else:
            return sample_rate

    # Sample Rate
    def validate_sample_rate(self) -> bool:
        """Checks if this frame has a valid/continuous sample rate.

        Returns:
            If this data frame has a valid/continuous sample rate.
        """
        sample_rates = np.asarray(self.sample_rates)
        return (sample_rates == sample_rates[0]).all()

    def resample(self, sample_rate: float | None = None, **kwargs: Any) -> None:
        """Resample the frame and contained frames.

        Args:
            sample_rate: The new sample rate to change to.
            **kwargs: Any additional kwargs need to change the shape of contained frames/objects.
        """
        if self.mode == 'r':
            raise IOError("not writable")

        if sample_rate is None:
            sample_rate = self.target_sample_rate

        for index, frame in enumerate(self.frames):
            if not frame.validate_sample_rate() or frame.sample_rate != sample_rate:
                self.frames[index].resample(sample_rate=sample_rate, **kwargs)

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
        for index, frame in enumerate(self.frames):
            # Check Each Frame
            discontinuities.append(frame.where_discontinuous())

            # Check Inbetween Frames
            if index + 1 < len(self.frames):
                first = frame.end_timestamp
                second = self.frames[index + 1].start_timestamp

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
            If this frame is continuous.
        """
        if tolerance is None:
            tolerance = self.time_tolerance

        for index, frame in enumerate(self.frames):
            # Check Each Frame
            if not frame.validate_continuous():
                return False

            # Check Inbetween Frames
            if index + 1 < len(self.frames):
                first = frame.end_timestamp
                second = self.frames[index + 1].start_timestamp

                if abs((second - first) - self.sample_period) > tolerance:
                    return False

        return True

    def make_continuous(self) -> None:
        """Rearranges the data and interpolates to fill missing data to make this frame continuous."""
        # Todo: Make actually functional.
        raise NotImplemented
        # if self.mode == 'r':
        #     raise IOError("not writable")
        #
        # fill_frames = []
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
        # for index, frame in enumerate(self.frames):
        #     # Make Each Frame Continuous
        #     if not frame.validate_continuous():
        #         frame.make_continuous()
        #
        #     # Make Continuous Between Frames
        #     if index + 1 < len(self.frames):
        #         first = frame.end_timestamp
        #         second = self.frames[index + 1].start_timestamp
        #
        #         if (second - first) - sample_period > self.time_tolerance:
        #             start_timestamp = first + sample_period
        #             end_timestamp = second + sample_period
        #             fill_frames.append(self.fill_type(start_timestamp=start_timestamp, end_timestamp=end_timestamp, sample_rate=sample_rate, shape=shape))
        #
        # if fill_frames:
        #     self.frames += fill_frames
        #     self.sort_frames()
        #     self.refresh()

    # Get Timestamps
    @timed_keyless_cache(call_method="clearing_call", collective=False)
    def get_timestamps(self) -> np.ndarray | None:
        """Gets all the timestamps of this frame.

        Returns:
            All the timestamps
        """
        if not self.frames:
            self.get_timestamps.clear_cache()
            return None

        # Create nan numpy array
        timestamps = np.empty(self.get_length())
        timestamps.fill(np.nan)

        return self.fill_timestamps_array(data_array=timestamps)

    def get_timestamp(self, super_index: int) -> float:
        """Get a timestamp from within this frame with a super index.

        Args:
            super_index: The super index to find.

        Returns:
            The requested timestamp.
        """
        frame_index, _, inner_index = self.find_inner_frame_index(super_index=super_index)
        return self.frames[frame_index].get_timestamp_from_index(super_index=inner_index)

    def get_timestamp_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Gets a range of timestamps along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the timestamps of the range.

        Returns:
            The requested range.
        """
        # Create nan numpy array
        start = 0 if start is None else start
        stop = self.length if stop is None else stop
        step = 1 if step is None else step
        length = (stop - start) // step

        timestamps = np.empty(length)
        # timestamps.fill(np.nan)

        return self.fill_timestamps_array(data_array=timestamps, slice_=slice(start, stop, step))
    
    def fill_timestamps_array(
        self,
        data_array: np.ndarray,
        array_slice: slice = slice(None),
        slice_: slice = slice(None),
    ) -> np.ndarray:
        """Fills a given array with timestamps from the contained frames/objects.

        Args:
            data_array: The numpy array to fill.
            array_slice: The slices to fill within the data_array.
            slice_: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        # Get indices range
        da_shape = data_array.shape
        range_frame_indices = self.parse_range_super_indices(start=slice_.start, stop=slice_.stop)

        start_frame = range_frame_indices.start.index
        stop_frame = range_frame_indices.stop.index
        inner_start = range_frame_indices.start.inner_index
        inner_stop = range_frame_indices.stop.inner_index
        slice_ = slice(inner_start, inner_stop, slice_.step)
        
        # Get start_timestamp and stop array locations
        array_start = 0 if array_slice.start is None else array_slice.start
        array_stop = da_shape[self.axis] if array_slice.stop is None else array_slice.stop

        # Contained frame/object fill kwargs
        fill_kwargs = {
            "data_array": data_array,
            "array_slice": array_slice,
            "slice_": slice_
        }

        # Get Data
        if start_frame == stop_frame:
            self.frames[start_frame].fill_timestamps_array(**fill_kwargs)
        else:
            # First Frame
            frame = self.frames[start_frame]
            d_size = len(frame) - inner_start
            a_stop = array_start + d_size
            fill_kwargs["array_slice"] = slice(array_start, a_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(inner_start, None, slice_.step)
            frame.fill_timestamps_array(**fill_kwargs)

            # Middle Frames
            fill_kwargs["slice_"] = slice(None, None, slice_.step)
            for frame in self.frames[start_frame + 1:stop_frame]:
                d_size = len(frame)
                a_start = a_stop
                a_stop = a_start + d_size
                fill_kwargs["array_slice"] = slice(a_start, a_stop, array_slice.step)
                frame.fill_timestamps_array(**fill_kwargs)

            # Last Frame
            a_start = a_stop
            fill_kwargs["array_slice"] = slice(a_start, array_stop, array_slice.step)
            fill_kwargs["slice_"] = slice(None, inner_stop, slice_.step)
            self.frames[stop_frame].fill_timestamps_array(**fill_kwargs)

        return data_array

    def get_datetimes(self) -> tuple[datetime.datetime]:
        """Gets all the datetimes of this frame.

        Returns:
            All the times as a tuple of datetimes.
        """
        datetimes = []
        for frame in self.frames:
            datetimes += list(frame.get_datetimes())
        return tuple(datetimes)
    
    # Find Time
    def find_frame(self, timestamp: datetime.datetime | float, tails: bool = False) -> IndexValue:
        """Finds a frame with a given timestamp in its range
        
        Args:
            timestamp: The time to find within the frames.
            tails: Determines if the flanking frames will be given if the timestamp is out of the range.

        Returns:
            The requested frame.
        """
        # Setup
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()

        index = None
        times = self.get_start_timestamps()

        if timestamp < self.start_timestamp:
            if tails:
                index = 0
        elif timestamp > self.end_timestamp:
            if tails:
                index = times.shape[0] - 1
        else:
            index = np.searchsorted(times, timestamp, side="right") - 1

        if index is None:
            raise IndexError("Frame not found. Timestamp out of range")

        frame = self.frames[index]

        return IndexValue(index, frame)

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
        if isinstance(timestamp, float):
            timestamp = datetime.datetime.fromtimestamp(timestamp)
        index, frame = self.find_frame(timestamp, tails)
        super_index = None
        true_datetime = None
        true_timestamp = None

        if index is not None:
            location, true_datetime, true_timestamp = frame.find_time_index(
                timestamp=timestamp,
                approx=approx,
                tails=tails,
            )
            super_index = self.frame_start_indices[index] + location
                
        return IndexDateTime(super_index, true_datetime, true_timestamp)


# Assign Cyclic Definitions
TimeSeriesFrame.default_return_frame_type = TimeSeriesFrame
