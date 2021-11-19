#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dataframe.py
Description:
"""
# Package Header #
from .__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from contextlib import contextmanager
from bisect import bisect_right
import math
from warnings import warn

# Third-Party Packages #
from baseobjects.cachingtools import timed_keyless_cache_method
import numpy as np

# Local Packages #
from .dataframeinterface import DataFrameInterface
from .datacontainer import DataContainer


# Definitions #
# Classes #
class DataFrame(DataFrameInterface):
    default_return_frame_type = None
    default_combine_type = DataContainer

    # Magic Methods
    # Construction/Destruction
    def __init__(self, frames=None, mode='a', update=True, init=True):
        # Parent Attributes #
        super().__init__(init=False)

        # New Attributes #
        # Descriptors #
        # System
        self._cache = False
        self.is_updating = True
        self.is_combine = False
        self.returns_frame = False
        self.mode = 'a'

        # Shape
        self.target_shape = None
        self.axis = 0

        # Assign Classes #
        self.combine_type = self.default_combine_type
        self.return_frame_type = self.default_return_frame_type

        # Containers #
        self.frames = []

        # Object Construction #
        if init:
            self.construct(frames=frames, mode=mode, update=update)

    @property
    def shapes(self):
        try:
            return self.get_shapes.caching_call()
        except AttributeError:
            return self.get_shapes()

    @property
    def shape(self):
        try:
            return self.get_shape.caching_call()
        except AttributeError:
            return self.get_shape

    @property
    def lengths(self):
        try:
            return self.get_lengths.caching_call()
        except AttributeError:
            return self.get_lengths()

    @property
    def length(self):
        try:
            return self.get_length.caching_call()
        except AttributeError:
            return self.get_length()

    @property
    def frame_start_indices(self):
        try:
            return self.get_frame_start_indices.caching_call()
        except AttributeError:
            return self.get_frame_start_indices()

    @property
    def frame_end_indices(self):
        try:
            return self.get_frame_end_indices.caching_call()
        except AttributeError:
            return self.get_frame_end_indices()

    # Arithmetic
    def __add__(self, other):
        if isinstance(other, DataFrame):
            return type(self)(self.frames + other.frames, self.is_updating)
        else:
            return type(self)(self.frames + other, self.is_updating)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, frames=None, mode=None, update=None):
        if frames is not None:
            self.frames = frames

        if mode is not None:
            self.mode = mode

        if update is not None:
            self.is_updating = update

    # Editable Copy Methods
    def default_editable_method(self):
        return self.combine_frames()

    # Cache and Memory
    def refresh(self):
        self.get_shapes()
        self.get_shape()
        self.get_lengths()
        self.get_length()

    def clear_all_caches(self):
        self.clear_caches()
        for frame in self.frames:
            frame.clear_all_caches()

    # Getters
    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_shapes(self):
        self.get_lengths.clear_cache()
        return tuple(frame.shape for frame in self.frames)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_min_shape(self):
        n_frames = len(self.frames)
        n_dims = [None] * n_frames
        shapes = [None] * n_frames
        for index, frame in enumerate(self.frames):
            shapes[index] = shape = frame.shape
            n_dims[index] = len(shape)

        max_dims = max(n_dims)
        shape_array = np.zeros((n_frames, max_dims), dtype='i')
        for index, s in enumerate(shapes):
            shape_array[index, :n_dims[index]] = s

        shape = [None] * max_dims
        for ax in range(max_dims):
            if ax == self.axis:
                shape[ax] = sum(shape_array[:, ax])
            else:
                shape[ax] = min(shape_array[:, ax])
        return tuple(shape)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_max_shape(self):
        n_frames = len(self.frames)
        n_dims = [None] * n_frames
        shapes = [None] * n_frames
        for index, frame in enumerate(self.frames):
            shapes[index] = shape = frame.shape
            n_dims[index] = len(shape)

        max_dims = max(n_dims)
        shape_array = np.zeros((n_frames, max_dims), dtype='i')
        for index, s in enumerate(shapes):
            shape_array[index, :n_dims[index]] = s

        shape = [None] * max_dims
        for ax in range(max_dims):
            if ax == self.axis:
                shape[ax] = sum(shape_array[:, ax])
            else:
                shape[ax] = max(shape_array[:, ax])
        return tuple(shape)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_shape(self):
        if not self.validate_shape():
            warn(f"The dataframe '{self}' does not have a valid shape, returning minimum shape." )
        try:
            return self.get_min_shape.caching_call()
        except AttributeError:
            return self.get_min_shape()

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_lengths(self):
        shapes = self.shapes
        lengths = [0] * len(shapes)
        for index, shape in enumerate(shapes):
            lengths[index] = shape[self.axis]

        self.get_length.clear_cache()
        return tuple(lengths)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_length(self):
        return int(sum(self.lengths))

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_frame_start_indices(self):
        lengths = self.lengths
        starts = [None] * len(lengths)
        previous = 0
        for index, frame_length in enumerate(self.lengths):
            starts[index] = int(previous)
            previous += frame_length
        return tuple(starts)

    @timed_keyless_cache_method(call_method="clearing_call", collective=False)
    def get_frame_end_indices(self):
        lengths = self.lengths
        ends = [None] * len(lengths)
        previous = -1
        for index, frame_length in enumerate(self.lengths):
            previous += frame_length
            ends[index] = int(previous)
        return tuple(ends)

    def get_item(self, item):
        if isinstance(item, slice):
            return self.get_range_slice(item)
        elif isinstance(item, (tuple, list)):
            is_slices = True
            for element in item:
                if isinstance(element, int):
                    is_slices = False
                    break
            if is_slices:
                return self.get_ranges(item)
            else:
                return self.get_index(item)
        elif isinstance(item, int):
            return self.get_frame(item)
        elif item is Ellipsis:
            return self.get_all_data()

    # Setters
    def set_editable_method(self, obj):
        self.editable_method = obj

    # Shape
    def validate_shape(self):
        shapes = list(self.shapes)
        if shapes:
            shape = list(shapes.pop())
            shape.pop(self.axis)
            for s in shapes:
                s = list(s)
                s.pop(self.axis)
                if s != shape:
                    return False
        return True

    def change_size(self, shape=None, **kwargs):
        if shape is None:
            shape = self.target_shape

        for frame in self.frames:
            if not frame.validate_shape() or frame.shape != shape:
                frame.change_size(shape, **kwargs)

    # Frames
    def frame_sort_key(self, frame):
        return frame

    def sort_frames(self, key=None, reverse=False):
        if key is None:
            key = self.frame_sort_key
        self.frames.sort(key=key, reverse=reverse)

    # Container
    def append(self, item):
        self.frames.append(item)

    # General
    # Todo: change how numpy appending works for a speed boost.
    def smart_append(self, a, b, axis=None):
        if axis is None:
            axis = self.axis

        if isinstance(a, np.ndarray):
            return np.append(a, b, axis)
        else:
            return a + b

    # Find within Frames
    def find_inner_frame_index(self, index):
        length = self.length
        frame_start_indices = self.frame_start_indices

        # Check if index is in range.
        if index >= length or (index + length) < 0:
            raise IndexError("index is out of range")

        # Change negative indexing into positive.
        if index < 0:
            index = length - index

        # Find
        frame_index = bisect_right(frame_start_indices, index) - 1
        frame_start_index = frame_start_indices[frame_index]
        return frame_index, int(index - frame_start_index), frame_start_index

    def find_inner_frame_indices(self, indices):
        length = self.length
        frame_start_indices = self.frame_start_indices
        indices = list(indices)
        inner_indices = [None] * len(indices)

        for i, index in enumerate(indices):
            # Check if index is in range.
            if index >= length or (index + length) < 0:
                raise IndexError("index is out of range")

            # Change negative indexing into positive.
            if index < 0:
                indices[i] = self.length + index

        if len(indices) <= 32:
            for i, index in enumerate(indices):
                frame_index = bisect_right(frame_start_indices, index) - 1
                frame_start_index = frame_start_indices[frame_index]
                inner_indices[i] = [frame_index, int(index - frame_start_index), frame_start_index]
        else:
            frame_indices = list(np.searchsorted(frame_start_indices, indices, side='right') - 1)
            for i, frame_index in enumerate(frame_indices):
                frame_start_index = frame_start_indices[frame_index]
                inner_indices[i] = [frame_index, int(indices[i] - frame_start_index), frame_start_index]

        return inner_indices

    # Get a Range of Frames
    def get_range(self, start=None, stop=None, step=None, frame=None):
        if start is not None and stop is not None:
            start_index, stop_index = self.find_inner_frame_indices([start, stop])
        elif start is not None:
            start_index = self.find_inner_frame_index(start)
            stop_index = [len(self.frames) - 1, None, None]
        elif stop is not None:
            stop_index = self.find_inner_frame_index(stop)
            start_index = [0, None, None]
        else:
            start_index = [0, None, None]
            stop_index = [len(self.frames) - 1, None, None]

        start_frame, start_inner, _ = start_index
        stop_frame, stop_inner, _ = stop_index

        return self.get_range_frame(start_frame, stop_frame, start_inner, stop_inner, step, frame)

    def get_range_frame(self, frame_start=None, frame_stop=None, inner_start=None, inner_stop=None, step=None,
                        frame=None):
        if (frame is None and self.returns_frame) or frame:
            return self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
        else:
            return self.get_range_frame_array(frame_start, frame_stop, inner_start, inner_stop, step)

    def get_range_frame_array(self, frame_start=None, frame_stop=None, inner_start=None, inner_stop=None, step=None):
        frame_start_indices = self.frame_start_indices

        # Parse Arguments
        if frame_start is None:
            frame_start = 0
        elif frame_start < 0:
            frame_start = len(self.frames) + frame_start
        if inner_start is None:
            inner_start = 0
        if frame_stop is None:
            frame_stop = len(self.frames) - 1
        elif frame_stop < 0:
            frame_stop = len(self.frames) + frame_stop
        if inner_stop is None:
            inner_stop = self.lengths[frame_stop]

        # Get Data
        if frame_start == frame_stop:
            data = self.frames[frame_start][inner_start:inner_stop:step]
        else:
            try:
                t_shape = list(self.get_max_shape.caching_call())
            except AttributeError:
                t_shape = list(self.get_max_shape())
            samples = frame_start_indices[frame_stop] + inner_stop - frame_start_indices[frame_start] - inner_start
            if step is not None:
                samples = math.ceil(samples/step)
            t_shape[self.axis] = samples
            data = np.empty(t_shape)
            data.fill(np.nan)

            previous = 0
            frist_data = self.frames[frame_start][inner_start::step]
            f_shape = frist_data.shape
            new = f_shape[self.axis]
            slices = [slice(0, length) for length in f_shape]
            slices[self.axis] = slice(previous, new)
            data[tuple(slices)] = frist_data
            previous = new

            for fi in range(frame_start + 1, frame_stop):
                n_data = self.frames[fi][::step]
                n_shape = n_data.shape
                new = previous + n_shape[self.axis]
                slices = [slice(0, length) for length in f_shape]
                slices[self.axis] = slice(previous, new)
                data[tuple(slices)] = n_data
                previous = new

            last_data = self.frames[frame_stop][:inner_stop:step]
            l_shape = last_data.shape
            new = previous + l_shape[self.axis]
            slices = [slice(0, length) for length in f_shape]
            slices[self.axis] = slice(previous, new)
            data[tuple(slices)] = last_data

        return data

    # Get a Range of Frames with a Slice
    def get_range_slice(self, item, frame=None):
        return self.get_range(item.start, item.stop, item.step, frame)

    # Get a Range of Frames with Slices
    def get_ranges(self, slices, axis=None, frame=None):
        if axis is None:
            axis = self.axis

        slices = list(slices)
        slice_ = slices[axis]
        start = slice_.start
        stop = slice_.stop
        step = slice_.step

        if start is not None and stop is not None:
            start_index, stop_index = self.find_inner_frame_indices([start, stop])
        elif start is not None:
            start_index = self.find_inner_frame_index(start)
            stop_index = [None, None, None]
        elif stop is not None:
            stop_index = self.find_inner_frame_index(stop)
            start_index = [None, None, None]
        else:
            start_index = [None, None, None]
            stop_index = [None, None, None]

        start_frame, start_inner, _ = start_index
        stop_frame, stop_inner, _ = stop_index

        slices[axis] = slice(start_inner, stop_inner, step)

        return self.get_ranges_frame(start_frame, stop_frame, slices, axis, frame)

    def get_ranges_frame(self, frame_start=None, frame_stop=None, slices=None, axis=None, frame=None):
        if (frame is None and self.returns_frame) or frame:
            data = self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
        else:
            if axis is None:
                axis = self.axis
            if frame_start is None:
                frame_start = 0
            if frame_stop is None:
                frame_stop = -1

            if frame_start == frame_stop:
                data = self.frames[frame_start][slices]
            else:
                slice_ = slices[axis]

                first = list(slices)
                inner = list(slices)
                last = list(slices)

                first[axis] = slice(slice_.start, None, slice_.step)
                inner[axis] = slice(None, None, slice_.step)
                last[axis] = slice(None, slice_.stop, slice_.step)

                data = self.frames[frame_start][tuple(first)]
                for fi in range(frame_start + 1, frame_stop):
                    data = self.smart_append(data, self.frames[fi][tuple(inner)])
                data = self.smart_append(data, self.frames[frame_stop][tuple(last)])
        return data

    # Get Frame based on Index
    def get_frame(self, index, frame=None):
        if (frame is None and self.returns_frame) or frame:
            return self.frames[index]
        else:
            return self.frames[index][...]

    def get_all_data(self, frame=None):
        if (frame is None and self.returns_frame) or frame:
            return self
        else:
            data = self.frames[0][...]
            for index in range(1, len(self.frames)):
                data = self.smart_append(data, self.frames[index][...])
            return data

    # Get Frame within by Index
    def get_index(self, indices, reverse=False, frame=True):
        if len(indices) == 1:
            item = indices[0]
            if isinstance(item, int):
                return self.get_frame(item, frame)
            elif isinstance(item, slice):
                return self.get_range_slice(item, frame)
            else:
                return self.get_ranges(item, frame=frame)
        else:
            indices = list(indices)
            if not reverse:
                index = indices.pop(0)
            else:
                index = indices.pop()

            return self.frames[index].get_index(indices, reverse, frame)

    def get_range_indices(self, start=None, stop=None, step=None, reverse=False, frame=None):
        if frame is None:
            frame = self.returns_frame

        if start is None and stop is None:
            if frame:
                return self
            else:
                return self[::step]

        frame_start = None
        frame_stop = None
        if start is not None and len(start) > 1:
            start = list(start)
            if not reverse:
                frame_start = start.pop(0)
            else:
                frame_start = start.pop()

        if stop is not None and len(stop) > 1:
            stop = list(stop)
            if not reverse:
                frame_stop = stop.pop(0)
            else:
                frame_stop = stop.pop()

        if frame_start and frame_start and (frame_start + 1) == frame_stop:
            return self[frame_start].get_range_indices(start, stop, step, reverse, frame)

        if start is None:
            if frame:
                start_data = self.return_frame_type(frames=[self.frames[frame_start:]])
            else:
                start_data = self.frames[0][::step]
                for fi in range(1, frame_stop):
                    start_data = self.smart_append(start_data, self.frames[fi][::step])
        elif len(start) == 1:
            start_data = self.get_frame_within(start, reverse, frame)
        else:
            start_data = self.frames[frame_start].get_range_indices(start, None, step, reverse, frame)

        if stop is None:
            if frame:
                stop_data = self.return_frame_type(frames=[self.frames[:frame_stop]])
            else:
                stop_data = self.smart_append(start_data, self.frames[frame_start + 1][::step])
                for fi in range(frame_start + 2, len(self.frames)):
                    stop_data = self.smart_append(stop_data, self.frames[fi][::step])
        elif len(stop) == 1:
            stop_data = self.get_frame_within(stop, reverse, frame)
        else:
            stop_data = self.frames[frame_stop].get_range_indices(None, stop, step, reverse, frame)

        if start is not None and stop is not None:
            if frame:
                data = self.return_frame_type(frames=[self.frames[frame_start:frame_stop]])
                self.smart_append(start_data, data)
            else:
                for fi in range(frame_start + 2, frame_stop):
                    start_data = self.smart_append(start_data, self.frames[fi][::step])

        return self.smart_append(start_data, stop_data)

    # Combine
    def combine_frames(self, start=None, stop=None, step=None):
        return self.combine_type(frames=self.frames[start:stop:step])


# Assign Cyclic Definitions
DataFrame.default_return_frame_type = DataFrame
