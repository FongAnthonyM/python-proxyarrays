"""proxyarray.py
A proxy for holding different data types which are similar to a numpy array.
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
from collections.abc import Iterable, Iterator, Generator
from bisect import bisect_right
from itertools import chain
from math import ceil, floor
from typing import Any, NamedTuple, Union
from warnings import warn

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch
from baseobjects.cachingtools import timed_keyless_cache
import numpy as np

# Local Packages #
from .baseproxyarray import BaseProxyArray, Slice
from proxyarrays.proxyarray.containerproxyarray import ContainerProxyArray


# Definitions #
# Classes #
class ProxyIndex(NamedTuple):
    index: int | None
    start_index: int | None
    inner_index: int | None


class SliceIndices(NamedTuple):
    start: ProxyIndex | int | None
    stop: ProxyIndex | int | None


class ProxyArray(BaseProxyArray):
    """A proxy for holding different data types which are similar to a numpy array.

    This object acts as an abstraction of several contained numpy like objects, to appear as combined numpy array.
    Accessing the data within can be used with standard numpy data indexing which is based on the overall data at the
    deepest layer of the structure.

    Class Attributes:
        default_return_proxy_type: The default type of proxy to return when returning a proxy.
        default_combine_type: The default type of combined proxy to return when combining proxies.

    Attributes:
        mode: Determines if this proxy is editable or read only.
        target_shape: The shape that proxy should be and if resized the shape it will default to.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        combine_type: The object type to return when combining proxies.
        return_proxy_type: The proxy type to return when returning proxies from this object.
        proxies: The list of proxies/objects in this proxy.

    Args:
        proxies: An iterable holding proxies/objects to store in this proxy.
        axis: The axis of the data which this proxy extends for the contained data proxies.
        mode: Determines if the contents of this proxy are editable or not.
        update: Determines if this proxy will start updating or not.
        *args: Arguments for inheritance.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """

    # TODO: Consider making the proxy work multidimensional. (Only single-dimensional right now.)
    default_return_proxy_type: type = BaseProxyArray
    default_combine_type: type | None = ContainerProxyArray

    # Magic Methods
    # Construction/Destruction
    def __init__(
        self,
        proxies: Iterable[BaseProxyArray] | None = None,
        axis: int = 0,
        mode: str = "a",
        update: bool = True,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        # Shape
        self.target_shape: Iterable[int] | None = None
        self.axis: int = 0

        # Assign Classes #
        self.combine_type: type = self.default_combine_type

        # Containers #
        self.proxies: list = []

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Object Construction #
        if init:
            self.construct(proxies=proxies, axis=axis, mode=mode, update=update, **kwargs)

    @property
    def shapes(self) -> tuple[tuple[int]]:
        """Returns the shapes of all contained proxies and uses cached value if available."""
        try:
            return self.get_shapes.caching_call()
        except AttributeError:
            return self.get_shapes()

    @property
    def min_shape(self) -> tuple[tuple[int]]:
        """Get the minimum shapes from the contained proxies/objects if they are different across axes."""
        try:
            return self.get_min_shape.caching_call()
        except AttributeError:
            return self.get_min_shape()

    @property
    def max_shape(self) -> tuple[tuple[int]]:
        """Get the maximum shapes from the contained proxies/objects if they are different across axes."""
        try:
            return self.get_max_shape.caching_call()
        except AttributeError:
            return self.get_max_shape()

    @property
    def shape(self) -> tuple[int]:
        """Returns the shape of this proxy if all contained shapes are the same and uses cached value if available."""
        try:
            return self.get_shape.caching_call()
        except AttributeError:
            return self.get_shape()

    @property
    def max_ndim(self) -> int:
        """The maximum number dimension in the contained proxies/objects if they are different across axes."""
        return len(self.max_shape)

    @property
    def min_ndim(self) -> int:
        """The minimum number dimension in the contained proxies/objects if they are different across axes."""
        return len(self.min_shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of this array."""
        if not self.validate_shape():
            warn(f"The ProxyArray '{self}' does not have a valid shape, returning minimum shape.")
        return self.min_ndim

    @property
    def lengths(self) -> tuple[int]:
        """Returns the lengths of each contained proxies as a tuple and uses cached value if available."""
        try:
            return self.get_lengths.caching_call()
        except AttributeError:
            return self.get_lengths()

    @property
    def length(self) -> int:
        """Returns the sum of all lengths of contained proxies and uses cached value if available."""
        try:
            return self.get_length.caching_call()
        except AttributeError:
            return self.get_length()

    @property
    def proxy_start_indices(self) -> tuple[int]:
        """Returns the start index of the contained proxies and uses cached value if available."""
        try:
            return self.get_proxy_start_indices.caching_call()
        except AttributeError:
            return self.get_proxy_start_indices()

    @property
    def proxy_end_indices(self) -> tuple[int]:
        """Returns the end index of the contained proxies and uses cached value if available."""
        try:
            return self.get_proxy_end_indices.caching_call()
        except AttributeError:
            return self.get_proxy_end_indices()

    # Container Methods
    def __len__(self) -> int:
        """Gets this object's length.

        Returns:
            The number of nodes in this object.
        """
        try:
            return self.get_length.caching_call()
        except AttributeError:
            return self.get_length()

    # Arithmetic
    def __add__(self, other: BaseProxyArray | list):
        """When the add operator is called it concatenates this proxy with other proxies or a list."""
        return self.concatenate(other=other)

    # Numpy ndarray Methods
    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Returns an ndarray representation of this object with an option to cast it to a dtype.

        Allows this object to be used as ndarray in numpy functions.

        Args:
            dtype: The dtype to cast the array to.

        Returns:
            The ndarray representation of this object.
        """
        return self.slices_array(dtype=dtype)

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        proxies: Iterable[BaseProxyArray] = None,
        axis: int | None = None,
        mode: str | None = None,
        update: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            proxies: An iterable holding proxies/objects to store in this proxy.
            axis: The axis of the data which this proxy extends for the contained data proxies.
            mode: Determines if the contents of this proxy are editable or not.
            update: Determines if this proxy will start updating or not.
            **kwargs: Keyword arguments for inheritance.
        """
        if axis is not None:
            self.axis = axis

        if update is not None:
            self._is_updating = update
            if update:
                self.enable_updating()
            else:
                self.disable_updating()

        if proxies is not None:
            self.proxies.clear()
            self.proxies.extend(proxies)

        if mode is not None:
            self.mode = mode

        super().construct(**kwargs)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "ProxyArray":
        """Create a new copy of this object without proxies.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)
        new_copy.target_shape = self.target_shape
        new_copy.axis = self.axis
        new_copy.combine_type = self.default_combine_type
        new_copy.return_proxy_type = self.default_return_proxy_type
        return new_copy

    def flat_copy(self) -> "ProxyArray":
        """Creates a flattened (proxy depth one) copy of this object.

        Returns:
            The flat copy of this object.
        """
        new_copy = self.empty_copy()
        new_copy.proxies.extend(self.flat_iterator())
        return new_copy

    def proxy_leaf_copy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a copy proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The copy of this proxy array.
        """
        proxy_copy = super().proxy_leaf_copy(type_=type_, **kwargs)
        proxy_copy.data = self.slices_array()
        return proxy_copy

    # Editable Copy Methods
    def _default_spawn_editable(self, *args: Any, **kwargs: Any) -> BaseProxyArray:
        """The default method for creating an editable version of this proxy.

        Returns:
            An editable version of this proxy.
        """
        return self.combine_proxies()

    # Cache and Memory
    def refresh(self) -> None:
        """Resets this proxy's caches and fills them with updated values."""
        self.get_shapes()
        self.get_shape()
        self.get_lengths()
        self.get_length()

    def clear_all_caches(self) -> None:
        """Clears this proxy's caches and all the contained proxy's caches."""
        self.clear_caches()
        for proxy in self.proxies:
            try:
                proxy.clear_all_caches()
            except AttributeError:
                continue

    # Updating
    def enable_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this proxy and all contained proxies/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self.timed_caching(get_caches=get_caches)
        for proxy in self.proxies:
            proxy.enable_updating(get_caches=get_caches)

    def enable_last_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this proxy and the last contained proxy/object.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self.timed_caching(get_caches=get_caches)
        try:
            self.proxies[-1].enable_updating(get_caches=get_caches)
        except IndexError as e:
            pass  # Maybe raise warning.

    def disable_updating(self, get_caches: bool = False) -> None:
        """Disables updating for this proxy and all contained proxies/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self.timeless_caching(get_caches=get_caches)
        for proxy in self.proxies:
            proxy.disable_updating(get_caches=get_caches)

    def clear(self) -> None:
        """Clears this object."""
        self.proxies.clear()
        self.clear_caches()

    # Getters
    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_any_updating(self) -> bool:
        """Checks if any contained proxies/objects are updating.

        Returns:
            If any contained proxies/objects are updating.
        """
        for proxy in self.proxies:
            if proxy.get_any_updating():
                return True
        return False

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_shapes(self) -> tuple[tuple[int]]:
        """Get the shapes from the contained proxies/objects.

        Returns:
            The shapes of the contained proxies/objects.
        """
        return tuple(proxy.get_shape() for proxy in self.proxies)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_min_shape(self) -> tuple[int]:
        """Get the minimum shapes from the contained proxies/objects if they are different across axes.

        Returns:
            The minimum shapes of the contained proxies/objects.
        """
        n_proxies = len(self.proxies)
        n_dims = [None] * n_proxies
        shapes = [None] * n_proxies
        for index, proxy in enumerate(self.proxies):
            shapes[index] = shape = proxy.get_shape()
            n_dims[index] = len(shape)

        max_dims = max(n_dims)
        shape_array = np.zeros((n_proxies, max_dims), dtype="i")
        for index, s in enumerate(shapes):
            shape_array[index, : n_dims[index]] = s

        shape = [None] * max_dims
        for ax in range(max_dims):
            if ax == self.axis:
                shape[ax] = shape_array[:, ax].sum()
            else:
                shape[ax] = shape_array[:, ax].min()
        return tuple(shape)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_max_shape(self) -> tuple[int]:
        """Get the maximum shapes from the contained proxies/objects if they are different across axes.

        Returns:
            The maximum shapes of the contained proxies/objects.
        """
        n_proxies = len(self.proxies)
        n_dims = [None] * n_proxies
        shapes = [None] * n_proxies
        for index, proxy in enumerate(self.proxies):
            shapes[index] = shape = proxy.get_shape()
            n_dims[index] = len(shape)

        max_dims = max(n_dims)
        shape_array = np.zeros((n_proxies, max_dims), dtype="i")
        for index, s in enumerate(shapes):
            shape_array[index, : n_dims[index]] = s

        shape = [None] * max_dims
        for ax in range(max_dims):
            if ax == self.axis:
                shape[ax] = shape_array[:, ax].sum()
            else:
                shape[ax] = shape_array[:, ax].max()
        return tuple(shape)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_shape(self) -> tuple[int]:
        """Get the shape of this proxy from the contained proxies/objects.

         If the contained proxies/object are different across axes this will raise a warning and return the minimum
         shape.

        Returns:
            The shape of this proxy or the minimum shapes of the contained proxies/objects.
        """
        if not self.validate_shape():
            warn(f"The ProxyArray '{self}' does not have a valid shape, returning minimum shape.")
        return self.get_min_shape()

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_lengths(self) -> tuple[int]:
        """Get the lengths of the contained proxies/objects.

        Returns:
            All the lengths of the contained proxies/objects.
        """
        shapes = self.get_shapes()
        lengths = [0] * len(shapes)
        for index, shape in enumerate(shapes):
            lengths[index] = shape[self.axis]

        return tuple(lengths)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_length(self) -> int:
        """Get the length of this proxy as the sum of the contained proxies/objects length.

        Returns:
            The length of this proxy.
        """
        return int(sum(self.get_lengths()))

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_proxy_start_indices(self) -> tuple[int]:
        """Get the start indices of the contained files based on the lengths of the contained proxies/objects.

        Returns:
            The start indices of each of the contained proxies/objects.
        """
        lengths = self.lengths
        starts = [None] * len(lengths)
        previous = 0
        for index, proxy_length in enumerate(self.lengths):
            starts[index] = int(previous)
            previous += proxy_length
        return tuple(starts)

    @timed_keyless_cache(lifetime=1.0, call_method="clearing_call", local=True)
    def get_proxy_end_indices(self) -> tuple[int]:
        """Get the end indices of the contained files based on the lengths of the contained proxies/objects.

        Returns:
            The end indices of each of the contained proxies/objects.
        """
        lengths = self.get_lengths()
        ends = [None] * len(lengths)
        previous = -1
        for index, proxy_length in enumerate(lengths):
            previous += proxy_length
            ends[index] = int(previous)
        return tuple(ends)

    # Main Get Item
    @singlekwargdispatch("item")
    def get_item(self, item: Any) -> Any:
        """Get an item from this proxy based on the input. For Ellipsis return all the data.

        Args:
            item: An object to get an item from this proxy.

        Returns:
            An object from this proxyarray
        """
        if item is Ellipsis:
            return self.get_all_data()
        else:
            raise TypeError(f"A {type(item)} cannot be used to get an item from {type(self)}.")

    @get_item.register
    def _(self, item: Slice) -> Any:
        """Get an item from this proxy based on a slice and return a range of contained data.

        Args:
            item: The slice to select the data range to return.

        Returns:
            The data of interest from within this proxy.
        """
        return self.slice(item)

    @get_item.register
    def _(self, item: Iterable) -> Any:
        """Get an item from this proxy based on an iterable and return a range of contained.

        Args:
            item: The slice to select the range to return.

        Returns:
            The of interest from within this proxy.
        """
        is_slices = True
        for element in item:
            if not isinstance(element, Slice):
                is_slices = False
                break

        if is_slices:
            return self.slices(item)
        else:
            return self.get_from_index(item)

    @get_item.register
    def _(self, item: int) -> BaseProxyArray:
        """Get an item from this proxy based on an int and return a proxy.

        Args:
            item: The index of the proxy to return.

        Returns:
            The proxy interest from within this proxy.
        """
        return self.get_proxy(item)

    # Shape
    def where_misshapen(self, shape: Iterable[int, ...] | None = None, exclude_axis: bool = True) -> tuple[int, ...]:
        """Gets the indices of the proxies which do not have matching shapes.

        Args:
            shape: The shape which all proxies must conform to.
            exclude_axis: Determines if the main axis will be excluded in the shape check.

        Returns:
            The indices of the misshapen proxies.
        """
        if shape is not None:
            shape = np.asarray(shape)
        elif self.target_shape is not None:
            shape = np.asarray(self.target_shape)
        else:
            raise ValueError("either a shape must be given or target shape must be set")
        shapes = np.asarray(self.shapes)
        if exclude_axis:
            return tuple(np.where(np.delete(shapes, self.axis, 1) != np.delete(shape, self.axis, 0))[0])
        else:
            return tuple(np.where(shapes != shape)[0])

    def where_shapes_change(self, exclude_axis: bool = True) -> tuple[int, ...]:
        """Gets the indices of the proxies where the shape does not match the previous shape.

        Args:
            exclude_axis: Determines if the main axis will be excluded in the shape check.

        Returns:
            The indices of the proxies that have different shapes than the previous.
        """
        shapes = np.delete(self.shapes, self.axis, 1) if exclude_axis else np.asarray(self.shapes)
        p_shapes = np.delete(shapes, -1, 0)
        n_shapes = np.delete(shapes, 0, 0)
        return tuple(np.where((p_shapes != n_shapes).any(axis=1))[0] + 1)

    def validate_shape(self) -> bool:
        """Checks if this proxy has a valid/continuous shape.

        Returns:
            If this proxy has a valid/continuous shape.
        """
        shapes = np.asarray(self.shapes)
        return np.delete(shapes == shapes[0], self.axis, 1).all()

    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the proxy without changing its data.

        Args:
            shape: The target shape to change this proxy to.
            **kwargs: Any additional kwargs need to change the shape of contained proxies/objects.
        """
        # Todo: Fix this for along length.
        if self.mode == "r":
            raise IOError("not writable")

        if shape is None:
            shape = self.target_shape

        for proxy in self.proxies:
            if not proxy.validate_shape() or proxy.shape != shape:
                proxy.change_size(shape, **kwargs)

    # Proxies
    def proxy_sort_key(self, proxy: Any) -> Any:
        """The key to be used in sorting with the proxy as the sort basis.

        Args:
            proxy: The proxy to sort.
        """
        return proxy

    def sort_proxies(self, key: Any = None, reverse: bool = False) -> None:
        """Sorts the contained proxies/objects.

        Args:
            key: The key base the sorting from.
            reverse: Determines if this proxy will be sorted in reverse order.
        """
        if key is None:
            key = self.proxy_sort_key
        self.proxies.sort(key=key, reverse=reverse)

    @singlekwargdispatch("other")
    def concatenate(self, other: BaseProxyArray | list[BaseProxyArray]) -> BaseProxyArray:
        """Concatenates this proxy object with another proxy or a list.

        Args:
            other: The other object to concatenate this proxy with.

        Returns:
            A new proxy which is the concatenation of this proxy and another object.
        """
        raise TypeError(f"A {type(other)} cannot be used to concatenate a {type(self)}.")

    @concatenate.register
    def _(self, other: BaseProxyArray) -> BaseProxyArray:
        """Concatenates this proxy object with another proxy.

        Args:
            other: The other proxy to concatenate this proxy with.

        Returns:
            A new proxy which is the concatenation of this proxy and the other proxy.
        """
        return type(self)(proxies=self.proxies + other.proxies, update=self.is_updating)

    @concatenate.register(list)
    def _(self, other: list[BaseProxyArray]) -> BaseProxyArray:
        """Concatenates this proxy object with another list.

        Args:
            other: The other list to concatenate this proxy with.

        Returns:
            A new proxy which is the concatenation of this proxy and another list.
        """
        return type(self)(proxies=self.proxies + other, update=self.is_updating)

    def append(self, item: Any) -> None:
        """Append an item to the proxy.

        Args:
            item: The object to add to this proxy.
        """
        self.proxies.append(item)

    def combine_proxies(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None
    ) -> BaseProxyArray:
        """Combines a range of proxies into a single proxy.

        Args:
            start: The start proxy.
            stop: The stop proxy.
            step: The step between proxies to combine.

        Returns:
            A single combined proxy.
        """
        return self.combine_type(proxies=self.proxies[start:stop:step])

    def flat_iterator(self) -> Iterator[BaseProxyArray, ...]:
        """Creates a generator which iterates over the innermost proxies.

        Returns:
            The innermost proxies.
        """
        return chain.from_iterable((p.flat_iterator() for p in self.proxies))

    def as_flattened(self, type_: type[BaseProxyArray] | None = None, **kwargs: Any) -> BaseProxyArray:
        """Creates a proxy array which contains flattened (proxy depth one) contents of this object.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The flat copy of this object.
        """
        return self.create_return_proxy(type_=type_, proxies=self.flat_iterator(), **kwargs)

    # Get proxy within by Index
    @singlekwargdispatch("indices")
    def get_from_index(
        self,
        indices: Iterable[int | Slice | Iterable] | int | Slice,
        reverse: bool = False,
        proxy: bool | None = None,
    ) -> Any:
        """Gets data from this object if given an index which can be in serval formats.

        Args:
            indices: The indices to find the data from.
            reverse:  Determines, when using a Iterable of indices, if it will be read in reverse order.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested proxy or data.
        """
        raise TypeError(f"A {type(indices)} be used to get from super_index for a {type(self)}.")

    @get_from_index.register(Iterable)
    def _(
        self,
        indices: Iterable[int | Slice | Iterable[Slice | int | None]],
        reverse: bool = False,
        proxy: bool | None = None,
    ) -> BaseProxyArray | np.ndarray:
        """Gets a nested proxy or data from within this proxy.

        Args:
            indices: A series of indices of the nested proxies to get from, can end with slices to get ranges of data.
            reverse: Determines if the indices series will be read in reverse order.
            proxy:  Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested proxy or data.
        """
        indices = list(indices)
        if not reverse:
            index = indices.pop(0)
        else:
            index = indices.pop()

        if indices:
            return self.proxies[index].get_from_index(indices=indices, reverse=reverse, proxy=proxy)
        elif isinstance(index, Iterable):
            return self.slices(slices=index, proxy=proxy)
        elif isinstance(index, int):
            return self.get_from_index(indices=index, proxy=proxy)
        else:
            return self.slice(item=indices, proxy=proxy)

    @get_from_index.register
    def _(self, indices: int, proxy: bool | None = None) -> Any:
        """Get a contained proxy/object or data from a contained proxy/object.

        Args:
            indices: The proxy index to get the data from.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested proxy or data.
        """
        proxy_index, _, inner_index = self.find_inner_proxy_index(super_index=indices)
        return self.proxies[proxy_index].get_from_index(indices=inner_index, proxy=proxy)

    # Find Inner Indices within proxies
    def find_inner_proxy_index(self, super_index: int) -> ProxyIndex:
        """Find the proxy and inner index of a super index.

        Args:
            super_index: The super index to find.

        Returns:
            The index information as a ProxyIndex.
        """
        length = self.length
        proxy_start_indices = self.proxy_start_indices

        # Check if index is in range.
        if super_index >= length or (super_index + length) < 0:
            raise IndexError("index is out of range")

        # Change negative indexing into positive.
        if super_index < 0:
            super_index = length - super_index

        # Find
        proxy_index = bisect_right(proxy_start_indices, super_index) - 1
        proxy_start_index = proxy_start_indices[proxy_index]
        proxy_inner_index = int(super_index - proxy_start_index)
        return ProxyIndex(proxy_index, proxy_start_index, proxy_inner_index)

    def find_inner_proxy_indices(self, super_indices: Iterable[int]) -> tuple[ProxyIndex, ...]:
        """Find the proxy and inner index of several super indices.

        Args:
            super_indices: The super indices to find.

        Returns:
            The indices' information as ProxyIndex objects.
        """
        length = self.length
        proxy_start_indices = self.proxy_start_indices
        super_indices = list(super_indices)
        inner_indices = [None] * len(super_indices)

        # Validate Indices
        for i, super_index in enumerate(super_indices):
            # Check if super_index is in range.
            if super_index >= length or (super_index + length) < 0:
                raise IndexError("super_index is out of range")

            # Change negative indexing into positive.
            if super_index < 0:
                super_indices[i] = self.length + super_index

        # Finding Methods
        if len(super_indices) <= 32:  # Few indices to find
            for i, super_index in enumerate(super_indices):
                proxy_index = bisect_right(proxy_start_indices, super_index) - 1
                proxy_start_index = proxy_start_indices[proxy_index]
                proxy_inner_index = int(super_index - proxy_start_index)
                inner_indices[i] = ProxyIndex(proxy_index, proxy_start_index, proxy_inner_index)
        else:  # Many indices to find
            proxy_indices = list(np.searchsorted(proxy_start_indices, super_indices, side="left"))
            for i, proxy_index in enumerate(proxy_indices):
                proxy_start_index = proxy_start_indices[proxy_index]
                proxy_inner_index = int(super_indices[i] - proxy_start_index)
                inner_indices[i] = ProxyIndex(proxy_index, proxy_start_index, proxy_inner_index)

        return tuple(inner_indices)

    def find_inner_proxy_indices_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SliceIndices:
        """Finds the inner indices for a slice and returns them as ProxyIndex objects.

        Args:
            start: The start index of the range.
            stop: The stop index of the range.
            step: The interval of the slice.

        Returns:
            The start and stop indices as ProxyIndex objects in a SliceIndices object.
        """
        length = self.length
        if start is None:
            start = 0

        if stop is None:
            stop = length
        elif stop < 1:
            stop = length + stop

        if step is None:
            step = 1

        stop = floor((stop - start) / step) * step + start

        if start == 0 and stop == 0:
            start_index = ProxyIndex(0, 0, 0)
            stop_index = ProxyIndex(0, 0, 0)
        elif start == 0:
            start_index = ProxyIndex(0, 0, 0)
            stop_index = self.find_inner_proxy_index(stop - 1)
            stop_index = ProxyIndex(stop_index[0], stop_index[1], stop_index[2] + 1)
        elif stop == 0:
            start_index = self.find_inner_proxy_index(start)
            stop_index = ProxyIndex(0, 0, 0)
        else:
            start_index, stop_index = self.find_inner_proxy_indices([start, stop - 1])
            stop_index = ProxyIndex(stop_index[0], stop_index[1], stop_index[2] + 1)

        return SliceIndices(start_index, stop_index)

    # Data Slicing
    def fill_slices_array(
        self,
        data_array: np.ndarray,
        array_slices: Iterable[Slice] | None = None,
        slices: Iterable[Slice | int | None] | None = None,
    ) -> np.ndarray:
        """Fills a given array with values from the contained proxies/objects.

        Args:
            data_array: The numpy array to fill.
            array_slices: The slices to fill within the data_array.
            slices: The slices to get the data from.

        Returns:
            The original array but filled.
        """
        slices = list(slices)
        # Get indices range
        da_shape = data_array.shape
        axis_slice = slices[self.axis]

        # Return no change if start and stop is the same.
        if axis_slice.start == axis_slice.stop and axis_slice.start is not None:
            return data_array

        range_proxy_indices = self.find_inner_proxy_indices_slice(start=axis_slice.start, stop=axis_slice.stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        inner_start = range_proxy_indices.start.inner_index
        inner_stop = range_proxy_indices.stop.inner_index
        slices[self.axis] = Slice(inner_start, inner_stop, axis_slice.step)

        # Get start and stop data array locations
        array_slices = [Slice(None)] * len(da_shape) if array_slices is None else list(array_slices)

        da_axis_slice = array_slices[self.axis]
        array_start = 0 if da_axis_slice.start is None else da_axis_slice.start
        array_stop = da_shape[self.axis] if da_axis_slice.stop is None else da_axis_slice.stop

        # Contained proxy/object fill kwargs
        fill_kwargs = {
            "data_array": data_array,
            "array_slices": array_slices,
            "slices": slices,
        }

        # Get Data
        if start_proxy == stop_proxy:
            self.proxies[start_proxy].fill_slices_array(**fill_kwargs)
        else:
            # First proxy
            proxy = self.proxies[start_proxy]
            d_size = len(proxy) - inner_start
            a_stop = array_start + d_size
            array_slices[self.axis] = Slice(array_start, a_stop, da_axis_slice.step)
            slices[self.axis] = Slice(inner_start, None, axis_slice.step)
            proxy.fill_slices_array(**fill_kwargs)

            # Middle proxies
            slices[self.axis] = Slice(None, None, axis_slice.step)
            for proxy in self.proxies[start_proxy + 1: stop_proxy]:
                d_size = len(proxy)
                a_start = a_stop
                a_stop = a_start + d_size
                array_slices[self.axis] = Slice(a_start, a_stop, da_axis_slice.step)
                proxy.fill_slices_array(**fill_kwargs)

            # Last proxy
            a_start = a_stop
            array_slices[self.axis] = Slice(a_start, array_stop, da_axis_slice.step)
            slices[self.axis] = Slice(None, inner_stop, axis_slice.step)
            self.proxies[stop_proxy].fill_slices_array(**fill_kwargs)

        return data_array

    def slices_array(
        self,
        slices: Iterable[Slice | int | None] | None = None,
        dtype: Any = None,
    ) -> np.ndarray:
        """Get data as an array using slices to determine the data slice.

        Args:
            slices: The ranges to get the data from.
            dtype: The dtype of array to return.

        Returns:
            The requested slice as an array.
        """
        if slices is None:
            slices = [Slice(None)] * self.max_ndim

        # Create nan numpy array
        slices = list(slices)
        full_slices = slices + [Slice(None)] * (self.max_ndim - len(slices))
        s_shape = [None] * len(full_slices)
        for i, slice_ in enumerate(full_slices):
            if slice_ is not None:
                start = 0 if slice_.start is None else slice_.start
                stop = self.max_shape[i] if slice_.stop is None else slice_.stop
                step = 1 if slice_.step is None else slice_.step
                s_shape[i] = int(stop - start) // step
            else:
                s_shape[i] = 1
        data = np.empty(shape=s_shape, dtype=dtype)
        # data.fill(np.nan)

        # Get range via filling the array with values
        return self.fill_slices_array(data_array=data, slices=full_slices)

    def slices_proxy(self, slices: Iterable[Slice] | None = None) -> BaseProxyArray:
        """Get data as a new proxy using slices to determine the data slice.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested slice as a proxy.
        """
        slices = list(slices)
        axis_slice = slices[self.axis]
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=axis_slice.start, stop=axis_slice.stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index + 1

        return self.create_return_proxy(proxies=self.proxies[start_proxy:stop_proxy])

    def islices(
        self,
        slices: Iterable[Slice | int | None] | None = None,
        islice: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Generator[Union["ContainerProxyArray", np.ndarray], None, None]:
        """Creates a generator which iterates over slices along an axis.

        Args:
            slices: The ranges of the data to get.
            islice: The range to data to iterate over.
            axis: The axis to iterate along.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        if axis is None:
            axis = self.axis

        length = len(self)
        if isinstance(slices, int):
            slice_size = slices
            slices = [slice(None, slice_size)]
            axis_step = None
        else:
            slices = list(slices)
            axis_slice = slices[self.axis]
            axis_start = 0 if axis_slice.start is None else axis_slice.start
            axis_stop = self.length if axis_slice.stop is None else axis_slice.stop
            axis_step = axis_slice.step
            slice_size = axis_stop - axis_start

        # Get shape of slices
        full_slices = slices + [Slice(None)] * (self.max_ndim - len(slices))
        s_shape = [None] * len(full_slices)
        for i, slice_ in enumerate(full_slices):
            if slice_ is not None:
                start = 0 if slice_.start is None else slice_.start
                stop = self.max_shape[i] if slice_.stop is None else slice_.stop
                s_shape[i] = int(stop - start)
            else:
                s_shape[i] = 1

        # Get iteration slice information
        if islice is None:
            istep = None
            outer_start = 0
            outer_stop = length
            outer_step = slice_size
        else:
            istep = islice.step
            outer_start = 0 if islice.start is None else islice.start
            outer_stop = length if islice.stop is None else islice.stop
            outer_step = slice_size * (1 if islice.step is None else islice.step)

        # Adjust outer stop if there is not enough data to fill last slice
        last_index = (((length - outer_start) // outer_step) * outer_step + outer_start)
        adjusted_stop = last_index if (length - last_index) > slice_size else outer_stop

        # Get indices range
        range_proxy_indices = self.find_inner_proxy_indices_slice(start=outer_start, stop=adjusted_stop)

        start_proxy = range_proxy_indices.start.index
        stop_proxy = range_proxy_indices.stop.index
        inner_start = range_proxy_indices.start.inner_index
        inner_stop = range_proxy_indices.stop.inner_index

        # Iterate over data
        if start_proxy == stop_proxy:
            # If all data is in a proxy, call its islice.
            iter_ = self.proxies[start_proxy].islices(
                slices=full_slices,
                islice=Slice(inner_start, inner_stop, istep),
                dtype=dtype,
                proxy=proxy,
            )
            for item in iter_:
                yield item

        else:
            iter_start = inner_start
            fill_array_slices = [Slice(None)] * len(s_shape)
            fill_slices = full_slices.copy()
            gap_data = None
            gap_offset = 0
            for proxy_index in range(start_proxy, stop_proxy):
                proxy_length = self.lengths[proxy_index]
                # Adjust inner iterator stop if there is not enough data to fill last slice
                last_proxy_index = (((proxy_length - iter_start) // outer_step) * outer_step + iter_start)
                remainder = proxy_length - last_proxy_index
                adjusted_iter_stop = last_proxy_index if remainder < slice_size else proxy_length

                # Handle Gap data
                if gap_data is not None:
                    gap_fill = min(gap_offset + proxy_length, slice_size)
                    fill_array_slices[axis] = Slice(gap_offset, gap_fill)
                    fill_slices[axis] = Slice(0, gap_fill - gap_offset)
                    self.proxies[proxy_index].fill_slices_array(
                        data_array=gap_data,
                        array_slices=fill_array_slices,
                        slices=fill_slices,
                    )
                    if gap_fill == slice_size:
                        yield self.create_return_proxy_node(data=gap_data[::axis_step]) if proxy else gap_data
                        gap_data = None
                        gap_offset = 0
                    else:
                        gap_offset = gap_fill
                        continue

                # Iterate inner proxy
                iter_ = self.proxies[proxy_index].islices(
                    slices=full_slices,
                    islice=Slice(iter_start, adjusted_iter_stop, istep),
                    dtype=dtype,
                    proxy=proxy,
                )
                for item in iter_:
                    yield item

                # Yield gap data
                if remainder != 0 and remainder < slice_size:
                    # Create Array
                    gap_data = np.empty(shape=s_shape, dtype=dtype)

                    fill_array_slices[axis] = Slice(None, remainder)
                    fill_slices[axis] = Slice(adjusted_iter_stop, None)
                    self.proxies[proxy_index].fill_slices_array(
                        data_array=gap_data,
                        array_slices=fill_array_slices,
                        slices=fill_slices,
                    )
                    gap_offset = remainder

                # Setup next loop
                iter_start = 0 if remainder == 0 else outer_step - remainder
                self.proxies[proxy_index].clear_all_caches()

            # Last proxy iteration
            proxy_length = self.lengths[stop_proxy]
            # Adjust inner iterator stop if there is not enough data to fill last slice
            last_proxy_index = (((inner_stop - iter_start) // outer_step) * outer_step + iter_start)
            adjusted_iter_stop = last_proxy_index if (proxy_length - last_proxy_index) < slice_size else proxy_length

            # Handle Gap data
            if gap_data is not None:
                gap_fill = min(gap_offset + proxy_length, slice_size)
                fill_array_slices[axis] = Slice(gap_offset, gap_fill)
                fill_slices[axis] = Slice(0, gap_fill - gap_offset)
                self.proxies[stop_proxy].fill_slices_array(
                    data_array=gap_data,
                    array_slices=fill_array_slices,
                    slices=fill_slices,
                )
                if gap_fill == slice_size:
                    yield self.create_return_proxy_node(data=gap_data[::axis_step]) if proxy else gap_data
                    gap_data = None
                    gap_offset = 0

            # Iterate inner proxy
            if gap_offset == 0:
                iter_ = self.proxies[stop_proxy].islices(
                    slices=full_slices,
                    islice=Slice(iter_start, adjusted_iter_stop, istep),
                    dtype=dtype,
                    proxy=proxy,
                )
                for item in iter_:
                    yield item

    # Get proxy based on Index
    def get_proxy(self, index: int, proxy: bool | None = None) -> BaseProxyArray | np.ndarray:
        """Get a contained proxy/object or data from a contained proxy/object.

        Args:
            index: The proxy index to get the data from.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
        The requested proxy or data.
        """
        if (proxy is None and self.returns_proxy) or proxy:
            return self.proxies[index]
        else:
            return self.proxies[index].slices_array()

    def get_all_data(self, proxy: bool | None = None) -> Any:
        """Get all the contained proxies/objects as another proxy or data from the contained proxies/objects.

        Args:
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The new proxy or all the data as an array
        """
        if (proxy is None and self.returns_proxy) or proxy:
            return self
        else:
            return self.slices_array()


# Assign Cyclic Definitions
ProxyArray.default_return_proxy_type = ProxyArray
