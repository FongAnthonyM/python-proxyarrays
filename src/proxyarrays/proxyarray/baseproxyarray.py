"""baseproxyarray.py
A base which outlines the basis for a proxy array.
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
from collections.abc import Callable, Iterable, Iterator, Generator
from typing import Any, Union

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch, MethodMultiplexer, CallableMultiplexObject
from baseobjects.typing import AnyCallable
from baseobjects.cachingtools import CachingObject
import numpy as np

# Local Packages #


# Definitions #
# Typing Aliases #
Slice = slice


# Classes #
# Todo: Create a file/edit mode base object to inherit from
class BaseProxyArray(CallableMultiplexObject, CachingObject):
    """A base which outlines the basis for a proxy array.

    Attributes:
        _is_updating: Determines if this proxy is updating or not.
        spawn_editable: The method to create an editable version of this proxy.
        returns_proxy: Determines if methods will return proxies or numpy arrays.
        mode: Determines if this proxy is editable or read only.
        *args: Arguments for inheritance.
        **kwargs: Keyword arguments for inheritance.
    """
    default_return_proxy_node: type | None = None
    default_return_proxy_leaf: type | None = None
    default_return_proxy_type: type | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # New Attributes #
        self._is_updating: bool = False
        self.returns_proxy: bool = True
        self.mode: str = "a"

        self.return_proxy_node: type = self.default_return_proxy_node
        self.return_proxy_leaf: type = self.default_return_proxy_leaf
        self.return_proxy_type: type = self.default_return_proxy_type
        self.spawn_editable: MethodMultiplexer = MethodMultiplexer(instance=self, select="_default_spawn_editable")

        # Parent Attributes #
        super().__init__(*args, **kwargs)

    # Container Methods
    def __len__(self) -> int:
        """Gets this object's length.

        Returns:
            The number of nodes in this object.
        """
        return self.get_length()

    def __getitem__(self, item: Any) -> Any:
        """Gets an item of this proxy based on the input item.

        Args:
            item: The object to be used to get a specific item within this proxy.

        Returns:
            An item within this proxy.
        """
        return self.get_item(item)

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
    # Constructors/Destructors
    def create_proxy(self, type_: type["BaseProxyArray"], *args: Any, **kwargs: Any) -> "BaseProxyArray":
        """Creates a new proxy array with the same attributes as this proxy.

        Args:
            type_: The type of proxy array to create.
            *args: The arguments for creating the proxy array.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        return type_(**({"axis": self.axis, "mode": self.mode, "update": self._is_updating} | kwargs))

    def create_return_proxy_node(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a new proxy array with the same attributes as this proxy, default type is the return proxy node.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        return self.create_proxy(type_=self.return_proxy_node if type_ is None else type_, **kwargs)

    def create_return_proxy_leaf(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a new proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        return self.create_proxy(type_=self.return_proxy_leaf if type_ is None else type_, **kwargs)

    def create_return_proxy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a new proxy array with the same attributes as this proxy, default type is the return proxy.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The new proxy array.
        """
        return self.create_proxy(type_=self.return_proxy_type if type_ is None else type_, **kwargs)

    def empty_copy(self, *args: Any, **kwargs: Any) -> "BaseProxyArray":
        """Create a new copy of this object without proxies.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        return self.create_proxy(type_=self.__class__, **kwargs)

    @abstractmethod
    def proxy_leaf_copy(self, type_: type["BaseProxyArray"] | None = None, **kwargs: Any) -> "BaseProxyArray":
        """Creates a copy proxy array with the same attributes as this proxy, default type is the return proxy leaf.

        Args:
            type_: The type of proxy array to create.
            **kwargs: The keyword arguments for creating the proxy array.

        Returns:
            The copy of this proxy array.
        """
        return self.create_proxy(type_=self.return_proxy_leaf if type_ is None else type_, **kwargs)

    # Editable Copy Methods
    def editable_copy(self, *args: Any, **kwargs: Any) -> Any:
        """Creates an editable copy of this proxy.

        Args:
            *args: The arguments for creating a new editable copy.
            **kwargs: The keyword arguments for creating a new editable copy.

        Returns:
            A editable copy of this object.
        """
        return self._spawn_editable(*args, **kwargs)

    def _default_spawn_editable(self, *args: Any, **kwargs: Any) -> Any:
        """The default method for creating an editable version of this proxy.

        Args:
            *args: Arguments to help create the new editable proxy.
            **kwargs: Keyword arguments to help create the new editable proxy.

        Returns:
            An editable version of this proxy.
        """
        new_proxy = self.copy()
        new_proxy.mode = "a"
        return new_proxy

    # Caching
    def clear_all_caches(self) -> None:
        """Clears the caches within this proxy and any contained proxies."""
        self.clear_caches()

    # Updating
    def enable_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this proxy and all contained proxies/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = True

    def enable_last_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this proxy and the last contained proxy/object.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = True

    def disable_updating(self, get_caches: bool = False) -> None:
        """Disables updating for this proxy and all contained proxies/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = False

    # Getters
    def get_any_updating(self) -> bool:
        """Checks if any contained proxies/objects are updating.

        Returns:
            If any contained proxies/objects are updating.
        """
        return self._is_updating

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

    def shape_from_slices(self, slices: Iterable[Slice | int | None] | None = None) -> tuple[int, ...]:
        """Creates a shape from given slices for this object.

        Args:
            slices: The slices to generate the data from.

        Returns:
            The requested shape.
        """
        if slices is None:
            shape = self.shape
        else:
            shape = []
            for index, slice_ in enumerate(slices):
                if isinstance(slice_, int):
                    shape.append(1)
                else:
                    start = 0 if slice_.start is None else slice_.start
                    stop = self.shape[index] if slice_.stop is None else slice_.stop
                    step = 1 if slice_.step is None else slice_.step

                    if start < 0:
                        start = self.shape[index] + start

                    if stop < 0:
                        stop = self.shape[index] + stop

                    if start < 0 or start > self.shape[index] or stop < 0 or stop > self.shape[index]:
                        raise IndexError("index is out of range")

                    size = stop - start
                    if size < 0:
                        raise IndexError("start index is greater than stop")
                    shape.append(size // step)

        return tuple(shape)

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

    # Data Slicing
    @singlekwargdispatch("start")
    def generate_full_slices(
        self,
        start: int | Slice | None = None,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        axis: int | None = None,
    ) -> tuple[Slice, ...]:
        """Generates a tuple of slices based on the shape of this proxy array and sets a given slice based on axis.

            Args:
                start: Either first index of the slice or the slice itself to set.
                stop: The length of the slice to set.
                step: The interval of the slice to set.
                slice_: The slice to set.
                axis: The axis set the slice.

            Returns:
                All the slices of this proxy array.
            """
        if start is None and (slice_ is None or isinstance(slice_, Slice)):
            if slice_ is None:
                return self._generate_full_slices_slice(start=slice(None, stop, step), axis=axis)
            else:
                return self._generate_full_slices_slice(start=slice_, axis=axis)
        else:
            raise TypeError(f"A {type(start)} cannot be used to slice a {self.__class__}.")

    @generate_full_slices.register(Slice)
    def _generate_full_slices_slice(
        self,
        start: Slice,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        axis: int | None = None,
    ) -> tuple[Slice, ...]:
        """Generates a tuple of slices based on the shape of this proxy array and sets a given slice based on axis.

        Args:
            start: The slice to set.
            stop: The length of the slice to set.
            step: The interval of the slice to set.
            slice_: The slice to set.
            axis: The axis set the slice.

        Returns:
            All the slices of this proxy array.
        """
        if axis is None:
            axis = self.axis

        slices = [Slice(None)] * self.ndim
        slices[axis] = start
        return tuple(slices)

    @generate_full_slices.register(int)
    def _generate_full_slices_int(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        axis: int | None = None,
    ) -> tuple[Slice, ...]:
        """Generates a tuple of slices based on the shape of this proxy array and sets a given slice based on axis.

        Args:
            start: The first index of the slice to set.
            stop: The length of the slice to set.
            step: The interval of the slice to set.
            slice_: The slice to set.
            axis: The axis set the slice.

        Returns:
            All the slices of this proxy array.
        """
        if axis is None:
            axis = self.axis

        slices = [Slice(None)] * self.ndim
        slices[axis] = Slice(start, stop, step)
        return tuple(slices)

    @abstractmethod
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
        pass

    @abstractmethod
    def slices_array(
        self, slices: Iterable[Slice | int | None] | None = None,
        dtype: Any = None,
    ) -> np.ndarray:
        """Get data as an array using slices to determine the data slice.

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

    def slices(
        self,
        slices: Iterable[Slice | int | None] | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Union["BaseProxyArray", np.ndarray]:
        """Get data using slices to determine the data slice.

        Args:
            slices: The slices along the axes to get data from.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested slice as an array or proxy.
        """
        if (proxy is None and self.returns_proxy) or proxy:
            return self.slices_proxy(slices=slices)
        else:
            return self.slices_array(slices=slices, dtype=dtype)

    def slice(
        self,
        start: int | Slice | None = None,
        stop: int | None = None,
        step: int | None = None,
        slice_: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Union["BaseProxyArray", np.ndarray]:
        """Gets a slice of data along an axis.

        Args:
            start: Either the first super index of the slice or the slice itself to get the data from.
            stop: The length of the slice to get.
            step: The interval to get the data of the slice.
            slice_: The slice to get the data from.
            axis: The axis to get the data along.
            dtype: The dtype of array to return.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested slice as an array or proxy.
        """
        return self.slices(
            slices=self.generate_full_slices(start=start, stop=stop, step=step, axis=axis),
            dtype=dtype,
            proxy=proxy,
        )

    @abstractmethod
    def islices(
        self,
        slices: Iterable[Slice | int | None] | None = None,
        islice: Slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Generator[Union["BaseProxyArray", np.ndarray], None, None]:
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
