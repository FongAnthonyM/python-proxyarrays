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
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Union

# Third-Party Packages #
from baseobjects.functions import singlekwargdispatch, MethodMultiplexer, CallableMultiplexObject
from baseobjects.typing import AnyCallable
from baseobjects.cachingtools import CachingObject
import numpy as np

# Local Packages #


# Definitions #
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

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # New Attributes #
        self._is_updating: bool = False
        self.returns_proxy: bool = True
        self.mode: str = "a"

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

    # Get proxy within by Index
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

    # Get Ranges of Data with Slices
    @abstractmethod
    def get_slices_array(self, slices: Iterable[slice | int | None] | None = None) -> np.ndarray:
        """Gets a range of data as an array.

        Args:
            slices: The ranges to get the data from.

        Returns:
            The requested range as an array.
        """
        pass

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
    def get_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int | None = None,
        proxy: bool | None = None,
    ) -> Union["BaseProxyArray", np.ndarray]:
        """Gets a range of data along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the data of the range.
            axis: The axis to get the data along.
            proxy: Determines if returned object is a proxy or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        pass
