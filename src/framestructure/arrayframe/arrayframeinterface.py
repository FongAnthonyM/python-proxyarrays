"""arrayframeinterface.py
An interface which outlines the basis for an array frame.
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
from baseobjects.functions import singlekwargdispatch
from baseobjects.typing import AnyCallable
from baseobjects import BaseObject
from baseobjects.cachingtools import CachingObject
import numpy as np

# Local Packages #


# Definitions #
# Classes #
# Todo: Create a file/edit mode base object to inherit from
class ArrayFrameInterface(CachingObject):
    """An interface which outlines the basis for an array frame.

    Attributes:
        _is_updating: Determines if this frame is updating or not.
        _spawn_editable: The method to create an editable version of this frame.
        returns_frame: Determines if methods will return frames or numpy arrays.
        mode: Determines if this frame is editable or read only.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Parent Attributes #
        super().__init__(*args, **kwargs)

        # New  Attributes #
        self._is_updating: bool = False

        self._spawn_editable: AnyCallable = self._default_spawn_editable.__func__

        self.returns_frame: bool = False

        self.mode: str = "a"

    @property
    def spawn_editable(self) -> AnyCallable:
        """The method used to create an editable version of this frame."""
        return self._spawn_editable.__get__(self, self.__class__)

    # Container Methods
    def __len__(self) -> int:
        """Gets this object's length.

        Returns:
            The number of nodes in this object.
        """
        return self.get_length()

    def __getitem__(self, item: Any) -> Any:
        """Gets an item of this frame based on the input item.

        Args:
            item: The object to be used to get a specific item within this frame.

        Returns:
            An item within this frame.
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
        """Creates an editable copy of this frame.

        Args:
            *args: The arguments for creating a new editable copy.
            **kwargs: The keyword arguments for creating a new editable copy.

        Returns:
            A editable copy of this object.
        """
        return self._spawn_editable(*args, **kwargs)

    def _default_spawn_editable(self, *args: Any, **kwargs: Any) -> Any:
        """The default method for creating an editable version of this frame.

        Args:
            *args: Arguments to help create the new editable frame.
            **kwargs: Keyword arguments to help create the new editable frame.

        Returns:
            An editable version of this frame.
        """
        new_frame = self.copy()
        new_frame.mode = "a"
        return new_frame

    @singlekwargdispatch("method")
    def set_spawn_editable(self, method: AnyCallable | str) -> None:
        """Sets the _spawn_editable method to another function or a method within this object can be given to select it.

        Args:
            method: The function or method name to set the _spawn_editable method to.
        """
        raise TypeError(f"A {type(method)} cannot be used to set a {type(self)} _spawn_editable.")

    @set_spawn_editable.register(Callable)
    def _(self, method: AnyCallable) -> None:
        """Sets the _spawn_editable method to another function or a method within this object can be given to select it.

        Args:
            method: The function to set the _spawn_editable method to.
        """
        self._spawn_editable = method

    @set_spawn_editable.register
    def _(self, method: str) -> None:
        """Sets the _spawn_editable method to another function or a method within this object can be given to select it.

        Args:
            method: The method name to set the _spawn_editable method to.
        """
        self._spawn_editable = getattr(self, method).__func__

    # Caching
    def clear_all_caches(self) -> None:
        """Clears the caches within this frame and any contained frames."""
        self.clear_caches()

    # Updating
    def enable_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this frame and all contained frames/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = True

    def enable_last_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this frame and the last contained frame/object.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = True

    def disable_updating(self, get_caches: bool = False) -> None:
        """Disables updating for this frame and all contained frames/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        self._is_updating = False

    # Getters
    def get_any_updating(self) -> bool:
        """Checks if any contained frames/objects are updating.

        Returns:
            If any contained frames/objects are updating.
        """
        return self._is_updating

    @abstractmethod
    def get_shape(self) -> tuple[int]:
        """Get the shape of this frame from the contained frames/objects.

        Returns:
            The shape of this frame.
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """Gets the length of this frame.

        Returns:
            The length of this frame.
        """
        pass

    @abstractmethod
    def get_item(self, item: Any) -> Any:
        """Gets an item from within this frame based on an input item.

        Args:
            item: The object to be used to get a specific item within this frame.

        Returns:
            An item within this frame.
        """
        pass

    # Shape
    @abstractmethod
    def validate_shape(self) -> bool:
        """Checks if this frame has a valid/continuous shape.

        Returns:
            If this frame has a valid/continuous shape.
        """
        pass

    @abstractmethod
    def resize(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the frame without changing its data."""
        pass

    # Get Frame within by Index
    @abstractmethod
    def get_from_index(
        self,
        indices: Iterator | Iterable | int,
        reverse: bool = False,
        frame: bool = True,
    ) -> Any:
        """Get an item recursively from within this frame using indices.

        Args:
            indices: The indices used to get an item within this frame.
            reverse: Determines if the indices should be used in the reverse order.
            frame: Determines if the

        Returns:
            The item recursively from within this frame.
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
        """Fills a given array with values from the contained frames/objects.

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
        frame: bool | None = None,
    ) -> Union["ArrayFrameInterface", np.ndarray]:
        """Gets a range of data along an axis.

        Args:
            start: The first super index of the range to get.
            stop: The length of the range to get.
            step: The interval to get the data of the range.
            axis: The axis to get the data along.
            frame: Determines if returned object is a Frame or an array, default is this object's setting.

        Returns:
            The requested range.
        """
        pass
