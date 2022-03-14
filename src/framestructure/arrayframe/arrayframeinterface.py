""" arrayframeinterface.py
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
from typing import Any

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
from baseobjects.types_ import AnyCallable
from baseobjects.cachingtools import CachingObject

# Local Packages #


# Definitions #
# Classes #
# Todo: Create a cache base object and a file/edit mode base object to inherit from
class ArrayFrameInterface(CachingObject):
    """An interface which outlines the basis for an array frame.

    Attributes:
        _spawn_editable: The method to create an editable version of this frame.

    Args:
        init: Determines if this object will construct.
    """

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init: bool = True) -> None:
        # Parent Attributes #
        super().__init__()

        # New  Attributes #
        self._is_updating: bool = False

        self._spawn_editable: AnyCallable = self._default_spawn_editable()

        # Object Construction #
        if init:
            self.construct()

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

    def __getitem__(self, item: Any) -> Any:
        """Gets an item of this frame based on the input item.

        Args:
            item: The object to be used to get a specific item within this frame.

        Returns:
            An item within this frame.
        """
        return self.get_item(item)

    # Instance Methods #
    # Constructors/Destructors
    def editable_copy(self, *args: Any, **kwargs: Any) -> Any:
        """Creates an editable copy of this frame.

        Args:
            *args: The arguments for creating a new editable copy.
            **kwargs: The keyword arguments for creating a new editable copy.

        Returns:
            A editable copy of this object.
        """
        return self._spawn_editable(*args, **kwargs)

    # Caching
    @abstractmethod
    def clear_all_caches(self) -> None:
        """Clears the caches within this frame and any contained frames."""
        pass

    # Updating
    @abstractmethod
    def enable_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this frame and all contained frames/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        pass

    @abstractmethod
    def enable_last_updating(self, get_caches: bool = False) -> None:
        """Enables updating for this frame and the last contained frame/object.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        pass

    @abstractmethod
    def disable_updating(self, get_caches: bool = False) -> None:
        """disables updating for this frame and all contained frames/objects.

        Args:
            get_caches: Determines if get_caches will run before setting the caches.
        """
        pass

    # Getters
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

    # Editable Copy Methods
    def _default_spawn_editable(self, *args: Any, **kwargs: Any) -> Any:
        """The default method for creating a new editable version of this frame."""
        raise NotImplemented

    # Setters
    @singlekwargdispatchmethod("method")
    def set_spawn_editable(self, method: AnyCallable | str) -> None:
        """Sets the _spawn_editable method to another function or a method within this object can be given to select it.

        Args:
            method: The function or method name to set the _spawn_editable method to.
        """
        raise NotImplementedError(f"A {type(method)} cannot be used to set a {type(self)} _spawn_editable.")

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
        self._spawn_editable = getattr(self, method)

    # Shape
    @abstractmethod
    def validate_shape(self) -> bool:
        """Checks if this frame has a valid/continuous shape.

        Returns:
            If this frame has a valid/continuous shape.
        """
        pass

    @abstractmethod
    def reshape(self, shape: Iterable[int] | None = None, **kwargs: Any) -> None:
        """Changes the shape of the frame without changing its data."""
        pass

    # Get Frame within by Index
    @abstractmethod
    def get_from_index(self, indices: Iterator | Iterable | int, reverse: bool = False, frame: bool = True) -> Any:
        """Get an item recursively from within this frame using indices.

        Args:
            indices: The indices used to get an item within this frame.
            reverse: Determines if the indices should be used in the reverse order.
            frame: Determines if the

        Returns:
            The item recursively from within this frame.
        """
        pass
