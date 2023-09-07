"""directorytimeseries.py
A proxy for directory/file objects which contain time series data.
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
import pathlib
from typing import Any

# Third-Party Packages #

# Local Packages #
from ..timeseries import TimeSeriesProxy
from .basedirectorytimeseries import BaseDirectoryTimeSeries


# Definitions #
# Classes #
class DirectoryTimeSeriesProxy(TimeSeriesProxy, BaseDirectoryTimeSeries):
    """A proxy for directory/file objects which contain time series data.

    Class Attributes:
        default_return_proxy_type: The default type of proxy to return when returning a proxy.
        default_proxy_type: The default type proxy to create from the contents of the directory.

    Attributes:
        _path: The path of the directory to wrap.
        glob_condition: The glob string to use when using the glob method.
        proxy_type: The type of proxy to create from the contents of the directory.
        proxy_paths: The paths to the contained proxies.

    Args:
        path: The path for this proxy to wrap.
        proxies: An iterable holding proxies/objects to store in this proxy.
        mode: Determines if the contents of this proxy are editable or not.
        update: Determines if this proxy will start_timestamp updating or not.
        open_: Determines if the proxies will remain open after construction.
        build: Determines if the proxies will be constructed.
        **kwargs: The keyword arguments to create contained proxies.
        init: Determines if this object will construct.
    """

    default_return_proxy_type: type = TimeSeriesProxy
    default_proxy_type: type = None

    # Class Methods #
    @classmethod
    def validate_path(cls, path: str | pathlib.Path) -> bool:
        """Validates if the path can be used as Directory TimeProxy proxy.

        Args:
            path: The path to directory/file object that this proxy will wrap.

        Returns:
            If the path is usable.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        return path.is_dir()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        path: pathlib.Path | str | None = None,
        proxies: Iterable[BaseDirectoryTimeSeries] | None = None,
        mode: str = "r",
        update: bool = True,
        open_: bool = False,
        build: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self._path: pathlib.Path | None = None

        self.glob_condition: str = "*"

        self.proxy_type: type = self.default_proxy_type
        self.proxy_paths: dict[pathlib.Path, "DirectoryTimeSeriesProxy"] = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Object Construction #
        if init:
            self.construct(
                path=path,
                proxies=proxies,
                mode=mode,
                update=update,
                open_=open_,
                build=build,
                **kwargs,
            )

    @property
    def path(self) -> pathlib.Path:
        """The path this proxy wraps."""
        return self._path

    @path.setter
    def path(self, value: pathlib.Path | str) -> None:
        if isinstance(value, pathlib.Path) or value is None:
            self._path = value
        else:
            self._path = pathlib.Path(value)

    # Context Managers
    def __enter__(self) -> "BaseDirectoryTimeSeries":
        """The context enter which opens the directory.

        Returns:
            This object.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """The context exit which closes the file."""
        self.close()

    # Instance Methods
    # Constructors/Destructors
    def construct(
        self,
        path: pathlib.Path | str | None = None,
        proxies: Iterable[BaseDirectoryTimeSeries] | None = None,
        mode: str | None = None,
        update: bool = True,
        open_: bool = False,
        build: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            path: The path for this proxy to wrap.
            proxies: An iterable holding proxies/objects to store in this proxy.
            mode: Determines if the contents of this proxy are editable or not.
            update: Determines if this proxy will start_timestamp updating or not.
            open_: Determines if the proxies will remain open after construction.
            build: Determines if the proxies will be constructed.
            **kwargs: The keyword arguments to create contained proxies.
        """
        if path is not None:
            self.path = path

        super().construct(proxies=proxies, mode=mode, update=update)

        if build:
            if self.path.is_dir():
                self.construct_proxies(open_=open_, **kwargs)
            else:
                raise IOError(f"{self.path.as_posix()} does not exist.")

    def empty_copy(self, *args: Any, **kwargs: Any) -> "DirectoryTimeSeriesProxy":
        """Create a new copy of this object without data.

        Args:
            *args: The arguments for creating the new copy.
            **kwargs: The keyword arguments for creating the new copy.

        Returns:
            The new copy without proxies.
        """
        new_copy = super().empty_copy(*args, **kwargs)

        new_copy._path = self._path

        new_copy.glob_condition = self.glob_condition

        new_copy.proxy_type = self.proxy_type
        new_copy.proxy_paths = self.proxy_paths

        return new_copy

    def construct_proxies(self, open_=False, **kwargs) -> None:
        """Constructs the proxies for this object.

        Args:
            open_: Determines if the proxies will remain open after construction.
            **kwargs: The keyword arguments to create contained proxies.
        """
        for path in self.path.glob(self.glob_condition):
            if path not in self.proxy_paths:
                if self.proxy_creation_condition(path):
                    self.proxy_paths[path] = proxy = self.proxy_type(path, open_=open_, mode=self.mode, **kwargs)
                    self.proxies.append(proxy)
        self.proxies.sort(key=lambda proxy: proxy.start_timestamp)
        self.clear_caches()

    # Proxies
    def proxy_creation_condition(
        self,
        path: str | pathlib.Path,
        proxy: BaseDirectoryTimeSeries | None = None,
        **kwargs: Any,
    ) -> bool:
        """Determines if a proxy will be constructed.

        Args:
            path: The path to create a proxy from.
            proxy: A proxy to check if it should be created.
            **kwargs: Additional keyword arguments for deciding if the proxy will be created.

        Returns:
            If the path can be constructed.
        """
        return self.proxy_type.validate_path(path)

    def create_child(
        self,
        path: str | list[str],
        open_: bool = False,
        **kwargs
    ) -> None:
        """Creates a child proxy from the given child path.

        Args:
            path: The child path to create a proxy from.
            open_: Determines if the arrays will remain open after construction.
            **kwargs: The keyword arguments to create contained arrays.
        """
        path = path.split('/') if isinstance(path, str) else path.copy()

        child_path = self.path / path.pop(0)
        proxy = self.proxy_paths.get(child_path, None)
        if proxy is None:
            proxy = self.proxy_type(path=child_path, mode=self.mode, open_=open_, **kwargs)
            self.proxies.append(proxy)
            self.proxy_paths[child_path] = proxy

        if path:
            proxy.create_child(path=path, open_=open_, **kwargs)

        self.proxies.sort(key=lambda p: p.start_timestamp)
        self.clear_caches()

    def create_children(self, paths: list[dict], open_: bool = False, sort: bool = False, **kwargs) -> None:
        """Creates child arrays the given child paths.

        Args:
            paths: The child paths and keyword arguments to create arrays from.
            open_: Determines if the arrays will remain open after construction.
            sort: Determines if the arrays will be sorted after update.
            **kwargs: The keyword arguments to create contained arrays.
        """
        children_info = {}
        for path_kwargs in paths:
            path = path_kwargs["path"]
            path = path_kwargs["path"] = path.split('/') if isinstance(path, str) else path.copy()
            child_path = self.path / path.pop(0)
            info = children_info.get(child_path, None)
            if info is None:
                children_info[child_path] = {"kwargs": path_kwargs | {"path": child_path}, "children": [path_kwargs]}
            else:
                info["children"].append(path_kwargs)

        for child_path, info in children_info.items():
            proxy = self.proxy_paths.get(child_path, None)
            if proxy is None:
                self.proxy_paths[child_path] = proxy = self.proxy_type(
                    path=child_path,
                    mode=self.mode,
                    open_=open_,
                    build=False,
                )
                self.proxies.append(proxy)
            if info["children"]:
                proxy.create_children(paths=info["children"], open_=open_, sort=sort)

        if sort:
            self.proxies.sort(key=lambda p: p.start_timestamp)
            self.clear_caches()

    # Path and File System
    def open(self, mode: str | None = None, **kwargs: Any) -> BaseDirectoryTimeSeries:
        """Opens this directory proxy which opens all the contained proxies.

        Args:
            mode: The mode to open all the proxies in.
            **kwargs: The keyword arguments to open all the proxies with.

        Returns:
            This object.
        """
        if mode is None:
            mode = self.mode
        for proxy in self.proxies:
            proxy.open(mode, **kwargs)
        return self

    def close(self) -> None:
        """Closes this directory proxy which closes all the contained proxies."""
        for proxy in self.proxies:
            proxy.close()

    def require_path(self) -> None:
        """Creates this directory if it does not exist."""
        if not self.path.is_dir():
            self.path.mkdir()

    def require_proxies(self, **kwargs: Any) -> None:
        """Creates the contained proxies if they do not exist.

        Args:
            **kwargs: Keyword arguments for creating the files.
        """
        for proxy in self.proxies:
            try:
                proxy.require(**kwargs)
            except AttributeError:
                continue

    def require(self, **kwargs: Any) -> None:
        """Create this directory and all the contained proxies if they do not exist.

        Args:
            **kwargs: Keyword arguments for requiring the directory.
        """
        self.require_path()
        self.require_proxies(**kwargs)
