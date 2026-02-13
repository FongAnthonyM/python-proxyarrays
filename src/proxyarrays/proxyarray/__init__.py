"""__init__.py
Base Array proxies.
"""

# Header #
__package_name__ = "proxyarrays"

__author__ = "Anthony Fong"
__credits__ = ["Anthony Fong"]
__copyright__ = "Copyright 2021, Anthony Fong"
__license__ = "MIT"

__version__ = "0.7.0"

# Imports
# Local Packages #
from .baseproxyarray import BaseProxyArray
from .proxyarray import ProxyArray
from .containerproxyarray import ContainerProxyArray
from .blankproxyarray import BlankProxyArray
