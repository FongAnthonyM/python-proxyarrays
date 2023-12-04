"""__init__.py
Numpy array like objects that are proxies for remote or virtual arrays.
"""
# Package Header #
from .header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Local Packages #
from .proxyarray import *
from .timeproxy import *
from .timeaxis import *
from .timeseries import *
from .directorytimeseries import *
from .filetimeseries import *
