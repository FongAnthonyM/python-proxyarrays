#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" __init__.py
framestructure provides several classes for creating structures of dataframes.
"""
# Package Header #
from .__header__ import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Local Packages #
from .dataframeinterface import DataFrameInterface
from .blankdataframe import BlankDataFrame
from .dataframe import DataFrame
from .timeseriesframe import *
from .directorytimeframe import *
from .filetimeframe import *
