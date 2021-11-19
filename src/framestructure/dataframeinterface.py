#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dataframeinterface.py
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
from abc import abstractmethod

# Third-Party Packages #
from baseobjects.cachingtools import CachingObject

# Local Packages #


# Definitions #
# Classes #
# Todo: Create a cache base object and a file/edit mode base object to inherit from
class DataFrameInterface(CachingObject):
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init=True):
        super().__init__()
        self.editable_method = self.default_editable_method

        if init:
            self.construct()

    # Container Methods
    def __len__(self):
        try:
            return self.get_length.caching_call()
        except AttributeError:
            return self.get_length()

    def __getitem__(self, item):
        return self.get_item(item)

    # Instance Methods #
    # Constructors/Destructors
    def editable_copy(self, **kwargs):
        return self.editable_method(**kwargs)

    # Caching
    @abstractmethod
    def clear_all_caches(self):
        pass

    # Getters
    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_item(self, item):
        pass

    # Editable Copy Methods
    def default_editable_method(self, **kwargs):
        raise NotImplemented

    # Setters
    def set_editable_method(self, obj):
        self.editable_method = obj

    # Shape
    @abstractmethod
    def validate_shape(self):
        pass

    @abstractmethod
    def change_size(self, shape=None, **kwargs):
        pass

    # Get Frame within by Index
    @abstractmethod
    def get_index(self, indices, reverse=False, frame=True):
        pass

