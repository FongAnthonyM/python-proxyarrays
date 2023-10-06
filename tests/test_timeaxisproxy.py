#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_timeaxisproxy.py
Description:
"""

# Standard Libraries #
import cProfile
import datetime
import io
import os
import pathlib
import pstats
import timeit

import numpy as np

# Third-Party Packages #
import pytest

# Local Packages #
from src.proxyarrays import TimeAxisProxy, ContainerTimeAxis, BlankTimeAxis


# Definitions #
# Classes #
class ClassTest:
    """Default class tests that all classes should pass."""

    class_ = None
    timeit_runs = 2
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestXLTEKStudy(ClassTest):

    def test_nanostamp_islice_time(self):
        sample_rate = 1024.0
        time_axis = TimeAxisProxy()
        for s in range(0, 100, 10):
            dat = BlankTimeAxis(start=s, sample_rate=1024.0, shape=(10000,), precise=True)[...]
            time_axis.proxies.append(ContainerTimeAxis(data=dat, sample_rate=sample_rate, precise=True))

        print()


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
