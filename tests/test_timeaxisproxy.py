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
        generator = BlankTimeAxis(start=0, sample_rate=1024.0, shape=(100000,), precise=True)
        lengths = (
            int(sample_rate * 10),
            int(sample_rate * 1),
            int(sample_rate * 0.5),
            int(sample_rate * 0.5),
            int(sample_rate * 10),
            int(sample_rate * 10),
        )
        gaps = (
            0,
            0,
            0,
            0,
            0,
            10000,
        )
        end = 0

        for length, gap in zip(lengths, gaps):
            start = end + gap
            end = start + length
            dat = generator[start:end]
            time_axis.proxies.append(ContainerTimeAxis(data=dat, sample_rate=sample_rate, precise=True))

        iter_ = time_axis.nanostamps_islice_time(start=0.0, stop=50.0, step=1.0, approx=True, tails=True)
        chunks = [c for c in iter_]


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
