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

# Third-Party Packages #
import numpy as np
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


class TestTimeAxisProxy(ClassTest):

    def create_time_axis(self, sample_rate):
        time_axis = TimeAxisProxy()
        generator = BlankTimeAxis(start=0, sample_rate=1024.0, shape=(100000,), precise=True)
        segments = (
            (0, int(sample_rate * 10)),
            (0, int(sample_rate * 1)),
            (0, int(sample_rate * 0.5)),
            (0, int(sample_rate * 0.5)),
            (0, int(sample_rate * 10)),
            (100, int(sample_rate * 10)),
            (0, int(sample_rate * 10)),
            (0, int(sample_rate * 10.5)),
            (0, int(sample_rate * 10)),
        )
        end = 0

        for gap, length in segments:
            start = end + gap
            end = start + length
            dat = generator[start:end]
            time_axis.proxies.append(ContainerTimeAxis(data=dat, sample_rate=sample_rate, precise=True))

        return time_axis

    def test_islice(self):
        sample_rate = 1024.0
        time_axis = self.create_time_axis(sample_rate)
        start = int(3*sample_rate)
        step = int(sample_rate)

        n_slices = (len(time_axis) - start) // step
        step_shape = (step,)

        iter_ = time_axis.islices((slice(None, step),), slice(start, None))
        chunks = [(c.shape == step_shape) for c in iter_]
        assert all(chunks)
        assert len(chunks) == n_slices

    def test_nanostamp_islice_time(self):
        sample_rate = 1024.0
        time_axis = self.create_time_axis(sample_rate)

        iter_ = time_axis.nanostamp_islice_time(start=0.0, stop=50.0, step=1.0, approx=True, tails=True)
        chunks = [c for c in iter_]
        print(chunks)


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
