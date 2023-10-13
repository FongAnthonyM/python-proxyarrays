#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_timeaseriesproxy.py
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
from src.proxyarrays import TimeSeriesProxy, ContainerTimeSeries, BlankTimeAxis, ContainerTimeAxis


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


class TestTimeSeriesProxy(ClassTest):

    def create_time_series(self, sample_rate, channels):
        time_series = TimeSeriesProxy()
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
            times = generator[start:end]
            time_axis = ContainerTimeAxis(data=times, sample_rate=sample_rate, precise=True)
            data = np.random.rand(len(times), channels) - 0.5
            time_series.proxies.append(ContainerTimeSeries(data=data, time_axis=time_axis))

        return time_series

    def test_find_slice_time(self):
        sample_rate = 1024.0
        channels = 50
        time_series = self.create_time_series(sample_rate, channels)

        data = time_series.find_data_slice(start=1.0, stop=32.0, approx=True, tails=True)
        print(data)

    def test_islice(self):
        sample_rate = 1024.0
        channels = 50
        time_series = self.create_time_series(sample_rate, channels)
        start = int(3*sample_rate)
        step = int(sample_rate)

        n_slices = (len(time_series) - start) // step
        step_shape = (step,)

        iter_ = time_series.islices((slice(None, step),), slice(start, None))
        chunks = [(c.shape == step_shape) for c in iter_]
        assert all(chunks)
        assert len(chunks) == n_slices

    def test_nanostamp_islice_time(self):
        sample_rate = 1024.0
        time_series = self.create_time_series(sample_rate)

        iter_ = time_series.nanostamps_islice_time(start=0.0, stop=50.0, step=1.0, approx=True, tails=True)
        chunks = [c for c in iter_]
        print(chunks)


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
