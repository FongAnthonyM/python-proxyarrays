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
import pickle
import pstats
from pstats import Stats, f8, func_std_string
import timeit
import time

# Third-Party Packages #
import numpy as np
import pytest

# Local Packages #
from src.proxyarrays import TimeSeriesProxy, ContainerTimeSeries, BlankTimeSeries, BlankTimeAxis, ContainerTimeAxis


# Definitions #
# Classes #
# Classes #
class StatsMicro(Stats):
    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = "  \n"
        for func in self.top_level:
            print(indent, func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end=" ", file=self.stream)
        print("in %.3f microseconds" % (self.total_tt * 1000000), file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            print('ncalls'.rjust(16), end='  ', file=self.stream)
            print('tottime'.rjust(12), end='  ', file=self.stream)
            print('percall'.rjust(12), end='  ', file=self.stream)
            print('cumtime'.rjust(12), end='  ', file=self.stream)
            print('percall'.rjust(12), end='  ', file=self.stream)
            print('filename:lineno(function)', file=self.stream)
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_line(self, func):  # hack: should print percentages
        cc, nc, tt, ct, callers = self.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + "/" + str(cc)
        print(c.rjust(16), end="  ", file=self.stream)
        print(f8(tt * 1000000).rjust(12), end="  ", file=self.stream)
        if nc == 0:
            print(" " * 12, end="  ", file=self.stream)
        else:
            print(f8(tt / nc * 1000000).rjust(12), end=" ", file=self.stream)
        print(f8(ct * 1000000).rjust(12), end="  ", file=self.stream)
        if cc == 0:
            print(" " * 12, end="  ", file=self.stream)
        else:
            print(f8(ct / cc * 1000000).rjust(12), end=" ", file=self.stream)
        print(func_std_string(func), file=self.stream)


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
        start = int(1*sample_rate)
        step = int(sample_rate)

        n_slices = (len(time_series) - start) // step
        step_shape = (step, channels)

        pr = cProfile.Profile()
        pr.enable()

        iter_ = time_series.islices(step, slice(start, None))
        chunks = [(c.shape == step_shape) for c in iter_]

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = StatsMicro(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_islice_single(self):
        sample_rate = 1024.0
        channels = 50
        time_series = self.create_time_series(sample_rate, channels)
        start = int(1*sample_rate)
        step = int(sample_rate)

        n_slices = (len(time_series) - start) // step
        step_shape = (step, channels)

        iter_ = time_series.islices(step, slice(start, None))

        pr = cProfile.Profile()
        pr.enable()

        chunk = next(iter_)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = StatsMicro(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_find_data_islice_time(self):
        sample_rate = 1024.0
        channels = 50
        time_series = self.create_time_series(sample_rate, channels)

        time_series.insert_missing(fill_method="full", fill_kwargs={"fill_value": 10})

        start = int(1 * sample_rate)
        step = int(sample_rate)
        # n_slices = (len(time_series) - start) // step
        step_shape = (step, channels)

        pr = cProfile.Profile()
        pr.enable()

        iter_ = time_series.find_data_islice_time(start=0.0, stop=40.0, step=1.0, approx=True, tails=True)
        chunks = [c for c in iter_]

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = StatsMicro(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
