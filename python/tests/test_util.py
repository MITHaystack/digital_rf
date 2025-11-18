# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Tests for the digital_rf.digital_rf_hdf5 module."""

from __future__ import annotations

import datetime
import fractions

import numpy as np

import digital_rf


def reference_time_to_sample_ceil(timedelta, sample_rate):
    if isinstance(timedelta, tuple):
        t_sec, t_psec = timedelta
    elif isinstance(timedelta, np.timedelta64):
        onesec = np.timedelta64(1, "s")
        t_sec = timedelta // onesec
        t_psec = (timedelta % onesec) // np.timedelta64(1, "ps")
    elif isinstance(timedelta, datetime.timedelta):
        t_sec = int(timedelta.total_seconds())
        t_psec = 1000000 * timedelta.microseconds
    else:
        t_sec = int(timedelta)
        t_psec = int(np.round((timedelta % 1) * 1e12))
    # ensure that sample_rate is a fractions.Fraction
    if not isinstance(sample_rate, fractions.Fraction):
        sample_rate = digital_rf.util.get_samplerate_frac(sample_rate)
    # calculate rational values for the second and picosecond parts
    s_frac = t_sec * sample_rate + t_psec * sample_rate / 10**12
    # get an integer value through ceiling rounding
    return int(s_frac) + ((s_frac % 1) != 0)


def reference_sample_to_time_floor(nsamples, sample_rate):
    nsamples = int(nsamples)
    # ensure that sample_rate is a fractions.Fraction
    if not isinstance(sample_rate, fractions.Fraction):
        sample_rate = digital_rf.util.get_samplerate_frac(sample_rate)

    # get the timedelta as a Fraction
    t_frac = nsamples / sample_rate

    seconds = int(t_frac)
    picoseconds = int((t_frac % 1) * 10**12)

    return (seconds, picoseconds)


###############################################################################
#  tests  #####################################################################
###############################################################################


def test_sample_timestamp_conversion(
    sample_rate_numerator, sample_rate_denominator, sample_rate, start_global_index
):
    # test that sample index round trips through get_sample_ceil(get_timestamp_floor())
    for global_index in range(start_global_index, start_global_index + 100):
        ref_second, ref_picosecond = reference_sample_to_time_floor(
            global_index, sample_rate
        )

        second, picosecond = digital_rf._py_rf_write_hdf5.get_timestamp_floor(
            global_index, sample_rate_numerator, sample_rate_denominator
        )
        assert second == ref_second
        assert picosecond == ref_picosecond
        second2, picosecond2 = digital_rf.util.sample_to_time_floor(
            global_index, sample_rate
        )
        assert second2 == ref_second
        assert picosecond2 == ref_picosecond

        ref_rt_global_index = reference_time_to_sample_ceil(
            (ref_second, ref_picosecond), sample_rate
        )
        assert ref_rt_global_index == global_index

        rt_global_index = digital_rf._py_rf_write_hdf5.get_sample_ceil(
            second, picosecond, sample_rate_numerator, sample_rate_denominator
        )
        assert rt_global_index == ref_rt_global_index
        rt_global_index2 = digital_rf.util.time_to_sample_ceil(
            (second, picosecond), sample_rate
        )
        assert rt_global_index2 == ref_rt_global_index

    # test passing array to get_sample_ceil(get_timestamp_floor())
    global_idxs = np.arange(start_global_index, start_global_index + 100, dtype="int64")
    seconds, picoseconds = digital_rf.util.sample_to_time_floor(
        global_idxs, sample_rate
    )
    rt_global_idxs = digital_rf.util.time_to_sample_ceil(
        (seconds, picoseconds), sample_rate
    )
    np.testing.assert_equal(rt_global_idxs, global_idxs)
