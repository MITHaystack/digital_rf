# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
from __future__ import annotations

import datetime
import itertools
import os

import numpy as np
import pytest

import digital_rf


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "firstonly(fixturename1, fixturename2, ...): Generate a test only for the first parameter of the listed fixtures.",
    )


def pytest_collection_modifyitems(items):
    selected_items = []

    for item in items:
        for firstonly in item.iter_markers("firstonly"):
            for param in firstonly.args:
                # check if specified param is not first and skip
                idx = item.callspec.indices.get(param, None)
                if idx is not None and idx > 0:
                    break
            else:
                # if item is not skipped on the basis of this set of params,
                # move on to next mark in loop
                continue
            # if we broke out of the previous for loop and are skipping the
            # item, we have to break out of this loop too
            break
        else:
            selected_items.append(item)

    items[:] = selected_items


###############################################################################
#  constant fixtures  #########################################################
###############################################################################


@pytest.fixture(scope="session")
def start_timestamp_tuple():
    start_dt = datetime.datetime(
        2014, 3, 9, 12, 30, 30, 0, tzinfo=datetime.timezone.utc
    )
    timedelta = start_dt - digital_rf.util.epoch
    seconds = int(timedelta.total_seconds())
    picoseconds = timedelta.microseconds * 1000000
    return (seconds, picoseconds)


###############################################################################
#  parametrized fixtures  #####################################################
###############################################################################


_dtype_strs = [
    "s8",
    "sc8",
    "u8",
    "uc8",
    "s16",
    "sc16",
    "u16",
    "uc16",
    "s32",
    "sc32",
    "u32",
    "uc32",
    "s64",
    "sc64",
    "u64",
    "uc64",
    "f32",
    "fc32",
    "f64",
    "fc64",
]
_num_subchannels = [(1, "x1"), (8, "x8")]
_df_params = list(itertools.product(_dtype_strs, (sc[0] for sc in _num_subchannels)))
_df_ids = list(
    "".join(p)
    for p in itertools.product(_dtype_strs, (sc[1] for sc in _num_subchannels))
)


@pytest.fixture(scope="session", params=_df_params, ids=_df_ids)
def data_params(request):
    dtypes = dict(
        s8=np.dtype("i1"),
        sc8=np.dtype([("r", "i1"), ("i", "i1")]),
        u8=np.dtype("u1"),
        uc8=np.dtype([("r", "u1"), ("i", "u1")]),
        s16=np.dtype("i2"),
        sc16=np.dtype([("r", "i2"), ("i", "i2")]),
        u16=np.dtype("u2"),
        uc16=np.dtype([("r", "u2"), ("i", "u2")]),
        s32=np.dtype("i4"),
        sc32=np.dtype([("r", "i4"), ("i", "i4")]),
        u32=np.dtype("u4"),
        uc32=np.dtype([("r", "u4"), ("i", "u4")]),
        s64=np.dtype("i8"),
        sc64=np.dtype([("r", "i8"), ("i", "i8")]),
        u64=np.dtype("u8"),
        uc64=np.dtype([("r", "u8"), ("i", "u8")]),
        f32=np.dtype("f4"),
        fc32=np.dtype("c8"),
        f64=np.dtype("f8"),
        fc64=np.dtype("c16"),
    )
    dtype_str = request.param[0]
    dtype = dtypes[dtype_str]
    is_complex = dtype.names is not None or np.issubdtype(dtype, np.complexfloating)
    num_subchannels = request.param[1]
    return dict(
        dtype_str=dtype_str,
        dtype=dtype,
        is_complex=is_complex,
        num_subchannels=num_subchannels,
    )


@pytest.fixture(scope="session", params=[True, False], ids=["continuous", "gapped"])
def file_params(request):
    return dict(is_continuous=request.param)


@pytest.fixture(
    scope="session",
    params=[
        # compression_level, checksum
        (0, False),
        (9, True),
    ],
    ids=["nozip,nochecksum", "zip,checksum"],
)
def hdf_filter_params(request):
    return dict(compression_level=request.param[0], checksum=request.param[1])


@pytest.fixture(
    scope="session",
    params=[
        # sample rates must be set so that start_timestamp_tuple is an exact sample time
        # srnum, srden, sdcsec, fcms
        (200, 3, 2, 400)
    ],
    ids=["200/3hz,2s,400ms"],
)
def sample_params(request):
    return dict(
        sample_rate_numerator=request.param[0],
        sample_rate_denominator=request.param[1],
        subdir_cadence_secs=request.param[2],
        file_cadence_millisecs=request.param[3],
    )


###############################################################################
#  remaining fixtures  ########################################################
###############################################################################


@pytest.fixture(scope="session")
def param_dict(
    data_params, file_params, hdf_filter_params, sample_params, start_global_index
):
    params = dict(start_global_index=start_global_index)
    params.update(data_params)
    params.update(file_params)
    params.update(hdf_filter_params)
    params.update(sample_params)
    return params


@pytest.fixture(scope="session")
def param_str(param_dict):
    s = (
        "{dtype_str}_x{num_subchannels}_n{sample_rate_numerator}"
        "_d{sample_rate_denominator}_{subdir_cadence_secs}s"
        "_{file_cadence_millisecs}ms_c{is_continuous:b}"
    ).format(**param_dict)
    return s


@pytest.fixture(scope="session")
def dtype_str(data_params):
    return data_params["dtype_str"]


@pytest.fixture(scope="session")
def dtype(data_params):
    return data_params["dtype"]


@pytest.fixture(scope="session")
def is_complex(data_params):
    return data_params["is_complex"]


@pytest.fixture(scope="session")
def num_subchannels(data_params):
    return data_params["num_subchannels"]


@pytest.fixture(scope="session")
def is_continuous(file_params):
    return file_params["is_continuous"]


@pytest.fixture(scope="session")
def compression_level(hdf_filter_params):
    return hdf_filter_params["compression_level"]


@pytest.fixture(scope="session")
def checksum(hdf_filter_params):
    return hdf_filter_params["checksum"]


@pytest.fixture(scope="session")
def sample_rate_numerator(sample_params):
    return sample_params["sample_rate_numerator"]


@pytest.fixture(scope="session")
def sample_rate_denominator(sample_params):
    return sample_params["sample_rate_denominator"]


@pytest.fixture(scope="session")
def subdir_cadence_secs(sample_params):
    return sample_params["subdir_cadence_secs"]


@pytest.fixture(scope="session")
def file_cadence_millisecs(sample_params):
    return sample_params["file_cadence_millisecs"]


@pytest.fixture(scope="session")
def sample_rate(sample_rate_numerator, sample_rate_denominator):
    return digital_rf.util.get_samplerate_frac(
        sample_rate_numerator, sample_rate_denominator
    )


@pytest.fixture(scope="session")
def start_global_index(sample_rate, start_timestamp_tuple):
    return digital_rf.util.time_to_sample_ceil(start_timestamp_tuple, sample_rate)


@pytest.fixture(scope="session")
def start_datetime(start_timestamp_tuple):
    seconds, picoseconds = start_timestamp_tuple
    start_dt = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    start_dt += datetime.timedelta(microseconds=picoseconds // 1000000)
    return start_dt


@pytest.fixture(scope="session")
def end_global_index(
    file_cadence_millisecs,
    sample_rate,
    start_global_index,
    subdir_cadence_secs,
):
    # want data to span at least two subdirs to test creation + naming
    # also needs to span at least 8 files to accommodate write blocks (below)
    nsamples_subdirs = digital_rf.util.time_to_sample_ceil(
        1.5 * subdir_cadence_secs, sample_rate
    )
    nsamples_files = digital_rf.util.time_to_sample_ceil(
        8 * file_cadence_millisecs / 1000, sample_rate
    )
    nsamples = max(nsamples_subdirs, nsamples_files)
    return start_global_index + nsamples - 1


@pytest.fixture(scope="session")
def bounds(
    end_global_index,
    file_cadence_millisecs,
    is_continuous,
    sample_rate_numerator,
    sample_rate_denominator,
    start_global_index,
):
    if is_continuous:
        srn = sample_rate_numerator
        srd = sample_rate_denominator
        fcms = file_cadence_millisecs

        # bounds are at file boundaries
        # milliseconds of file with start_global_index
        sms = (((start_global_index * srd * 1000) // srn) // fcms) * fcms
        # sample index at start of file (ceil b/c start at first whole sample)
        ss = ((sms * srn) + (srd * 1000) - 1) // (srd * 1000)
        # milliseconds of file after end_global_index
        ems = ((((end_global_index * srd * 1000) // srn) // fcms) + 1) * fcms
        # sample index at start of file (ceil b/c start at first whole sample)
        es = ((ems * srn) + (srd * 1000) - 1) // (srd * 1000)
        return (ss, es - 1)
    return (start_global_index, end_global_index)


@pytest.fixture(scope="session")
def data_block_slices(
    bounds,
    end_global_index,
    file_cadence_millisecs,
    sample_rate,
    start_global_index,
):
    # blocks = [(start_sample, stop_sample)]
    blocks = []
    samples_per_file = digital_rf.util.time_to_sample_ceil(
        file_cadence_millisecs / 1000, sample_rate
    )

    # first block stops in middle of second file
    sstart = start_global_index
    sstop = bounds[0] + int(1.5 * samples_per_file)
    blocks.append((sstart, sstop))

    # second block continues where first ended, stops in middle of third file
    sstart = sstop
    sstop = bounds[0] + int(2.5 * samples_per_file)
    blocks.append((sstart, sstop))

    # first gap, no skipped file
    # third block starts in middle of 4th file, ends in middle of 5th
    sstart = bounds[0] + int(3.5 * samples_per_file)
    sstop = bounds[0] + int(4.5 * samples_per_file)
    blocks.append((sstart, sstop))

    # second gap, skipping one file completely
    # fourth block starts in middle of 7th file, ends later in 7th file
    sstart = bounds[0] + int(6.5 * samples_per_file)
    sstop = bounds[0] + int(6.9 * samples_per_file)
    blocks.append((sstart, sstop))

    # fifth and final block continues just after previous write and goes to end
    sstart = sstop + 2
    sstop = end_global_index + 1
    blocks.append((sstart, sstop))

    return blocks


def generate_rf_data(shape, dtype, seed):
    np.random.seed(seed % 2**32)
    nitems = np.prod(shape)
    byts = np.random.randint(0, 256, nitems * dtype.itemsize, "u1")
    return byts.view(dtype).reshape(shape)


@pytest.fixture(scope="session")
def data(bounds, data_block_slices, dtype, num_subchannels):
    data_len = bounds[1] - bounds[0] + 1
    if num_subchannels == 1:
        shape = (data_len,)
    else:
        shape = (data_len, num_subchannels)

    # start off with data array containing its appropriate fill value
    if np.issubdtype(dtype, np.inexact):
        fill_value = np.nan
    elif dtype.names is not None:
        fill_value = np.empty(1, dtype=dtype)[0]
        fill_value["r"] = np.iinfo(dtype["r"]).min
        fill_value["i"] = np.iinfo(dtype["i"]).min
    else:
        fill_value = np.iinfo(dtype).min
    data = np.full(shape, fill_value, dtype=dtype)

    rdata = generate_rf_data(shape, dtype, 0)
    # fill array data for what we will be writing
    for sstart, sstop in data_block_slices:
        bstart = sstart - bounds[0]
        bend = sstop - bounds[0]
        data[bstart:bend] = rdata[bstart:bend]

    return data


@pytest.fixture(scope="session")
def data_file_list(sample_params):
    # manually generated to match output of writing data given
    # _sample_rate_params
    # get params as a tuple of values sorted in key order
    params = tuple(v for k, v in sorted(sample_params.items()))
    file_lists = {
        (400, 3, 200, 2): [
            os.path.join("2014-03-09T12-30-30", "rf@1394368230.000.h5"),
            os.path.join("2014-03-09T12-30-30", "rf@1394368230.400.h5"),
            os.path.join("2014-03-09T12-30-30", "rf@1394368230.800.h5"),
            os.path.join("2014-03-09T12-30-30", "rf@1394368231.200.h5"),
            os.path.join("2014-03-09T12-30-30", "rf@1394368231.600.h5"),
            os.path.join("2014-03-09T12-30-32", "rf@1394368232.400.h5"),
            os.path.join("2014-03-09T12-30-32", "rf@1394368232.800.h5"),
        ]
    }
    return file_lists[params]


@pytest.fixture(scope="class")
def datadir(param_str, tmpdir_factory):
    return tmpdir_factory.mktemp(f"{param_str}_", numbered=True)


@pytest.fixture(scope="class")
def chdir(param_str, datadir):
    chdir = datadir.mkdir(param_str)
    return chdir


@pytest.fixture(scope="class")
def channel(chdir):
    return str(chdir.basename)


@pytest.fixture(scope="class")
def drf_writer_param_dict(chdir, param_dict):
    return dict(
        directory=str(chdir),
        dtype=param_dict["dtype"],
        subdir_cadence_secs=param_dict["subdir_cadence_secs"],
        file_cadence_millisecs=param_dict["file_cadence_millisecs"],
        start_global_index=param_dict["start_global_index"],
        sample_rate_numerator=param_dict["sample_rate_numerator"],
        sample_rate_denominator=param_dict["sample_rate_denominator"],
        uuid_str=chdir.basename,
        compression_level=param_dict["compression_level"],
        checksum=param_dict["checksum"],
        is_complex=param_dict["is_complex"],
        num_subchannels=param_dict["num_subchannels"],
        is_continuous=param_dict["is_continuous"],
        marching_periods=False,
    )


@pytest.fixture(scope="class")
def drf_writer_factory(drf_writer_param_dict):
    def writer_factory(**kwargs_updates):
        kwargs = drf_writer_param_dict.copy()
        kwargs.update(**kwargs_updates)
        return digital_rf.DigitalRFWriter(**kwargs)

    return writer_factory


@pytest.fixture(scope="class")
def drf_writer(drf_writer_factory):
    with drf_writer_factory() as dwo:
        yield dwo


@pytest.fixture(scope="class")
def drf_reader(chdir):
    with digital_rf.DigitalRFReader(chdir.dirname) as dro:
        yield dro
