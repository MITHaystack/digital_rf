# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Benchmark I/O of Digital RF read in different configurations.

Assumes the benchmark write script has already been run.

"""
from __future__ import absolute_import, division, print_function

import os
import sys
import tempfile
import time

import digital_rf

# constants
WRITE_BLOCK_SIZE = 1000
FILE_SAMPLES = 1000000
N_WRITES = int(1e9 // WRITE_BLOCK_SIZE)


def test_read(channel_name, test_read_obj):
    """test_read measures the speed of reading all the data back into numpy
    arrays for a given channel_name and digital_rf.DigitalRFReader object
    """
    if sys.platform == "darwin":
        os.system("purge")
    t2 = time.time()
    start_index, end_index = test_read_obj.get_bounds(channel_name)
    print(
        "get_bounds returned %i - %i and took %f seconds"
        % (start_index, end_index, time.time() - t2)
    )
    t2 = time.time()
    result = test_read_obj.get_continuous_blocks(start_index, end_index, channel_name)
    print(
        "get_continuous_blocks returned <%s> and took %f seconds"
        % (str(result), time.time() - t2)
    )

    next_sample = start_index
    t2 = time.time()
    count = 0
    while next_sample < end_index - FILE_SAMPLES:
        arr = test_read_obj.read(
            next_sample, next_sample + (FILE_SAMPLES - 1), channel_name
        )
        key = list(arr.keys())[0]
        if len(arr[key]) != FILE_SAMPLES:
            raise IOError("%i != %i" % (len(arr), FILE_SAMPLES))
        next_sample += FILE_SAMPLES
        count += 1
        if count % 100 == 0:
            print("%i out of 1000" % (count))
    seconds = time.time() - t2
    speedMB = (count * FILE_SAMPLES * 4) / (1.0e6 * seconds)
    print("Total read time %i seconds, speed %1.2f MB/s" % (int(seconds), speedMB))


t = time.time()
datadir = os.path.join(tempfile.gettempdir(), "benchmark_digital_rf")
test_read_obj = digital_rf.DigitalRFReader(datadir)
print("metadata analysis took %f seconds" % (time.time() - t))

print("\nTest 0 - read Hdf5 files with no compress, no checksum - channel name = junk0")
test_read("junk0", test_read_obj)

print(
    "\nTest 1 - read Hdf5 files with no compress, no checksum, chunked - channel name = junk1"
)
test_read("junk1", test_read_obj)

print(
    "\nTest 2 -read Hdf5 files with no compress, but with level 9 checksum - channel name = junk2"
)
test_read("junk2", test_read_obj)

print(
    "\nTest 3 - read Hdf5 files with compress, and with level 9 checksum - channel name = junk3"
)
test_read("junk3", test_read_obj)
