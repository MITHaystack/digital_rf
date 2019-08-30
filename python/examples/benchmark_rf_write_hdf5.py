# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Benchmark I/O of Digital RF write in different configurations.

"""
from __future__ import absolute_import, division, print_function

import os
import shutil
import tempfile
import time

import digital_rf
import numpy as np

# constants
WRITE_BLOCK_SIZE = 1000
N_WRITES = int(1e9 // WRITE_BLOCK_SIZE)
SAMPLE_RATE_NUMERATOR = int(1e9)
SAMPLE_RATE_DENOMINATOR = 1
sample_rate = np.longdouble(np.uint64(SAMPLE_RATE_NUMERATOR)) / np.longdouble(
    np.uint64(SAMPLE_RATE_DENOMINATOR)
)
subdir_cadence_secs = 3600
file_cadence_millisecs = 10

# start 2014-03-09 12:30:30 plus one sample
start_global_index = int(np.uint64(1394368230 * sample_rate)) + 1

# data to write
data_int16 = np.zeros((WRITE_BLOCK_SIZE, 2), dtype="i2")
# make random
for i in range(WRITE_BLOCK_SIZE):
    j = i * 2
    k = i * 2 + 1
    data_int16[i][0] = (j % 32768) * (j + 8192) * (j % 13)
    data_int16[i][1] = (k % 32768) * (k + 8192) * (k % 13)

datadir = os.path.join(tempfile.gettempdir(), "benchmark_digital_rf")
print("creating top level dir {0}".format(datadir))
shutil.rmtree(datadir, ignore_errors=True)
os.makedirs(datadir)

print(
    "Test 0 - simple single write to multiple files, no compress, no checksum - channel 0"
)
chdir = os.path.join(datadir, "junk0")
os.makedirs(chdir)
print("Start writing")
channelObj = digital_rf.DigitalRFWriter(
    chdir,
    "i2",
    subdir_cadence_secs,
    file_cadence_millisecs,
    start_global_index,
    SAMPLE_RATE_NUMERATOR,
    SAMPLE_RATE_DENOMINATOR,
    "Fake_uuid",
    0,
    False,
)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0e6 * seconds)
print("Total time %i seconds, speed %1.2f MB/s" % (int(seconds), speedMB))

print(
    "Test 1 - simple single write to multiple files, no compress, no checksum, chunked - channel 1"
)
chdir = os.path.join(datadir, "junk1")
os.makedirs(chdir)
print("Start writing")
channelObj = digital_rf.DigitalRFWriter(
    chdir,
    "i2",
    subdir_cadence_secs,
    file_cadence_millisecs,
    start_global_index,
    SAMPLE_RATE_NUMERATOR,
    SAMPLE_RATE_DENOMINATOR,
    "Fake_uuid",
    0,
    False,
    is_continuous=False,
)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0e6 * seconds)
print("Total time %i seconds, speed %1.2f MB/s" % (int(seconds), speedMB))


print(
    "Test 2 - simple single write to multiple files, no compress, with checksum - channel 2"
)
chdir = os.path.join(datadir, "junk2")
os.makedirs(chdir)
print("Start writing")
channelObj = digital_rf.DigitalRFWriter(
    chdir,
    "i2",
    subdir_cadence_secs,
    file_cadence_millisecs,
    start_global_index,
    SAMPLE_RATE_NUMERATOR,
    SAMPLE_RATE_DENOMINATOR,
    "Fake_uuid",
    0,
    True,
)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0e6 * seconds)
print("Total time %i seconds, speed %1.2f MB/s" % (int(seconds), speedMB))


print(
    "Test 3 - simple single write to multiple files, compress to level 9, with checksum - channel 3"
)
chdir = os.path.join(datadir, "junk3")
os.makedirs(chdir)
print("Start writing")
channelObj = digital_rf.DigitalRFWriter(
    chdir,
    "i2",
    subdir_cadence_secs,
    file_cadence_millisecs,
    start_global_index,
    SAMPLE_RATE_NUMERATOR,
    SAMPLE_RATE_DENOMINATOR,
    "Fake_uuid",
    9,
    True,
)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0e6 * seconds)
print("Total time %i seconds, speed %1.2f MB/s" % (int(seconds), speedMB))
