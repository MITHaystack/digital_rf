# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""A simple example of writing Digital RF with python.

Writes continuous complex short data.

"""
from __future__ import absolute_import, division, print_function

import os
import shutil
import tempfile

import digital_rf
import numpy as np

datadir = os.path.join(tempfile.gettempdir(), "example_digital_rf")
chdir = os.path.join(datadir, "junk0")

# writing parameters
sample_rate_numerator = int(100)  # 100 Hz sample rate - typically MUCH faster
sample_rate_denominator = 1
sample_rate = np.longdouble(sample_rate_numerator) / sample_rate_denominator
dtype_str = "i2"  # short int
sub_cadence_secs = (
    4  # Number of seconds of data in a subdirectory - typically MUCH larger
)
file_cadence_millisecs = 400  # Each file will have up to 400 ms of data
compression_level = 1  # low level of compression
checksum = False  # no checksum
is_complex = True  # complex values
is_continuous = True
num_subchannels = 1  # only one subchannel
marching_periods = False  # no marching periods when writing
uuid = "Fake UUID - use a better one!"
vector_length = 100  # number of samples written for each call - typically MUCH longer

# create short data in r/i to test using that to write
arr_data = np.ones(
    (vector_length, num_subchannels), dtype=[("r", np.int16), ("i", np.int16)]
)
for i in range(len(arr_data)):
    arr_data[i]["r"] = 2 * i
    arr_data[i]["i"] = 3 * i

# start 2014-03-09 12:30:30 plus one sample
start_global_index = int(np.uint64(1394368230 * sample_rate)) + 1

# set up top level directory
shutil.rmtree(chdir, ignore_errors=True)
os.makedirs(chdir)

print(
    (
        "Writing complex short to multiple files and subdirectores in {0}"
        " channel junk0"
    ).format(datadir)
)

# init
dwo = digital_rf.DigitalRFWriter(
    chdir,
    dtype_str,
    sub_cadence_secs,
    file_cadence_millisecs,
    start_global_index,
    sample_rate_numerator,
    sample_rate_denominator,
    uuid,
    compression_level,
    checksum,
    is_complex,
    num_subchannels,
    is_continuous,
    marching_periods,
)
# write
for i in range(7):  # will write 700 samples - so creates two subdirectories
    result = dwo.rf_write(arr_data)
    print("Last file written = %s" % (dwo.get_last_file_written()))
    print("Last dir written = %s" % (dwo.get_last_dir_written()))
    print("UTC timestamp of last write is %i" % (dwo.get_last_utc_timestamp()))

# close
dwo.close()
print("done test")
