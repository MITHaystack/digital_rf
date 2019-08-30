# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""A simple example of writing Digital Metadata with python.

Now writes data into two levels of dictionaries/groups. API allow any finite
number of levels.

"""
from __future__ import absolute_import, division, print_function

import os
import shutil
import tempfile

import digital_rf
import numpy as np

metadata_dir = os.path.join(tempfile.gettempdir(), "example_metadata")
subdirectory_cadence_seconds = 3600
file_cadence_seconds = 60
samples_per_second_numerator = 10
samples_per_second_denominator = 9
file_name = "rideout"
stime = 1447082580

shutil.rmtree(metadata_dir, ignore_errors=True)
os.makedirs(metadata_dir)

dmw = digital_rf.DigitalMetadataWriter(
    metadata_dir,
    subdirectory_cadence_seconds,
    file_cadence_seconds,
    samples_per_second_numerator,
    samples_per_second_denominator,
    file_name,
)
print("first create okay")

data_dict = {}
start_idx = int(np.uint64(stime * dmw.get_samples_per_second()))
# To save an array of data, make sure the first axis has the same length
# as the samples index
idx_arr = np.arange(70, dtype=np.int64) + start_idx

int_data = np.arange(70, dtype=np.int32)
data_dict["int_data"] = int_data
# can even do multi-dimensional arrays!
int_mat = np.arange(70 * 3 * 2, dtype=np.int32).reshape(70, 3, 2)
data_dict["int_mat"] = int_mat
# These single dimensional arrays will save each individual elements to each
# time.
float_data = np.arange(70, dtype=np.float32)
data_dict["float_data"] = float_data
complex_data = np.arange(70, dtype=np.complex64)
data_dict["complex_data"] = complex_data
single_int = 5
data_dict["single_int"] = np.int32(single_int)
single_float = 6.0
data_dict["single_float"] = np.float64(single_float)
single_complex = 7.0 + 8.0j
data_dict["single_complex"] = np.complex(single_complex)

# now create subdirectories
sub_dict = {}
sub_dict["single_int"] = np.int32(single_int)
sub_dict["single_float"] = np.float64(single_float)
sub_dict["single_complex"] = np.complex(single_complex)
level2_dict = {}  # embed yey another level
level2_dict["single_float"] = np.float64(single_float)
sub_dict["level2"] = level2_dict


data_dict["sub_system"] = sub_dict

# complex python dmwect
n = np.ones((10, 4), dtype=np.float64)
n[5, :] = 17.0
data_dict["numpy_obj"] = [n for i in range(70)]


dmw.write(idx_arr, data_dict)
print("first write_metadata okay")

# write same data again after incrementating inx
idx_arr += 70

del dmw
dmw = digital_rf.DigitalMetadataWriter(
    metadata_dir,
    subdirectory_cadence_seconds,
    file_cadence_seconds,
    samples_per_second_numerator,
    samples_per_second_denominator,
    file_name,
)
print("second create okay")


dmw.write(idx_arr, data_dict)
print("second write_metadata okay")
