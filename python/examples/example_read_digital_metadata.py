# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""An example of reading Digital Metadata in python.

Assumes the example Digital Metadata write script has already been run.

"""
from __future__ import absolute_import, division, print_function

import os
import tempfile

import digital_rf
import numpy as np

metadata_dir = os.path.join(tempfile.gettempdir(), "example_metadata")
stime = 1447082580

try:
    dmr = digital_rf.DigitalMetadataReader(metadata_dir)
except IOError:
    print("Run example_write_digital_metadata.py before running this script.")
    raise

print("init okay")
start_idx = int(np.uint64(stime * dmr.get_samples_per_second()))
first_sample, last_sample = dmr.get_bounds()
print("bounds are %i to %i" % (first_sample, last_sample))

fields = dmr.get_fields()
print("Available fields are <%s>" % (str(fields)))

print("first read - just get one column simple_complex")
data_dict = dmr.read(start_idx, start_idx + 2, "single_complex")
for key in data_dict.keys():
    print((key, data_dict[key]))

print("second read - just 2 columns: simple_complex and numpy_obj")
data_dict = dmr.read(start_idx, start_idx + 2, ("single_complex", "numpy_obj"))
for key in data_dict.keys():
    print((key, data_dict[key]))

print("third read - get all columns")
data_dict = dmr.read(start_idx, start_idx + 2)
for key in data_dict.keys():
    print((key, data_dict[key]))

print("just get latest metadata")
latest_meta = dmr.read_latest()
print(latest_meta)

print("test of get_samples_per_second")
sps = dmr.get_samples_per_second()
print(sps)
