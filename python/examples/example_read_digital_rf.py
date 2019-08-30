# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""An example of reading Digital RF data in python.

Assumes the example Digital RF write script has already been run.

"""
from __future__ import absolute_import, division, print_function

import os
import tempfile

import digital_rf

datadir = os.path.join(tempfile.gettempdir(), "example_digital_rf")
try:
    dro = digital_rf.DigitalRFReader(datadir)
except ValueError:
    print("Please run the example write script before running this example.")
    raise

channels = dro.get_channels()
print("found channels: %s" % (str(channels)))

print("working on channel junk0")
start_index, end_index = dro.get_bounds("junk0")
print("get_bounds returned %i - %i" % (start_index, end_index))
cont_data_arr = dro.get_continuous_blocks(start_index, end_index, "junk0")
print(
    (
        "The following is a OrderedDict of all continuous block of data in"
        "(start_sample, length) format: %s"
    )
    % (str(cont_data_arr))
)

# read data - the first 3 reads of four should succeed, the fourth read
# will be beyond the available data
start_sample = list(cont_data_arr.keys())[0]
for i in range(4):
    try:
        result = dro.read_vector(start_sample, 200, "junk0")
        print(
            "read number %i got %i samples starting at sample %i"
            % (i, len(result), start_sample)
        )
        start_sample += 200
    except IOError:
        print("Read number %i went beyond existing data as expected" % (i))

# finally, get all the built in rf properties
rf_dict = dro.get_properties("junk0")
print(
    ("Here is the metadata built into drf_properties.h5 (valid for all data):" " %s")
    % str(rf_dict)
)
