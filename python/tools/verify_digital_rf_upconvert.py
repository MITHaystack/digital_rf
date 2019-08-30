#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""verify_digital_rf_upconvert will test if a digital rf 2 subdirectory matches sample for sample a digital rf 1 subdirectory.

$Id$
"""
from __future__ import absolute_import, division, print_function

import argparse
import sys
import traceback

import digital_rf
import numpy as np
from digital_rf import digital_rf_deprecated_hdf5  # for reading old formatter

read_len = 1000000  # default read len


### main begins here ###
if __name__ == "__main__":

    # command line interface
    parser = argparse.ArgumentParser(
        description="verify_digital_rf_upconvert will test if a digital rf 2 subdirectory matches sample for sample a digital rf 1 subdirectory. Prints True or False"
    )
    parser.add_argument(
        "--drf2",
        metavar="drf2Dir",
        help="Digital RF v 2 top level directory to be compared",
        required=True,
    )
    parser.add_argument(
        "--drf1",
        metavar="drf1Dir",
        help="Digital RF v 1 top level directory to be compared",
        required=True,
    )
    args = parser.parse_args()

    try:
        reader2 = digital_rf.DigitalRFReader(args.drf2)
    except:
        traceback.print_exc()
        sys.exit(-1)

    try:
        reader1 = digital_rf_deprecated_hdf5.read_hdf5(args.drf1)
    except:
        traceback.print_exc()
        sys.exit(-1)

    # compare each channel separately
    for channel in reader2.get_channels():
        print("comparing channel %s" % (channel))
        bounds2 = reader2.get_bounds(channel)
        bounds1 = reader1.get_bounds(channel)

        cont_blocks_2 = reader2.get_continuous_blocks(bounds2[0], bounds2[1], channel)
        cont_blocks_1 = reader1.get_continuous_blocks(bounds1[0], bounds1[1], channel)
        for i, key in enumerate(cont_blocks_2.keys()):
            s2 = key
            n2 = cont_blocks_2[key]
            s1 = cont_blocks_1[i, 0]
            n1 = cont_blocks_1[i, 1]
            if n1 > read_len:
                # keep array size reasonable
                n1 = read_len
            data1 = reader1.read_vector_raw(s1, n1, channel)
            data2 = reader2.read_vector_raw(s1, n1, channel)
            if not np.array_equal(data1, data2):
                print(data1)
                print(data2)
                print("False")
                sys.exit(-1)

    print("True")
