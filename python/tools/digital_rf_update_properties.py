#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Convert early Digital RF 2 (< 2.4) metadata.h5 file to drf_properties.h5."""

from __future__ import absolute_import, division, print_function

import argparse
import os

import digital_rf
import h5py


def update_properties_file(channel_dir):
    """Helper function to create top-level drf_properties.h5 from metadata.h5.

    This function creates a drf_properties.h5 file in a Digital RF
    channel directory using the duplicate attributes stored in one of the data
    files and the legacy metadata.h5 file.


    Parameters
    ----------

    channel_dir : string
        Channel directory containing Digital RF subdirectories in the form
        YYYY-MM-DDTHH-MM-SS, but missing a drf_properties.h5 file.

    """
    attr_list = [
        "H5Tget_class",
        "H5Tget_size",
        "H5Tget_order",
        "H5Tget_offset",
        "subdir_cadence_secs",
        "file_cadence_millisecs",
        "sample_rate_numerator",
        "sample_rate_denominator",
        "is_complex",
        "num_subchannels",
        "is_continuous",
        "epoch",
        "digital_rf_time_description",
        "digital_rf_version",
    ]
    properties_file = os.path.join(channel_dir, "drf_properties.h5")
    if os.path.exists(properties_file):
        raise IOError("drf_properties.h5 already exists in %s" % (channel_dir))

    # metadata file check
    mdata_file = os.path.join(channel_dir, "metadata.h5")
    attr_dict = {i: None for i in attr_list}
    # any digital_rf version from 2 through 2.4 with a metadata.h5 is a valid
    # 2.5 version file with drf_properties.h5 in place
    attr_dict["digital_rf_version"] = "2.5.0"
    with h5py.File(mdata_file, "r") as fi:
        for i_attr in attr_list:
            if i_attr in fi.attrs and attr_dict[i_attr] is None:
                attr_dict[i_attr] = fi.attrs[i_attr]
            elif i_attr == "sample_rate_numerator":
                sps = fi.attrs["samples_per_second"].item()
                if sps % 1 != 0:
                    errstr = (
                        "No sample rate numerator value and sample rate is not"
                        " an integer."
                    )
                    raise IOError(errstr)
                attr_dict[i_attr] = fi.attrs["samples_per_second"]
            elif i_attr == "sample_rate_denominator":
                sps = fi.attrs["samples_per_second"].item()
                if sps % 1 != 0:
                    errstr = (
                        "No sample rate denominator value and sample rate is"
                        " not an integer."
                    )
                    raise IOError(errstr)
                attr_dict[i_attr] = fi.attrs["samples_per_second"]
                attr_dict[i_attr][0] = 1

    first_rf_file = digital_rf.ilsdrf(
        channel_dir, include_dmd=False, include_drf_properties=False
    ).next()

    with h5py.File(first_rf_file, "r") as fi:
        md = fi["rf_data"].attrs
        for i_attr in attr_list:
            if i_attr in md and attr_dict[i_attr] is None:
                attr_dict[i_attr] = md[i_attr]

    with h5py.File(properties_file, "w") as fo:
        for i_attr in attr_list:
            fo.attrs[i_attr] = attr_dict[i_attr]


if __name__ == "__main__":
    # command line interface
    parser = argparse.ArgumentParser(
        description="""Convert Digital RF channel metadata.h5 file to
                       drf_properties.h5."""
    )
    parser.add_argument(
        "chdir", help="Digital RF channel directory containing metadata.h5"
    )
    args = parser.parse_args()

    update_properties_file(args.chdir)
