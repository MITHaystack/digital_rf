#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Convert Digital RF 1 formatted data to the current version.

Use verify_digital_rf_upconvert.py if you want to test the conversion.

$Id$
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import math
import os
import warnings

import digital_rf
import h5py
import numpy as np
from digital_rf import digital_rf_deprecated_hdf5  # for reading old formatter

read_len = 1000000  # default read len


if __name__ == "__main__":
    # command line interface
    parser = argparse.ArgumentParser(
        description="""Convert Digital RF 1 formatted data to the current
                       version."""
    )
    parser.add_argument(
        "--source",
        metavar="sourceDir",
        required=True,
        help="""Source top level directory containing Digital RF channels to be
                converted""",
    )
    parser.add_argument(
        "--target",
        metavar="targetDir",
        required=True,
        help="""Target top level directory where Digital RF channels will be
                written after upconversion""",
    )
    parser.add_argument(
        "--dir_secs",
        metavar="Subdir_cadence_seconds",
        type=int,
        default=3600,
        help="""Number of seconds of data to store per subdirectory.
                Default is 3600 (one hour)""",
    )
    parser.add_argument(
        "--file_millisecs",
        metavar="File_cadence_milliseconds",
        type=int,
        default=1000,
        help="""Number of milliseconds of data to store per file.
                Default is 1000 (one second)""",
    )
    parser.add_argument(
        "--gzip",
        metavar="GZIP compression 0-9.",
        type=int,
        default=0,
        help="""Level of GZIP compression 1-9.  Default=0 (no compression)""",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="""Set this flag to turn on Hdf5 checksums""",
    )
    args = parser.parse_args()

    reader = digital_rf_deprecated_hdf5.read_hdf5(args.source)

    # convert each channel separately
    for channel in reader.get_channels():
        print("working on channel %s" % (channel))
        metaDict = reader.get_rf_file_metadata(channel)
        bounds = reader.get_bounds(channel)

        # this code only works if the sample rate is an integer
        sample_rate = metaDict["sample_rate"][0]
        if math.fmod(sample_rate, 1.0) != 0.0:
            errstr = (
                "Cannot guess the numerator and denominator with a fractional"
                " sample rate %f"
            )
            raise ValueError(errstr % sample_rate)
        sample_rate_numerator = int(sample_rate)
        sample_rate_denominator = int(1)

        # read critical metadata
        is_complex = bool(metaDict["is_complex"][0])
        num_subchannels = metaDict["num_subchannels"][0]
        uuid_str = str(metaDict["uuid_str"])

        # get first sample to find dtype
        data = reader.read_vector_raw(bounds[0], 1, channel)
        this_dtype = data.dtype
        cont_blocks = reader.get_continuous_blocks(bounds[0], bounds[1], channel)
        subdir = os.path.join(args.target, channel)
        if not os.access(subdir, os.R_OK):
            os.makedirs(subdir)

        # extract sampler_util metadata in any metadata* files
        metadataFiles = glob.glob(os.path.join(args.source, channel, "metadata*.h5"))
        metadataFiles.sort()
        if metadataFiles:
            # create metadata dir, dmd object, and write channel metadata
            mddir = os.path.join(subdir, "metadata")
            if not os.path.exists(mddir):
                os.makedirs(mddir)
            mdo = digital_rf.DigitalMetadataWriter(
                metadata_dir=mddir,
                subdir_cadence_secs=args.dir_secs,
                file_cadence_secs=1,
                sample_rate_numerator=sample_rate_numerator,
                sample_rate_denominator=sample_rate_denominator,
                file_name="metadata",
            )
            for filepath in metadataFiles:
                try:
                    with h5py.File(filepath, "r") as mdfile:
                        md = dict(
                            uuid_str=uuid_str,
                            sample_rate_numerator=sample_rate_numerator,
                            sample_rate_denominator=sample_rate_denominator,
                            # put in a list because we want the data to be a
                            # 1-D array and it would be a single value o/w
                            center_frequencies=[
                                mdfile["center_frequencies"][()].reshape((-1,))
                            ],
                        )
                        try:
                            # try for newer added metadata, usrp_id and friends
                            receiver_dict = dict(
                                description="UHD USRP source using GNU Radio",
                                gain=str(mdfile["usrp_gain"][()]),
                                id=str(mdfile["usrp_id"][()]),
                                stream_args=str(mdfile["usrp_stream_args"][()]),
                                subdev=str(mdfile["usrp_subdev"][()]),
                            )
                        except KeyError:
                            # fallback to original usrp metadata
                            receiver_dict = dict(
                                description="UHD USRP source using GNU Radio",
                                id=str(mdfile["usrp_ip"][()]),
                            )
                        md["receiver"] = receiver_dict
                except (IOError, KeyError):
                    # file is bad, warn and ignore
                    errstr = "Skipping bad metadata file {0}."
                    warnings.warn(errstr.format(filepath))
                else:
                    fname = os.path.basename(filepath)
                    t = np.longdouble(fname.split("@", 1)[1][:-3])
                    s = int((t * sample_rate_numerator) // sample_rate_denominator)
                    mdo.write(samples=s, data=md)

        # create a drf 2 writer
        writer = digital_rf.DigitalRFWriter(
            subdir,
            this_dtype,
            args.dir_secs,
            args.file_millisecs,
            bounds[0],
            sample_rate_numerator,
            sample_rate_denominator,
            uuid_str=uuid_str,
            is_complex=is_complex,
            num_subchannels=num_subchannels,
            compression_level=args.gzip,
            checksum=args.checksum,
            marching_periods=False,
        )

        # write all the data
        for startSample, sampleLen in cont_blocks:
            thisSample = startSample
            endSample = (startSample + sampleLen) - 1
            while thisSample < endSample:
                this_len = min((endSample - thisSample) + 1, read_len)
                data = reader.read_vector_raw(thisSample, this_len, channel)
                writer.rf_write(data, thisSample - bounds[0])
                thisSample += this_len

        writer.close()
