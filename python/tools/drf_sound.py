#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""

drf_sound.py

Create sound output for a set of digital_rf data. The user can either output
directly to sounddevice or through a wave file save out.

"""
from __future__ import absolute_import, division, print_function

import datetime
import optparse
import os
import sys

import dateutil
import digital_rf as drf
import numpy as np
import pytz
import scipy
import scipy.io.wavfile
import scipy.signal
import sounddevice as sd


class SoundDRF(object):
    def __init__(self, control):
        """ Initializes the SoundDRF class."""

        self.control = control
        ch = self.control.channel.split(":")
        self.channel = ch[0]
        self.sub_channel = int(ch[1])

        # open digital RF path
        self.dio = drf.DigitalRFReader(self.control.path)

        if self.control.verbose:
            print("channel bounds:", self.dio.get_bounds(self.channel))

        self.bounds = self.dio.get_bounds(self.channel)

        print("bounds ", self.bounds)

    def makeasound(self):
        """

            Iterate over the data set and output a sound through sounddevice.

        """
        sr = self.dio.get_properties(self.channel)["samples_per_second"]

        if self.control.verbose:
            print("sample rate: ", sr)

        bound = self.dio.get_bounds(self.channel)

        if self.control.verbose:
            print("data bounds: ", bound)

        if self.control.start:
            dtst0 = dateutil.parser.parse(self.control.start)
            st0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            st0 = int(st0 * sr)
        else:
            st0 = int(bound[0])

        if self.control.end:
            dtst0 = dateutil.parser.parse(self.control.end)
            et0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            et0 = int(et0 * sr)
        else:
            et0 = int(bound[1])

        if self.control.verbose:

            print("start sample st0: ", st0)
            print("end sample et0: ", et0)

        decimate = int(self.control.timedilation * sr / self.control.audiosampling)
        blocks = self.control.blocks

        dsamps = int(2 ** 15)

        samples_per_stripe = dsamps * decimate
        total_samples = blocks * samples_per_stripe

        if total_samples > (et0 - st0):
            print(
                "Insufficient samples for %d samples per stripe and %d blocks between %ld and %ld"
                % (samples_per_stripe, blocks, st0, et0)
            )
            return

        stripe_stride = (et0 - st0) // blocks
        reads_per_block = stripe_stride // samples_per_stripe

        start_sample = st0

        print("first ", start_sample)

        audiostuff = np.zeros((blocks, reads_per_block * dsamps), dtype=np.float)
        if self.control.verbose:
            print(
                "processing info : ",
                blocks,
                reads_per_block,
                samples_per_stripe,
                end=" ",
            )

        for iblock in range(blocks):

            for iread in range(reads_per_block):

                if self.control.verbose:
                    print(
                        "read vector :", self.channel, start_sample, samples_per_stripe
                    )

                data = self.dio.read_vector(
                    start_sample, samples_per_stripe, self.channel, self.sub_channel
                )

                if self.control.freqshift:
                    tvec = np.arange(len(data), dtype=np.float) / sr + start_sample / sr
                    f_osc = np.exp(1j * 2 * np.pi * self.control.freqshift * tvec)
                    data_fs = data * f_osc
                else:
                    data_fs = data

                if decimate > 1:
                    audiostuff[
                        iblock, iread * dsamps : (iread + 1) * dsamps
                    ] = scipy.signal.decimate(data_fs, decimate).real

                start_sample += samples_per_stripe
        audiostuff_cent = audiostuff - audiostuff.flatten().mean()
        audiostuff_norm = audiostuff_cent / np.abs(audiostuff_cent.flatten()).max()
        audiostuff_norm = audiostuff_norm.astype(float)
        if self.control.outname:
            fname = os.path.splitext(self.control.outname)[0]
            ext = ".wav"
            for i in range(blocks):
                try:
                    scipy.io.wavfile.write(
                        fname + str(i) + ext,
                        self.control.audiosampling,
                        audiostuff_norm[i],
                    )
                    print("Wrote {} file.".format(fname + str(i) + ext))
                except:
                    print("Failed to write {}.".format(fname + str(i) + ext))
        else:
            for i in range(blocks):
                sd.play(audiostuff_norm[i], self.control.audiosampling)


def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    if str_input is None:
        parser = optparse.OptionParser()
    else:
        parser = optparse.OptionParser(str_input)

    parser.add_option(
        "-s",
        "--start",
        dest="start",
        default=None,
        help="Use the provided start time instead of the first time in the data. format is ISO8601: 2015-11-01T15:24:00Z",
    )
    parser.add_option(
        "-e",
        "--end",
        dest="end",
        default=None,
        help="Use the provided end time for the plot. format is ISO8601: 2015-11-01T15:24:00Z",
    )

    parser.add_option(
        "-p",
        "--path",
        dest="path",
        help="Use data from the provided digital RF data <path>.",
    )
    parser.add_option(
        "-c",
        "--channel",
        dest="channel",
        default="ch0:0",
        help="Use data from the provided digital RF channel <channel>:<subchannel>.",
    )

    parser.add_option(
        "-d",
        "--decimation",
        dest="decimation",
        default=1,
        type="int",
        help="The decimation factor for the data (integer).",
    )
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print status messages to stdout.",
    )
    parser.add_option(
        "-o",
        "--outname",
        dest="outname",
        default=None,
        type=str,
        help="Name of file that figure will be saved under.",
    )
    parser.add_option(
        "-f",
        "--freqshift",
        dest="freqshift",
        default=None,
        type=float,
        help="Frequency shift in Hz for the signal.",
    )
    parser.add_option(
        "-a",
        "--audiosampling",
        dest="audiosampling",
        default=44100.0,
        type=float,
        help="Audio sampling frequency in Hz.",
    )
    parser.add_option(
        "-t",
        "--timedilation",
        dest="timedilation",
        default=1.0,
        type=float,
        help="Time dilation of data.",
    )
    parser.add_option(
        "-b",
        "--blocks",
        dest="blocks",
        default=1,
        type=int,
        help="Number of blocks the file will be broken into.",
    )

    (options, args) = parser.parse_args()

    return (options, args)


#
# MAIN PROGRAM
#

# Setup Defaults
if __name__ == "__main__":
    """
        Needed to add main function to use outside functions outside of module.
    """


# Parse the Command Line for configuration
(options, args) = parse_command_line()

if options.path == None:
    print("Please provide an input source with the -p option!")
    sys.exit(1)

# Activate the SoundDRF
sdrf = SoundDRF(options)

sdrf.makeasound()
