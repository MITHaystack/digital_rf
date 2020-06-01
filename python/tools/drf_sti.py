#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Create a spectral time intensity summary plot for a data set."""


import datetime
import optparse
import os
import sys
import time
import traceback

import dateutil
import digital_rf as drf
import matplotlib.gridspec
import matplotlib.mlab
import matplotlib.pyplot
import numpy as np
import pytz
import scipy
import scipy.signal


class DataPlotter(object):
    def __init__(self, control):
        """Initialize a data plotter for STI plotting."""
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

        # Figure setup

        self.f = matplotlib.pyplot.figure(
            figsize=(7, np.min([np.max([4, self.control.frames]), 7])), dpi=128
        )

        self.gridspec = matplotlib.gridspec.GridSpec(self.control.frames, 1)

        self.subplots = []

        """ Setup the subplots for this display """
        for n in np.arange(self.control.frames):
            ax = self.f.add_subplot(self.gridspec[n])
            self.subplots.append(ax)

    def plot(self):
        """Iterate over the data set and plot the STI into the subplot panels.

        Each panel is divided into a provided number of bins of a given
        integration length. Strides between the panels are made between
        integrations.

        """
        # initialize outside the loop to avoid memory leak

        # initial plotting scales
        vmin = 0
        vmax = 0

        sr = self.dio.get_properties(self.channel)["samples_per_second"]

        if self.control.verbose:
            print("sample rate: ", sr)

        # initial time info
        b = self.dio.get_bounds(self.channel)

        if self.control.verbose:
            print("data bounds: ", b)

        if self.control.start:
            dtst0 = dateutil.parser.parse(self.control.start)
            st0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            st0 = int(st0 * sr)
        else:
            st0 = int(b[0])

        if self.control.end:
            dtst0 = dateutil.parser.parse(self.control.end)
            et0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            et0 = int(et0 * sr)
        else:
            et0 = int(b[1])

        if self.control.verbose:

            print("start sample st0: ", st0)
            print("end sample et0: ", et0)

        blocks = self.control.bins * self.control.frames

        samples_per_stripe = (
            self.control.num_fft * self.control.integration * self.control.decimation
        )
        total_samples = blocks * samples_per_stripe

        if total_samples > (et0 - st0):
            print(
                "Insufficient samples for %d samples per stripe and %d blocks between %ld and %ld"
                % (samples_per_stripe, blocks, st0, et0)
            )
            return

        stripe_stride = (et0 - st0) / blocks

        bin_stride = stripe_stride / self.control.bins

        start_sample = st0

        print("first ", start_sample)

        # get metadata
        # this could be done better to ensure we catch frequency or sample rate
        # changes
        mdt = self.dio.read_metadata(st0, et0, self.channel)
        try:
            md = mdt[list(mdt.keys())[0]]
            cfreq = md["center_frequencies"].ravel()[self.sub_channel]
        except (IndexError, KeyError):
            cfreq = 0.0

        if self.control.verbose:
            print(
                "processing info : ",
                self.control.frames,
                self.control.bins,
                samples_per_stripe,
                bin_stride,
            )

        for p in np.arange(self.control.frames):
            sti_psd_data = np.zeros([self.control.num_fft, self.control.bins], np.float)
            sti_times = np.zeros([self.control.bins], np.complex128)

            for b in np.arange(self.control.bins, dtype=np.int_):

                if self.control.verbose:
                    print(
                        "read vector :", self.channel, start_sample, samples_per_stripe
                    )

                data = self.dio.read_vector(
                    start_sample, samples_per_stripe, self.channel, self.sub_channel
                )
                if self.control.decimation > 1:
                    data = scipy.signal.decimate(data, self.control.decimation)
                    sample_freq = sr / self.control.decimation
                else:
                    sample_freq = sr

                if self.control.mean:
                    detrend_fn = matplotlib.mlab.detrend_mean
                else:
                    detrend_fn = matplotlib.mlab.detrend_none

                try:
                    psd_data, freq_axis = matplotlib.mlab.psd(
                        data,
                        NFFT=self.control.num_fft,
                        Fs=float(sample_freq),
                        detrend=detrend_fn,
                        scale_by_freq=False,
                    )
                except:
                    traceback.print_exc(file=sys.stdout)

                sti_psd_data[:, b] = np.real(10.0 * np.log10(np.abs(psd_data) + 1e-12))

                sti_times[b] = start_sample / sr

                start_sample += stripe_stride

            # Now Plot the Data
            ax = self.subplots[p]

            # determine image x-y extent
            extent = (
                0,
                self.control.bins,
                np.min(freq_axis) / 1e3,
                np.max(freq_axis) / 1e3,
            )

            # determine image color extent in log scale units
            Pss = sti_psd_data

            if self.control.zaxis:
                vmin = int(self.control.zaxis.split(":")[0])
                vmax = int(self.control.zaxis.split(":")[1])
            else:
                med_Pss = np.nanmedian(Pss)
                max_Pss = np.nanmax(Pss)
                vmin = np.real(med_Pss - 6.0)
                vmax = np.real(med_Pss + (max_Pss - med_Pss) * 0.61803398875 + 50.0)

            im = ax.imshow(
                sti_psd_data,
                cmap="jet",
                origin="lower",
                extent=extent,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            ax.set_ylabel("f (kHz)", fontsize=8)

            # plot dates

            tick_spacing = np.arange(
                self.control.bins / 8,
                self.control.bins,
                self.control.bins / 8,
                dtype=np.int_,
            )
            ax.set_xticks(tick_spacing)
            tick_labels = []

            for s in tick_spacing:
                tick_time = sti_times[s]

                if tick_time == 0:
                    tick_string = ""
                else:
                    gm_tick_time = time.gmtime(np.real(tick_time))
                    tick_string = "%02d:%02d:%02d" % (
                        gm_tick_time[3],
                        gm_tick_time[4],
                        gm_tick_time[5],
                    )
                    tick_labels.append(tick_string)

            ax.set_xticklabels(tick_labels)

            # set the font sizes
            tl = ax.get_xticklabels()

            for tk in tl:
                tk.set_size(8)
            del tl

            tl = ax.get_yticklabels()

            for tk in tl:
                tk.set_size(8)
            del tl

        print("last ", start_sample)

        # create a time stamp
        start_time = st0 / sr
        srt_time = time.gmtime(start_time)
        sub_second = int(round((start_time - int(start_time)) * 100))

        timestamp = "%d-%02d-%02d %02d:%02d:%02d.%02d UT" % (
            srt_time[0],
            srt_time[1],
            srt_time[2],
            srt_time[3],
            srt_time[4],
            srt_time[5],
            sub_second,
        )

        self.f.suptitle(
            "%s %s %4.2f MHz (%s)"
            % (self.control.title, timestamp, cfreq / 1e6, self.control.path),
            fontsize=10,
        )

        # ax.legend(fontsize=8)
        ax.set_xlabel("time (UTC)", fontsize=8)

        # fixup ticks

        tl = ax.get_xticklabels()
        for tk in tl:
            tk.set_size(8)
        del tl
        tl = ax.get_yticklabels()
        for tk in tl:
            tk.set_size(8)
        del tl

        self.gridspec.update()

        self.f.tight_layout()

        self.f.subplots_adjust(top=0.95, right=0.88)
        cax = self.f.add_axes([0.9, 0.12, 0.02, 0.80])
        self.f.colorbar(im, cax=cax)
        if self.control.outname:
            fname, ext = os.path.splitext(self.control.outname)
            if ext == "":
                ext = ".png"
            print("Save plot as {}".format(fname + ext))
            matplotlib.pyplot.savefig(fname + ext)
        if self.control.appear or not self.control.outname:
            print("Show plot")
            matplotlib.pyplot.show()


def parse_command_line(str_input=None):
    # if str_input is None:
    #     parser = optparse.OptionParser()
    # else:
    #     parser = optparse.OptionParser(str_input)
    parser = optparse.OptionParser()

    parser.add_option(
        "-t",
        "--title",
        dest="title",
        default="Digital RF Data",
        help="Use title provided for the data.",
    )
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
        "-l",
        "--length",
        dest="length",
        default=0.04,
        type="float",
        help="The default data length in seconds for unframed data.",
    )
    parser.add_option(
        "-b",
        "--bins",
        dest="bins",
        default=128,
        type="int",
        help="The number of time bins for the STI.",
    )
    parser.add_option(
        "-f",
        "--frames",
        dest="frames",
        default=4,
        type="int",
        help="The number of sub-panel frames in the plot.",
    )
    parser.add_option(
        "-n",
        "--num_fft",
        dest="num_fft",
        default=128,
        type="int",
        help="The number of FFT bints for the STI.",
    )
    parser.add_option(
        "-i",
        "--integration",
        dest="integration",
        default=1,
        type="int",
        help="The number of rasters to integrate for each plot.",
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
        "-m",
        "--mean",
        dest="mean",
        action="store_true",
        default=False,
        help="Remove the mean from the data at the PSD processing step.",
    )
    parser.add_option(
        "-z",
        "--zaxis",
        dest="zaxis",
        default=None,
        type="string",
        help="zaxis colorbar setting e.g. -50:50",
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
        "-a",
        "--appear",
        action="store_true",
        dest="appear",
        default=False,
        help="Makes the plot appear through pyplot show.",
    )
    if str_input is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(str_input)

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

    if options.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)

    # Activate the DataPlotter
    dpc = DataPlotter(options)

    dpc.plot()
