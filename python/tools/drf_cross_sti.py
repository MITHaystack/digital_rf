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

drf_cross_sti.py
$Id$

Create a cross spectral time intensity summary plot for the given data sets.

"""


import datetime
import itertools as it
import optparse
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


class DataPlotter:
    def __init__(self, control):
        """__init__: Initializes a data plotter for STI plotting.

        """
        self.control = control

        self.dio = []
        self.dmd = []
        self.channel = []
        self.sub_channel = []
        self.bounds = []

        for idx, p in enumerate(self.control.path):

            ch = self.control.channel[idx].split(":")
            self.channel.append(ch[0])
            self.sub_channel.append(int(ch[1]))

            # open digital RF path
            self.dio.append(drf.DigitalRFReader(p))

            if self.control.verbose:
                print(("bounds:", self.dio[idx].get_bounds(self.channel[idx])))

            self.bounds.append(self.dio[idx].get_bounds(self.channel[idx]))

            # oepn digital metadata path
            self.dmd.append(
                drf.DigitalMetadataReader(p + "/" + self.channel[idx] + "/metadata")
            )

        # processing pair list
        pl = range(len(self.dio))

        if self.control.xtype == "self":
            self.xlist = list(it.product(pl, repeat=2))

        elif self.control.xtype == "pairs":
            args = [iter(pl)] * 2
            self.xlist = list(it.izip_longest(*args))

        elif self.control.xtype == "combo":
            self.xlist = list(it.combinations(pl, 2))

        elif self.control.xtype == "permute":
            self.xlist = list(it.permutations(pl, 2))
        else:
            print(("unknown processing pair type ", self.control.xtype))
            sys.exit(1)

        print(("pair list ", pl))
        print(("xlist ", self.xlist))

        # Figure setup
        # two plots coherence and phase for each pair
        self.f = []
        self.gridspec = []
        self.subplots = []

        for n in np.arange(len(self.xlist)):
            f = matplotlib.pyplot.figure(
                figsize=(7, np.min([np.max([4, self.control.frames]), 7])), dpi=128
            )
            self.f.append(f)

            gridspec = matplotlib.gridspec.GridSpec(self.control.frames * 2, 1)
            self.gridspec.append(gridspec)

            subplots = []
            self.subplots.append(subplots)

        """ Setup the subplots for this display """
        for n in np.arange(len(self.xlist)):
            for m in np.arange(self.control.frames * 2):
                ax = self.f[n].add_subplot(self.gridspec[n][m])
                self.subplots[n].append(ax)

    def plot(self):
        """

            Iterate over the data set and plot the Cross STI into the subplot panels. Each
            panel is divided into a provided number of bins of a given integration
            length. Strides between the panels are made between integrations. Both coherence and
            phase are plotted.

        """

        # initial plotting scales
        vmin = 0
        vmax = 0

        for fidx, xpair in enumerate(self.xlist):

            xidx, yidx = xpair

            if self.control.verbose:
                print(("pair is : ", xidx, yidx))

            # sample rate
            xsr = self.dio[xidx].get_properties(self.channel[xidx])[
                "samples_per_second"
            ]
            ysr = self.dio[yidx].get_properties(self.channel[yidx])[
                "samples_per_second"
            ]

            if self.control.verbose:
                print(("sample rate, x: ", xsr, " y: ", ysr))

            if xsr == ysr:
                sr = xsr
            else:
                print("problem, sample rates of data must currently match!")
                sys.exit(1)

            # initial time info
            xb = self.bounds[xidx]
            yb = self.bounds[yidx]

            if self.control.verbose:
                print(("data bounds, xb: ", xb, " yb: ", yb))

            b = (np.max([xb[0], yb[0]]), np.min([xb[1], yb[1]]))

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

                print(("start sample st0: ", st0))
                print(("end sample et0: ", et0))

            blocks = self.control.bins * self.control.frames

            samples_per_stripe = (
                self.control.num_fft
                * self.control.integration
                * self.control.decimation
            )
            total_samples = blocks * samples_per_stripe

            if total_samples > (et0 - st0):
                print(
                    (
                        "Insufficient samples for %d samples per stripe and %d blocks between %ld and %ld"
                        % (samples_per_stripe, blocks, st0, et0)
                    )
                )
                return

            stripe_stride = (et0 - st0) / blocks

            bin_stride = stripe_stride / self.control.bins

            start_sample = st0

            print(("first ", start_sample))

            # get metadata
            # this could be done better to ensure we catch frequency or sample rate
            # changes
            xmdf = self.dio[xidx].read_metadata(st0, et0, self.channel[xidx])
            try:
                xmd = xmdf[list(xmdf.keys())[0]]
                xcfreq = xmd["center_frequencies"].ravel()[self.sub_channel[xidx]]
            except (IndexError, KeyError):
                xcfreq = 0.0
            ymdf = self.dio[yidx].read_metadata(st0, et0, self.channel[yidx])
            try:
                ymd = ymdf[list(ymdf.keys())[0]]
                ycfreq = ymd["center_frequencies"].ravel()[self.sub_channel[yidx]]
            except (IndexError, KeyError):
                ycfreq = 0.0
            print("center frequencies ", xcfreq, ycfreq)

            if self.control.verbose:
                print(
                    "processing info : ",
                    self.control.frames,
                    self.control.bins,
                    samples_per_stripe,
                    bin_stride,
                )

            for p in np.arange(0, self.control.frames * 2, 2):

                sti_csd_data_coherence = np.zeros(
                    [self.control.num_fft, self.control.bins], np.float
                )
                sti_csd_data_phase = np.zeros(
                    [self.control.num_fft, self.control.bins], np.float
                )

                sti_times = np.zeros([self.control.bins], np.complex128)

                for b in np.arange(self.control.bins, dtype=np.int_):

                    if self.control.verbose:
                        print(
                            "read vector :",
                            self.channel,
                            start_sample,
                            samples_per_stripe,
                        )

                    xdata = self.dio[xidx].read_vector(
                        start_sample,
                        samples_per_stripe,
                        self.channel[xidx],
                        self.sub_channel[xidx],
                    )

                    ydata = self.dio[yidx].read_vector(
                        start_sample,
                        samples_per_stripe,
                        self.channel[yidx],
                        self.sub_channel[yidx],
                    )

                    if self.control.decimation > 1:
                        xdata = scipy.signal.decimate(xdata, self.control.decimation)
                        ydata = scipy.signal.decimate(ydata, self.control.decimation)
                        sample_freq = sr / self.control.decimation
                    else:
                        sample_freq = sr

                    if self.control.mean:
                        detrend_fn = matplotlib.mlab.detrend_mean
                    else:
                        detrend_fn = matplotlib.mlab.detrend_none

                    try:
                        csd_data, freq_axis = matplotlib.mlab.csd(
                            xdata,
                            ydata,
                            NFFT=self.control.num_fft,
                            Fs=float(sample_freq),
                            sides="default",
                            detrend=detrend_fn,
                            scale_by_freq=False,
                        )
                    except:
                        traceback.print_exc(file=sys.stdout)

                    sti_csd_data_coherence[:, b] = 10.0 * np.log10(
                        np.absolute(csd_data) + 1e-12
                    )
                    sti_csd_data_phase[:, b] = np.angle(csd_data)

                    sti_times[b] = start_sample / sr

                    start_sample += stripe_stride

                # Now Plot the Data
                ax = self.subplots[fidx][p]
                ax1 = self.subplots[fidx][p + 1]

                # determine image x-y extent
                extent = (
                    0,
                    self.control.bins,
                    -np.max(freq_axis) * 1.1 / 1e3,
                    np.max(freq_axis) * 1.1 / 1e3,
                )
                # determine image color extent in log scale units

                Pss = sti_csd_data_coherence
                Pss2 = sti_csd_data_phase
                vmin2 = -np.pi * 1.05
                vmax2 = np.pi * 1.05

                if self.control.zaxis:
                    vmin = int(self.control.zaxis.split(":")[0])
                    vmax = int(self.control.zaxis.split(":")[1])
                else:
                    med_Pss = np.nanmedian(Pss)
                    max_Pss = np.nanmax(Pss)
                    vmin = np.real(med_Pss - 6.0)
                    vmax = np.real(med_Pss + (max_Pss - med_Pss) * 0.61803398875 + 50.0)

                im = ax.imshow(
                    Pss,
                    cmap="jet",
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )
                im2 = ax1.imshow(
                    Pss2,
                    cmap="coolwarm",
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    vmin=vmin2,
                    vmax=vmax2,
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
                ax1.set_xticks(tick_spacing)
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
                ax1.set_xticklabels(tick_labels)

                # set the font sizes
                for tk in ax.get_xticklabels():
                    tk.set_size(8)
                for tk in ax.get_yticklabels():
                    tk.set_size(8)
                for tk in ax1.get_xticklabels():
                    tk.set_size(8)
                for tk in ax1.get_yticklabels():
                    tk.set_size(8)

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

            self.f[fidx].suptitle(
                "%s %s %4.2f MHz (%d,%d)"
                % (self.control.title, timestamp, xcfreq / 1e6, xidx, yidx),
                fontsize=8,
            )

            # ax.legend(fontsize=8)
            ax.set_xlabel("time (UTC)", fontsize=8)
            ax1.set_xlabel("time (UTC)", fontsize=8)
            # fixup ticks

            for tk in ax.get_xticklabels():
                tk.set_size(8)
            for tk in ax.get_yticklabels():
                tk.set_size(8)
            for tk in ax1.get_xticklabels():
                tk.set_size(8)
            for tk in ax1.get_yticklabels():
                tk.set_size(8)

            self.gridspec[fidx].update()
            print("show plot")
            self.f[fidx].tight_layout()

            self.f[fidx].subplots_adjust(top=0.95, right=0.88)
            cax = self.f[fidx].add_axes([0.9, 0.55, 0.015, 0.4])
            cax1 = self.f[fidx].add_axes([0.9, 0.10, 0.015, 0.4])
            self.f[fidx].colorbar(im, cax=cax)
            self.f[fidx].colorbar(im2, cax=cax1)
        matplotlib.pyplot.show()


def parse_command_line():
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
        action="append",
        help="Use data from the provided digital RF data <paths>. More than one required.",
    )
    parser.add_option(
        "-c",
        "--channel",
        dest="channel",
        action="append",
        help="Use data from the provided digital RF channel <channel>:<subchannel>. More than one required.",
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
        "-x",
        "--xtype",
        dest="xtype",
        default="pairs",
        type="string",
        help="Cross combination type : pairs, combo, self, permute",
    )
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print status messages to stdout.",
    )

    (options, args) = parser.parse_args()

    return (options, args)


#
# MAIN PROGRAM
#

# Setup Defaults
"""


"""

# Parse the Command Line for configuration
(options, args) = parse_command_line()

if options.path is None:
    print("Please provide an input source with the -p option!")
    sys.exit(1)

# Activate the DataPlotter
dpc = DataPlotter(options)

dpc.plot()
