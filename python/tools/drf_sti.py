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
import argparse
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


def intinttuple(s):
    """Get (int,int) tuple from int:int strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    else:
        return None


def strinttuple(s):
    """Get (string,int) tuple from str:int strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return parts[0], int(parts[1])
    else:
        return parts[0], None


class Extend(argparse.Action):
    """Action to split comma-separated arguments and add to a list."""

    def __init__(self, option_strings, dest, type=None, **kwargs):
        if type is not None:
            itemtype = type
        else:

            def itemtype(s):
                return s

        def split_string_and_cast(s):
            return [itemtype(a.strip()) for a in s.strip().split(",")]

        super(Extend, self).__init__(
            option_strings, dest, type=split_string_and_cast, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        cur_list = getattr(namespace, self.dest, [])
        if cur_list is None:
            cur_list = []
        cur_list.extend(values)
        setattr(namespace, self.dest, cur_list)


class DataPlotter(object):
    def __init__(self, opt):
        """Initialize a data plotter for STI plotting."""
        self.opt = opt

        # convert channel argument to separate tuples for channels and subchannels
        self.channels, self.subchannels = zip(*self.opt.channels)
        # replace None subchannels with 0
        self.subchannels = tuple(
            0 if subch is None else subch for subch in self.subchannels
        )

        # open digital RF path
        self.dio = drf.DigitalRFReader(self.opt.path)
        self.sr = self.dio.get_properties(self.channels[0])["samples_per_second"]

        self.bounds = self.dio.get_bounds(self.channels[0])
        self.dt_start = datetime.datetime.utcfromtimestamp(
            int(self.bounds[0] / self.sr)
        )
        self.dt_stop = datetime.datetime.utcfromtimestamp(int(self.bounds[1] / self.sr))

        print(
            "bound times {0} to {1} UTC".format(
                self.dt_start.utcnow().isoformat(), self.dt_stop.utcnow().isoformat()
            )
        )

        if self.opt.verbose:
            print("bound sample index {0}".format(self.bounds))

        # Figure setup

        self.f = matplotlib.pyplot.figure(
            figsize=(7, np.min([np.max([4, self.opt.frames]), 7])), dpi=128
        )

        self.gridspec = matplotlib.gridspec.GridSpec(self.opt.frames, 1)

        self.subplots = []

        """ Setup the subplots for this display """
        for n in np.arange(self.opt.frames):
            ax = self.f.add_subplot(self.gridspec[n])
            self.subplots.append(ax)

    def plot(self):
        """Iterate over the data set and plot the STI into the subplot panels.

        If multiple channels are provided they are either displayed successively
        or beamformed using the selected method and any phasings prior to a
        combined display.

        Each panel is divided into a provided number of bins of a given
        integration length. Strides between the panels are made between
        integrations.

        """
        # initialize outside the loop to avoid memory leak

        # initial plotting scales
        vmin = 0
        vmax = 0

        if self.opt.verbose:
            print("sample rate: {0}".format(self.sr))

            # initial time info
        b = self.bounds

        if self.opt.verbose:
            print("channel data bounds: {0}".format(b))

        if self.opt.start:
            dtst0 = dateutil.parser.parse(self.opt.start)
            st0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            st0 = int(st0 * self.sr)
        else:
            st0 = int(b[0])

        if self.opt.end:
            dtst0 = dateutil.parser.parse(self.opt.end)
            et0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()
            et0 = int(et0 * self.sr)
        else:
            et0 = int(b[1])

        if self.opt.verbose:
            print("start sample st0: {0}".format(st0))
            print("end sample et0: {0}".format(et0))

        blocks = self.opt.length * self.opt.frames

        samples_per_stripe = (
            self.opt.fft_bins * self.opt.integration * self.opt.decimation
        )
        total_samples = blocks * samples_per_stripe

        if total_samples > (et0 - st0):
            print(
                (
                    "Insufficient samples for {0} samples per stripe and {1} blocks"
                    " between {2} and {3}"
                ).format(samples_per_stripe, blocks, st0, et0)
            )
            return

        stripe_stride = (et0 - st0) / blocks

        bin_stride = stripe_stride / self.opt.length

        start_sample = st0

        print("first {0}".format(start_sample))

        # get metadata
        # this could be done better to ensure we catch frequency or sample rate
        # changes
        mdt = self.dio.read_metadata(st0, et0, self.channels[0])
        try:
            md = mdt[list(mdt.keys())[0]]
            cfreq = md["center_frequencies"].ravel()[int(self.subchannels[0])]
        except (IndexError, KeyError):
            cfreq = 0.0

        if self.opt.verbose:
            print(
                "processing info : {0} {1} {2} {3}".format(
                    self.opt.frames, self.opt.length, samples_per_stripe, bin_stride
                ),
            )

        for p in np.arange(self.opt.frames):
            sti_psd_data = np.zeros([self.opt.fft_bins, self.opt.length], np.float64)
            sti_times = np.zeros([self.opt.length], np.complex128)

            for b in np.arange(self.opt.length, dtype=np.int_):
                if self.opt.verbose:
                    print(
                        "read vector : {0} {1} {2}".format(
                            self.channels, start_sample, samples_per_stripe
                        )
                    )

                # if beamforming then load and beamform channels
                if self.opt.beamform:
                    if self.opt.verbose:
                        print("beamform: {0}".format(self.opt.beamform))

                    if self.opt.beamform == "sum":
                        if len(self.opt.phases) != len(self.channels):
                            print(
                                "Number of phases must match number of channels for"
                                " beamforming."
                            )

                        data = np.zeros([samples_per_stripe], np.complex128)

                        for idx, c in enumerate(self.channels):
                            if self.opt.verbose:
                                print("beamform sum channel {0}".format(c))

                            channel = c
                            subchannel = int(self.subchannels[idx])

                            try:
                                dv = self.dio.read_vector(
                                    start_sample,
                                    samples_per_stripe,
                                    channel,
                                    subchannel,
                                )
                            except Exception:
                                # handle data gaps better
                                dv = np.empty(samples_per_stripe, np.complex64)
                                dv[:] = np.nan

                            dv = dv * np.exp(
                                1j * np.deg2rad(float(self.opt.phases[idx]))
                            )

                        data = data + dv

                    else:
                        print(
                            "Unknown beamforming method {0}".format(self.opt.beamform)
                        )
                        return
                else:
                    if self.opt.verbose:
                        print("Using channel {0}".format(self.channels[0]))
                    channel = self.channels[0]
                    subchannel = int(self.subchannels[0])

                    try:
                        data = self.dio.read_vector(
                            start_sample, samples_per_stripe, channel, subchannel
                        )
                    except IOError:
                        if self.opt.verbose:
                            print(
                                "IO Error for channel {0}:{1} start sample {2}".format(
                                    channel,
                                    subchannel,
                                    start_sample,
                                )
                            )
                        # handle data gaps better
                        data = np.empty(samples_per_stripe, np.complex64)
                        data[:] = np.nan

                if self.opt.decimation > 1:
                    data = scipy.signal.decimate(data, self.opt.decimation)
                    sample_freq = self.sr / self.opt.decimation
                else:
                    sample_freq = self.sr

                if self.opt.mean:
                    detrend_fn = matplotlib.mlab.detrend_mean
                else:
                    detrend_fn = matplotlib.mlab.detrend_none

                try:
                    psd_data, freq_axis = matplotlib.mlab.psd(
                        data,
                        NFFT=self.opt.fft_bins,
                        Fs=float(sample_freq),
                        detrend=detrend_fn,
                        scale_by_freq=False,
                    )
                except Exception:
                    traceback.print_exc(file=sys.stdout)

                sti_psd_data[:, b] = np.real(10.0 * np.log10(np.abs(psd_data) + 1e-12))

                sti_times[b] = start_sample / self.sr

                start_sample += stripe_stride

            # Now Plot the Data
            ax = self.subplots[p]

            # determine image x-y extent
            extent = (
                0,
                self.opt.length,
                np.min(freq_axis) / 1e3,
                np.max(freq_axis) / 1e3,
            )

            # determine image color extent in log scale units
            Pss = sti_psd_data

            if self.opt.zaxis:
                vmin = self.opt.zaxis[0]
                vmax = self.opt.zaxis[1]
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
                self.opt.length / 8,
                self.opt.length,
                self.opt.length / 8,
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

        print("last {0}".format(start_sample))

        # create a time stamp
        start_time = int(st0 / self.sr)
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
            % (self.opt.title, timestamp, cfreq / 1e6, self.opt.path),
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
        if self.opt.outname:
            fname, ext = os.path.splitext(self.opt.outname)
            if ext == "":
                ext = ".png"
            print("Save plot as {0}".format(fname + ext))
            matplotlib.pyplot.savefig(fname + ext)
        if self.opt.appear or not self.opt.outname:
            print("Show plot")
            matplotlib.pyplot.show()


def parse_command_line():
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "drf_sti"
    copyright = "Copyright (c) 2022 Massachusetts Institute of Technology"
    shortdesc = "Spectral Time Intensity plotter for the DigitalRF format."
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format(copyright, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )

    parser = argparse.ArgumentParser(
        description=desc,
        prefix_chars="-",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="datadir_path",
        help="Path to data directory which has the DigitalRF channel subdirectories.",
    )
    parser.add_argument(
        "-t",
        "--title",
        dest="title",
        default="Digital RF Data",
        help="Use title provided for the data.",
    )
    parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        default=1,
        type=int,
        help="The number of sub-panel frames in the plot.",
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        default=None,
        help=(
            "Use the provided start time instead of the first time in the data."
            " format is ISO8601: 2015-11-01T15:24:00Z"
        ),
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        default=None,
        help=(
            "Use the provided end time for the plot."
            " format is ISO8601: 2015-11-01T15:24:00Z"
        ),
    )
    parser.add_argument(
        "-c",
        "--channel",
        dest="channels",
        action=Extend,
        type=strinttuple,
        default=[],
        help="""Input channel specification, including names and mapping from
                receiver channels. Multiple channels are only used for beamforming.
                Each input channel must be specified here
                using a unique name. Specifications are given as a receiver
                channel name and sub-channel pair, e.g. "ch0:0". The number and
                colon are optional; if omitted, the receive sub-channel is zero.
                (default: "ch0")""",
    )
    parser.add_argument(
        "-B",
        "--beamform",
        dest="beamform",
        default=None,
        choices=("sum",),
        help="Enable beamforming of multiple channels prior to plotting.",
    )
    parser.add_argument(
        "-p",
        "--phases",
        dest="phases",
        action=Extend,
        default=[],
        metavar="CHAN:PHASE_IN_DEG",
        help=(
            "Change the phase of Digital RF channel relative to other channels."
            " In degrees."
        ),
    )
    parser.add_argument(
        "-l",
        "--length",
        dest="length",
        default=1024,
        type=int,
        help="The number of time bins for the STI.",
    )
    parser.add_argument(
        "-b",
        "--fft_bins",
        dest="fft_bins",
        default=1024,
        type=int,
        help="The number of FFT bints for the STI.",
    )
    parser.add_argument(
        "-i",
        "--integration",
        dest="integration",
        default=1,
        type=int,
        help="The number of rasters to integrate for each plot.",
    )
    parser.add_argument(
        "-d",
        "--decimation",
        dest="decimation",
        default=1,
        type=int,
        help="The decimation factor for the data (integer).",
    )
    parser.add_argument(
        "-m",
        "--mean",
        dest="mean",
        action="store_true",
        default=False,
        help="Remove the mean from the data at the PSD processing step.",
    )
    parser.add_argument(
        "-z",
        "--zaxis",
        dest="zaxis",
        type=intinttuple,
        default=None,
        metavar="ZLOW:ZHIGH",
        help=(
            "zaxis colorbar setting e.g. -z=-50:50 ;"
            " = needed due to argparse issue with negative numbers"
        ),
    )
    parser.add_argument(
        "-o",
        "--outname",
        dest="outname",
        default=None,
        type=str,
        help="Name of file that figure will be saved under.",
    )
    parser.add_argument(
        "-a",
        "--appear",
        action="store_true",
        dest="appear",
        default=False,
        help="Makes the plot appear through pyplot show.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print status messages to stdout.",
    )

    options = parser.parse_args()

    return options


#
# MAIN PROGRAM
#

# Setup Defaults
if __name__ == "__main__":
    """
    Needed to add main function to use outside functions outside of module.
    """

    # Parse the Command Line for configuration
    options = parse_command_line()

    if options.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)

    if options.verbose:
        print("options: {0}".format(options))

    # Activate the DataPlotter
    dpc = DataPlotter(options)

    dpc.plot()
