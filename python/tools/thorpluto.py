#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2020 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Record data from the ADALM Pluto using GNURadio to Digital RF format."""
from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import re
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import dedent, fill, TextWrapper
import numpy as np
import pytz

import digital_rf as drf
import gr_digital_rf as gr_drf
from gnuradio import blocks
from gnuradio import filter as grfilter
from gnuradio import gr

try:
    from gnuradio import iio
except ImportError:
    import iio


def equiripple_lpf(cutoff=0.9, transition_width=0.2, attenuation=80, pass_ripple=None):
    """Get taps for an equiripple low-pass filter.

    All frequencies given must be normalized in the range [0, 1], with 1
    corresponding to the Nyquist frequency (Fs/2).

    Parameters
    ----------
    cutoff : float
        Normalized cutoff frequency (beginning of transition band).

    transition_width : float
        Normalized width (in frequency) of transition region from pass band to
        stop band.

    attenuation : float
        Attenuation of the stop band in dB.

    pass_ripple : float | None
        Maximum ripple in the pass band in dB. If None, the attenuation value
        is used.


    Returns
    -------
    taps : array_like
        Type I (even order) FIR low-pass filter taps meeting the given
        requirements.

    """
    if pass_ripple is None:
        pass_ripple = attenuation

    if cutoff <= 0:
        errstr = "Cutoff ({0}) must be strictly greater than zero."
        raise ValueError(errstr.format(cutoff))

    if transition_width <= 0:
        errstr = "Transition width ({0}) must be strictly greater than zero."
        raise ValueError(errstr.format(transition_width))

    if cutoff + transition_width >= 1:
        errstr = (
            "Cutoff ({0}) + transition width ({1}) must be strictly less than"
            " one, but it is {2}."
        ).format(cutoff, transition_width, cutoff + transition_width)
        raise ValueError(errstr)

    # pm_remez arguments
    bands = [0, cutoff, cutoff + transition_width, 1]
    ampl = [1, 1, 0, 0]
    error_weight = [10 ** ((pass_ripple - attenuation) / 20.0), 1]

    # get estimate for the filter order (Oppenheim + Schafer 2nd ed, 7.104)
    M = ((attenuation + pass_ripple) / 2.0 - 13) / 2.324 / (np.pi * transition_width)
    # round up to nearest even-order (Type I) filter
    M = int(np.ceil(M / 2.0)) * 2

    for _attempts in range(20):
        # get taps for order M
        try:
            taps = np.asarray(
                grfilter.pm_remez(
                    order=M, bands=bands, ampl=ampl, error_weight=error_weight
                )
            )
        except RuntimeError:
            M = M + 2
            continue

        # calculate frequency response and get error from ideal
        nfft = 16 * len(taps)
        h = np.fft.fft(taps, nfft)
        w = np.fft.fftfreq(nfft, 0.5)

        passband = h[(np.abs(w) >= bands[0]) & (np.abs(w) <= bands[1])]
        stopband = h[(np.abs(w) >= bands[2]) & (np.abs(w) <= bands[3])]

        act_ripple = -20 * np.log10(np.max(np.abs(ampl[0] - np.abs(passband))))
        act_atten = -20 * np.log10(np.max(np.abs(ampl[2] - np.abs(stopband))))

        if act_ripple >= pass_ripple and act_atten >= attenuation:
            break
        else:
            M = M + 2
    else:
        errstr = (
            "Could not calculate equiripple filter that meets requirements"
            "after {0} attempts (final order {1})."
        )
        raise RuntimeError(errstr.format(_attempts, M))

    return taps


class Thorpluto(object):
    """Record data from osmo compatible devices devices in DigitalRF format."""

    def __init__(self, datadir, **kwargs):
        options = dict(
            verbose=True,
            # mainboard group (num: len of mboards)
            mboards=[],
            clock_rates=[None],
            clock_sources=[""],
            time_sources=[""],
            # receiver group (apply to all)
            samplerate=1e6,
            buff_size=0x8000,
            time_sync=True,
            wait_for_lock=True,
            stop_on_dropped=False,
            realtime=False,
            test_settings=True,
            # receiver ch. group (num: matching channels from mboards)
            centerfreqs=[2.4e9],
            quad_track=[False],
            rfdc_track=[False],
            bbdc_track=[False],
            gains=[73],
            bandwidths=[20e6],
            # output channel group (num: len of channel_names)
            channel_names=["ch0"],
            channels=[None],
            ch_samplerates=[None],
            ch_centerfreqs=[False],
            ch_scalings=[1.0],
            ch_nsubchannels=[1],
            ch_lpf_cutoffs=[0.9],
            ch_lpf_transition_widths=[0.2],
            ch_lpf_attenuations=[80.0],
            ch_lpf_pass_ripples=[None],
            ch_out_types=[None],
            # digital_rf group (apply to all)
            file_cadence_ms=1000,
            subdir_cadence_s=3600,
            metadata={},
            uuid=None,
        )
        options.update(kwargs)
        op = self._parse_options(datadir=datadir, **options)
        self.op = op

        # test device settings, release device when done
        if op.test_settings:
            if op.verbose:
                print("Initialization: testing device settings.")
            self._pluto_setup()

            # finalize options (for settings that depend on USRP setup)
            self._finalize_options()

    @staticmethod
    def _parse_options(**kwargs):
        """Put all keyword options in a namespace and normalize them."""
        op = argparse.Namespace(**kwargs)

        # get USRP cpu_format based on output type and decimation requirements
        processing_required = (
            any(sr is not None for sr in op.ch_samplerates)
            or any(cf is not False for cf in op.ch_centerfreqs)
            or any(s != 1 for s in op.ch_scalings)
            or any(nsch != 1 for nsch in op.ch_nsubchannels)
        )
        if (
            all(ot is None or ot == "sc16" for ot in op.ch_out_types)
            and not processing_required
        ):
            # with only sc16 output and no processing, can use sc16 as cpu
            # format and disable conversion
            op.cpu_format = "sc16"
            op.ch_out_specs = [
                dict(
                    convert=None,
                    convert_kwargs=None,
                    dtype=np.dtype([(str("r"), np.int16), (str("i"), np.int16)]),
                    name="sc16",
                )
            ]
        else:
            op.cpu_format = "fc32"
            # get full specification for output types
            supported_out_types = {
                "sc8": dict(
                    convert="float_to_char",
                    convert_kwargs=dict(vlen=2, scale=float(2**7 - 1)),
                    dtype=np.dtype([(str("r"), np.int8), (str("i"), np.int8)]),
                    name="sc8",
                ),
                "sc16": dict(
                    convert="float_to_short",
                    convert_kwargs=dict(vlen=2, scale=float(2**15 - 1)),
                    dtype=np.dtype([(str("r"), np.int16), (str("i"), np.int16)]),
                    name="sc16",
                ),
                "sc32": dict(
                    convert="float_to_int",
                    convert_kwargs=dict(vlen=2, scale=float(2**31 - 1)),
                    dtype=np.dtype([(str("r"), np.int32), (str("i"), np.int32)]),
                    name="sc32",
                ),
                "fc32": dict(
                    convert=None,
                    convert_kwargs=None,
                    dtype=np.dtype("complex64"),
                    name="fc32",
                ),
            }
            supported_out_types[None] = supported_out_types["fc32"]
            type_dicts = []
            for ot in op.ch_out_types:
                try:
                    type_dict = supported_out_types[ot]
                except KeyError:
                    errstr = (
                        "Output type {0} is not supported. Must be one of {1}."
                    ).format(ot, list(supported_out_types.keys()))
                    raise ValueError(errstr)
                else:
                    type_dicts.append(type_dict)
            op.ch_out_specs = type_dicts
        # replace out_types to fill in None values with type name
        op.ch_out_types = [os["name"] for os in op.ch_out_specs]
        # repeat mainboard arguments as necessary
        op.nmboards = len(op.mboards) if len(op.mboards) > 0 else 1
        for mb_arg in ("clock_rates", "clock_sources", "time_sources"):
            val = getattr(op, mb_arg)
            mbval = list(islice(cycle(val), 0, op.nmboards))
            setattr(op, mb_arg, mbval)

        # get number of receiver channels by total number of subdevices over
        # all mainboards
        op.mboards_bychan = []
        op.mboardnum_bychan = []
        # add a default mboard if the argument was empty
        op.mboards = op.mboards if op.mboards else [""]
        for mbnum, mb in enumerate(op.mboards):
            op.mboards_bychan.extend([mb])
            op.mboardnum_bychan.extend([mbnum])
        op.nrchs = op.nmboards
        # repeat receiver channel arguments as necessary
        for rch_arg in (
            "bandwidths",
            "centerfreqs",
            "quad_track",
            "rfdc_track",
            "bbdc_track",
            "gains",
        ):
            val = getattr(op, rch_arg)
            rval = list(islice(cycle(val), 0, op.nrchs))
            setattr(op, rch_arg, rval)

        # repeat output channel arguments as necessary
        op.nochs = len(op.channel_names)
        for och_arg in (
            "channels",
            "ch_centerfreqs",
            "ch_lpf_attenuations",
            "ch_lpf_cutoffs",
            "ch_lpf_pass_ripples",
            "ch_lpf_transition_widths",
            "ch_nsubchannels",
            "ch_out_specs",
            "ch_out_types",
            "ch_samplerates",
            "ch_scalings",
        ):
            val = getattr(op, och_arg)
            rval = list(islice(cycle(val), 0, op.nochs))
            setattr(op, och_arg, rval)

        # fill in unspecified (None) channels values
        rchannels = set(range(op.nrchs))
        ochannels = set(c for c in op.channels if c is not None)
        if not ochannels.issubset(rchannels):
            errstr = (
                "Invalid channel specification. Output channel uses"
                " non-existent receiver channel: {0}."
            )
            raise ValueError(errstr.format(list(ochannels - rchannels)))
        avail = sorted(rchannels - ochannels)
        try:
            op.channels = [c if c is not None else avail.pop(0) for c in op.channels]
        except IndexError:
            errstr = (
                "No remaining receiver channels left to assign to unspecified"
                " (None) output channel. You probably need to explicitly"
                " specify the receiver channels to output."
            )
            raise ValueError(errstr)
        unused_rchs = set(range(op.nrchs)) - set(op.channels)
        if unused_rchs:
            errstr = (
                "Receiver channels {0} are unused in the output. Either"
                " remove them from the mainboard/subdevice specification or"
                " correct the output channel specification."
            )
            raise ValueError(errstr.format(unused_rchs))

        # copy desired centerfreq from receiver to output channel if requested
        op.ch_centerfreqs = [
            op.centerfreqs[rch] if f in (None, True) else f
            for f, rch in zip(op.ch_centerfreqs, op.channels)
        ]

        # create device_addr string to identify the requested device(s)
        op.mboard_strs = []
        for n, mb in enumerate(op.mboards):
            if re.match(r"[^0-9]+:.+", mb):
                idtype, mb = mb.split(":")
            elif re.match(r"[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}", mb):
                idtype = "usb"
            elif re.match(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", mb):
                idtype = "ip"
            else:
                # no idtype specified, not a format that we recognize, so just pass it
                # (this will catch empty string for using default device)
                op.mboard_strs.append(mb)
                continue
            if len(op.mboards) == 1:
                # do not use identifier numbering if only using one mainboard
                s = "{type}:{mb}".format(type=idtype, mb=mb.strip())
            else:
                s = "{type}{n}:{mb}".format(type=idtype, n=n, mb=mb.strip())
            op.mboard_strs.append(s)

        if op.verbose:
            opstr = (
                dedent(
                    """\
                Main boards: {mboard_strs}
                Clock rates: {clock_rates}
                Clock sources: {clock_sources}
                Time sources: {time_sources}
                Sample rate: {samplerate}
                Buffer Size: {buff_size}
                Bandwidth: {bandwidths}
                Frequency: {centerfreqs}
                Gain: {gains}
                Quadrature Tracking: {quad_track}
                RF DC Tracking: {rfdc_track}
                Baseband DC Tracking: {bbdc_track}
                Output channels: {channels}
                Output channel names: {channel_names}
                Output sample rate: {ch_samplerates}
                Output frequency: {ch_centerfreqs}
                Output scaling: {ch_scalings}
                Output subchannels: {ch_nsubchannels}
                Output type: {ch_out_types}
                Data dir: {datadir}
                Metadata: {metadata}
                UUID: {uuid}
            """
                )
                .strip()
                .format(**op.__dict__)
            )
            print(opstr)

        return op

    def _pluto_setup(self):
        """Create, set up, and return a dictionary of RTL source objects."""
        op = self.op
        op.otw_format = "sc32"

        pluto_sources = {}
        # Possible gain settings for RTL Dongle
        # sup_gs = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7,
        #           20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2,
        #           42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
        # List of possible sample rates for RTL dongle. Here for reference for now.
        # sup_sr = [1.024e6, 1.4e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.56e6]

        for mnum in range(op.nmboards):
            # HACK two questions
            # 1 Do I need to use the pluto source or you use the iio device source?
            # ans: Can use fmcomms2_source
            """make(std::string const & uri, unsigned long long frequency,
            unsigned long samplerate, unsigned long bandwidth, bool ch1_en,
            bool ch2_en, bool ch3_en, bool ch4_en, unsigned long buffer_size,
            bool quadrature, bool rfdc, bool bbdc, char const * gain1,
            double gain1_value, char const * gain2, double gain2_value,
            char const * rf_port_select, char const * filter, bool auto_filter=True)
            """
            pluto_sources[mnum] = iio.pluto_source(
                uri=op.mboard_strs[mnum],
                frequency=int(op.centerfreqs[mnum]),
                samplerate=int(op.samplerate),
                bandwidth=int(op.bandwidths[mnum]),
                buffer_size=0x8000,
                quadrature=op.quad_track[mnum],
                rfdc=op.rfdc_track[mnum],
                bbdc=op.bbdc_track[mnum],
                gain="manual" if op.gains[mnum] >= 0 else "fast_attack",
                gain_value=op.gains[mnum],
                filter="",
                auto_filter=True,
            )

            # set master clock rate
            # HACK can you set clock rates
            # clock_rate = op.clock_rates[mnum]
            # if clock_rate is not None:
            #     osmo_sources[mnum].set_clock_rate(clock_rate, 0)
            # op.clock_rates[mnum] = osmo_sources[mnum].get_clock_rate()
        # HACK need to see if I can get the sample rates out of source block
        samplerate = int(op.samplerate)
        # calculate longdouble precision/rational sample rate
        # (integer division of clock rate)
        # HACK Get clock rate too
        # cr = osmo_sources[0].get_clock_rate(0)

        op.samplerate = samplerate
        op.samplerate_frac = Fraction(samplerate).limit_denominator()
        return pluto_sources

    def _finalize_options(self):
        op = self.op

        op.ch_samplerates_frac = []
        op.resampling_ratios = []
        op.resampling_filter_taps = []
        op.resampling_filter_delays = []
        op.channelizer_filter_taps = []
        op.channelizer_filter_delays = []

        for ko, (osr, nsc) in enumerate(zip(op.ch_samplerates, op.ch_nsubchannels)):
            # get output sample rate fraction
            # (op.samplerate_frac final value is set in _pluto_setup
            #  so can't get output sample rate until after that is done)
            if osr is None:
                ch_samplerate_frac = op.samplerate_frac
            else:
                ch_samplerate_frac = Fraction(osr).limit_denominator()
            op.ch_samplerates_frac.append(ch_samplerate_frac)

            # get resampling ratio
            ratio = ch_samplerate_frac / op.samplerate_frac
            op.resampling_ratios.append(ratio)

            # get resampling low-pass filter taps
            if ratio == 1:
                op.resampling_filter_taps.append(np.zeros(0))
                op.resampling_filter_delays.append(0)
            else:
                taps = equiripple_lpf(
                    cutoff=float(op.ch_lpf_cutoffs[ko] * ratio),
                    transition_width=float(op.ch_lpf_transition_widths[ko] * ratio),
                    attenuation=op.ch_lpf_attenuations[ko],
                    pass_ripple=op.ch_lpf_pass_ripples[ko],
                )
                op.resampling_filter_taps.append(taps)
                op.resampling_filter_delays.append((len(taps) - 1) // 2)

            # get channelizer low-pass filter taps
            if nsc > 1:
                taps = equiripple_lpf(
                    cutoff=(op.ch_lpf_cutoffs[ko] / nsc),
                    transition_width=(op.ch_lpf_transition_widths[ko] / nsc),
                    attenuation=op.ch_lpf_attenuations[ko],
                    pass_ripple=op.ch_lpf_pass_ripples[ko],
                )
                op.channelizer_filter_taps.append(taps)
                op.channelizer_filter_delays.append((len(taps) - 1) // 2)
            else:
                op.channelizer_filter_taps.append(np.zeros(0))
                op.channelizer_filter_delays.append(0)

    def run(self, starttime=None, endtime=None, duration=None, period=10):
        op = self.op

        # window in seconds that we allow for setup time so that we don't
        # issue a start command that's in the past when the flowgraph starts
        SETUP_TIME = 10

        # print current time and NTP status

        if op.verbose and sys.platform.startswith("linux"):
            try:
                call(("timedatectl", "status"))
            except OSError:
                # no timedatectl command, ignore
                pass

        # parse time arguments
        st = drf.util.parse_identifier_to_time(starttime)
        if st is not None:
            # find next suitable start time by cycle repeat period
            now = datetime.utcnow()
            now = now.replace(tzinfo=pytz.utc)
            soon = now + timedelta(seconds=SETUP_TIME)
            diff = max(soon - st, timedelta(0)).total_seconds()
            periods_until_next = (diff - 1) // period + 1
            st = st + timedelta(seconds=periods_until_next * period)

            if op.verbose:
                ststr = st.strftime("%a %b %d %H:%M:%S %Y")
                stts = (st - drf.util.epoch).total_seconds()
                print("Start time: {0} ({1})".format(ststr, stts))

        et = drf.util.parse_identifier_to_time(endtime, ref_datetime=st)
        if et is not None:
            if op.verbose:
                etstr = et.strftime("%a %b %d %H:%M:%S %Y")
                etts = (et - drf.util.epoch).total_seconds()
                print("End time: {0} ({1})".format(etstr, etts))

            if (
                et
                < (pytz.utc.localize(datetime.utcnow()) + timedelta(seconds=SETUP_TIME))
            ) or (st is not None and et <= st):
                raise ValueError("End time is before launch time!")

        if op.realtime:
            r = gr.enable_realtime_scheduling()

            if op.verbose:
                if r == gr.RT_OK:
                    print("Realtime scheduling enabled")
                else:
                    print("Note: failed to enable realtime scheduling")

        # create data directory so ringbuffer code can be started while waiting
        # to launch
        if not os.path.isdir(op.datadir):
            os.makedirs(op.datadir)

        # wait for the start time if it is not past
        while (st is not None) and (
            (st - pytz.utc.localize(datetime.utcnow())) > timedelta(seconds=SETUP_TIME)
        ):
            ttl = int((st - pytz.utc.localize(datetime.utcnow())).total_seconds())
            if (ttl % 10) == 0:
                print("Standby {0} s remaining...".format(ttl))
                sys.stdout.flush()
            time.sleep(1)

        # get pluto sources

        pluto_dict = self._pluto_setup()

        # finalize options (for settings that depend on setup function)
        self._finalize_options()

        # set device time
        tt = time.time()
        # HACK doesn't seem to have software way of setting device time
        # if op.time_sync:
        #     # wait until time 0.2 to 0.5 past full second, then latch
        #     # we have to trust NTP to be 0.2 s accurate
        #     while tt - math.floor(tt) < 0.2 or tt - math.floor(tt) > 0.3:
        #         time.sleep(0.01)
        #         tt = time.time()
        #     if op.verbose:
        #
        #         print("Latching at " + str(tt))
        #     # waits for the next pps to happen
        #     # (at time math.ceil(tt))
        #     # then sets the time for the subsequent pps
        #     # (at time math.ceil(tt) + 1.0)
        #     for ichan in pluto_dict.keys():
        #         pluto_chan = pluto_dict[ichan]
        #             pluto_chan.set_time_unknown_pps(osmosdr.time_spec_t(math.ceil(tt) + 1.0))
        # else:
        #     for ichan in pluto_dict.keys():
        #         pluto_chan = pluto_dict[ichan]
        #         pluto_chan.set_time_now(iio.time_spec_t(tt))

        # set launch time
        # (at least 2 seconds out so USRP start time can be set properly and
        #  there is time to set up flowgraph)
        if st is not None:
            lt = st
        else:
            now = pytz.utc.localize(datetime.utcnow())
            # launch on integer second by default for convenience (ceil + 2)
            lt = now.replace(microsecond=0) + timedelta(seconds=3)
        ltts = (lt - drf.util.epoch).total_seconds()
        # adjust launch time forward so it falls on an exact sample since epoch
        lt_rsamples = int(np.ceil(ltts * op.samplerate))
        ltts = lt_rsamples / op.samplerate
        lt = drf.util.sample_to_datetime(lt_rsamples, op.samplerate)
        if op.verbose:
            ltstr = lt.strftime("%a %b %d %H:%M:%S.%f %Y")
            msg = "Launch time: {0} ({1})\nSample index: {2}"
            print(msg.format(ltstr, repr(ltts), lt_rsamples))

        # populate flowgraph one channel at a time
        fg = gr.top_block()
        # store the graph so that the blocks are not garbage collected
        graph = []
        for ko in range(op.nochs):
            # receiver channel number corresponding to this output channel
            kr = op.channels[ko]
            # mainboard number corresponding to this receiver's channel
            mbnum = op.mboardnum_bychan[kr]
            pluto_chan = pluto_dict[mbnum]

            # output settings that get modified depending on processing
            ch_samplerate_frac = op.ch_samplerates_frac[ko]
            ch_centerfreq = op.ch_centerfreqs[ko]
            start_sample_adjust = 0
            # make resampling filter blocks if necessary
            rs_ratio = op.resampling_ratios[ko]
            scaling = op.ch_scalings[ko]
            if rs_ratio != 1:
                rs_taps = op.resampling_filter_taps[ko]

                # integrate scaling into filter taps
                rs_taps *= scaling
                conv_scaling = 1.0

                # frequency shift filter taps to band-pass if necessary
                if ch_centerfreq is not False:
                    f_shift = ch_centerfreq - op.centerfreqs[kr]
                    phase_inc = 2 * np.pi * f_shift / op.samplerate
                    rotator = np.exp(phase_inc * 1j * np.arange(len(rs_taps)))
                    rs_taps = (rs_taps * rotator).astype("complex64")

                    # create band-pass filter (complex taps)
                    # don't use rational_resampler because its delay is wrong
                    resampler = grfilter.pfb_arb_resampler_ccc(
                        rate=float(rs_ratio),
                        taps=rs_taps.tolist(),
                        # since resampling is rational, we know we only need a
                        # number of filters equal to the interpolation factor
                        filter_size=rs_ratio.numerator,
                    )
                else:
                    # create low-pass filter (float taps)
                    # don't use rational_resampler because its delay is wrong
                    resampler = grfilter.pfb_arb_resampler_ccf(
                        rate=float(rs_ratio),
                        taps=rs_taps.tolist(),
                        # since resampling is rational, we know we only need a
                        # number of filters equal to the interpolation factor
                        filter_size=rs_ratio.numerator,
                    )

                # declare sample delay for the filter block so that tags are
                # propagated to the correct sample
                resampler.declare_sample_delay(int(op.resampling_filter_delays[ko]))

                # adjust start sample to account for filter delay so first
                # sample going to output is shifted to an earlier time
                # (adjustment is in terms of filter output samples, so need to
                #  take the input filter delay and account for the output rate)
                start_sample_adjust = int(
                    (start_sample_adjust - op.resampling_filter_delays[ko]) * rs_ratio
                )
            else:
                conv_scaling = scaling
                resampler = None

            # make frequency shift block if necessary
            if ch_centerfreq is not False:
                f_shift = ch_centerfreq - op.centerfreqs[kr]
                phase_inc = -2 * np.pi * f_shift / ch_samplerate_frac
                rotator = blocks.rotator_cc(phase_inc)
            else:
                ch_centerfreq = op.centerfreqs[kr]
                rotator = None

            # make channelizer if necessary
            nsc = op.ch_nsubchannels[ko]
            if nsc > 1:
                sc_taps = op.channelizer_filter_taps[ko]

                # build a hierarchical block for the channelizer so output
                # is a vector of channels as expected by digital_rf
                channelizer = gr.hier_block2(
                    "lpf",
                    gr.io_signature(1, 1, gr.sizeof_gr_complex),
                    gr.io_signature(1, 1, nsc * gr.sizeof_gr_complex),
                )
                s2ss = blocks.stream_to_streams(gr.sizeof_gr_complex, nsc)
                filt = grfilter.pfb_channelizer_ccf(
                    numchans=nsc, taps=sc_taps, oversample_rate=1.0
                )
                s2v = blocks.streams_to_vector(gr.sizeof_gr_complex, nsc)
                channelizer.connect(channelizer, s2ss)
                for ksc in range(nsc):
                    channelizer.connect((s2ss, ksc), (filt, ksc), (s2v, ksc))
                channelizer.connect(s2v, channelizer)

                # declare sample delay for the filter block so that tags are
                # propagated to the correct sample
                # (for channelized, delay is applied for each filter in the
                #  polyphase bank, so this needs to be the output sample delay)
                filt.declare_sample_delay(int(op.channelizer_filter_delays[ko] / nsc))

                # adjust start sample to account for filter delay so first
                # sample going to output is shifted to an earlier time
                # (adjustment is in terms of filter output samples, so need to
                #  take the input filter delay and account for the output rate)
                start_sample_adjust = int(
                    (start_sample_adjust - op.channelizer_filter_delays[ko]) / nsc
                )

                # modify output settings accordingly
                ch_centerfreq = ch_centerfreq + np.fft.fftfreq(
                    nsc, 1 / float(ch_samplerate_frac)
                )
                ch_samplerate_frac = ch_samplerate_frac / nsc
            else:
                channelizer = None

            # make conversion block if necessary
            ot_dict = op.ch_out_specs[ko]
            converter = ot_dict["convert"]
            if converter is not None:
                kw = ot_dict["convert_kwargs"]
                # increase vector length of input due to channelizer
                kw["vlen"] *= nsc
                # incorporate any scaling into type conversion block
                kw["scale"] *= conv_scaling
                convert = getattr(blocks, converter)(**kw)
            elif conv_scaling != 1:
                convert = blocks.multiply_const_cc(conv_scaling, nsc)
            else:
                convert = None

            # get start sample
            ch_samplerate_ld = np.longdouble(
                ch_samplerate_frac.numerator
            ) / np.longdouble(ch_samplerate_frac.denominator)
            start_sample = int(np.uint64(ltts * ch_samplerate_ld))
            # create digital RF sink
            dst = gr_drf.digital_rf_channel_sink(
                channel_dir=os.path.join(op.datadir, op.channel_names[ko]),
                dtype=np.complex64,
                subdir_cadence_secs=op.subdir_cadence_s,
                file_cadence_millisecs=op.file_cadence_ms,
                sample_rate_numerator=ch_samplerate_frac.numerator,
                sample_rate_denominator=ch_samplerate_frac.denominator,
                start=start_sample,
                ignore_tags=False,
                is_complex=True,
                num_subchannels=nsc,
                uuid_str=op.uuid,
                center_frequencies=ch_centerfreq,
                metadata=dict(
                    # receiver metadata for USRP
                    receiver=dict(
                        description="ADALM-PLUTO Recording using GNU Radio block",
                        bandwidth=op.bandwidths[kr],
                        center_freq=op.centerfreqs[kr],
                        clock_rate=op.clock_rates[mbnum],
                        clock_source=op.clock_sources[mbnum],
                        quad_track=op.quad_track[mbnum],
                        rfdc_track=op.rfdc_track[mbnum],
                        bbdc_track=op.bbdc_track[mbnum],
                        gain=op.gains[kr],
                        id=op.mboards_bychan[kr],
                        otw_format=op.otw_format,
                        samp_rate=op.samplerate,  # need to get sample ratertl_chan.get_sample_rate(),
                        time_source=op.time_sources[mbnum],
                    ),
                    processing=dict(
                        channelizer_filter_taps=op.channelizer_filter_taps[ko],
                        decimation=op.resampling_ratios[ko].denominator,
                        interpolation=op.resampling_ratios[ko].numerator,
                        resampling_filter_taps=op.resampling_filter_taps[ko],
                        scaling=op.ch_scalings[ko],
                    ),
                    **op.metadata
                ),
                is_continuous=True,
                compression_level=0,
                checksum=False,
                marching_periods=True,
                stop_on_skipped=op.stop_on_dropped,
                debug=op.verbose,
            )
            connections = [(pluto_chan, 0)]
            if resampler is not None:
                connections.append((resampler, 0))
            if rotator is not None:
                connections.append((rotator, 0))
            if channelizer is not None:
                connections.append((channelizer, 0))
            if convert is not None:
                connections.append((convert, 0))
            connections.append((dst, 0))
            connections = tuple(connections)
            # make channel connections in flowgraph
            fg.connect(*connections)

            # store the graph so that the blocks are not garbage collected
            graph.append(connections)

        # start the flowgraph once we are near the launch time
        while (lt - pytz.utc.localize(datetime.utcnow())) > timedelta(seconds=0.5):
            time.sleep(0.1)

        # start the flowgraph, samples should start at launch time
        fg.start()

        if et is None and duration is not None:
            et = lt + timedelta(seconds=duration)
        try:
            if et is None:
                fg.wait()
            else:
                # sleep until end time
                while pytz.utc.localize(datetime.utcnow()) < et - timedelta(seconds=2):
                    time.sleep(1)
                else:
                    # (actually a little after to allow for inexact computer time)
                    while pytz.utc.localize(datetime.utcnow()) < et + timedelta(
                        seconds=0.2
                    ):
                        time.sleep(0.1)
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        finally:
            fg.stop()
            # need to wait for the flowgraph to clean up, otherwise it won't
            # exit and we will wait forever after finishing this process
            fg.wait()
            print("done")
            sys.stdout.flush()


def evalint(s):
    """Evaluate string to an integer."""
    return int(eval(s, {}, {}))


def evalfloat(s):
    """Evaluate string to a float."""
    return float(eval(s, {}, {}))


def intstrtuple(s):
    """Get (int, string) tuple from int:str strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    else:
        return None, parts[0]


def noneorstr(s):
    """Turn empty or 'none' string to None."""
    if s.lower() in ("", "none"):
        return None
    else:
        return s


def noneorfloat(s):
    """Turn empty or 'none' to None, else evaluate to float."""
    if s.lower() in ("", "none"):
        return None
    else:
        return evalfloat(s)


def noneorbool(s):
    """Turn empty or 'none' string to None, all others to boolean."""
    if s.lower() in ("", "none"):
        return None
    elif s.lower() in ("true", "t", "yes", "y", "1"):
        return True
    else:
        return False


def noneorboolorfloat(s):
    """Turn empty or 'none' to None, else evaluate to a boolean or float."""
    if s.lower() in ("", "none"):
        return None
    elif s.lower() in ("auto", "true", "t", "yes", "y"):
        return True
    elif s.lower() in ("false", "f", "no", "n"):
        return False
    else:
        return evalfloat(s)


def noneorboolorcomplex(s):
    """Turn empty or 'none' to None, else evaluate to a boolean or complex."""
    if s.lower() in ("", "none"):
        return None
    elif s.lower() in ("auto", "true", "t", "yes", "y"):
        return True
    elif s.lower() in ("false", "f", "no", "n"):
        return False
    else:
        return complex(eval(s, {}, {}))


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


def _add_dir_group(parser):
    dirgroup = parser.add_mutually_exclusive_group(required=True)
    dirgroup.add_argument(
        "datadir",
        nargs="?",
        default=None,
        help="""Data directory, to be filled with channel subdirectories.""",
    )
    dirgroup.add_argument(
        "-o",
        "--out",
        dest="outdir",
        default=None,
        help="""Data directory, to be filled with channel subdirectories.""",
    )
    return parser


def _add_mainboard_group(parser):
    mbgroup = parser.add_argument_group(title="mainboard")
    mbgroup.add_argument(
        "-t",
        "--radiotype",
        dest="radtype",
        help="""The type of radio that is used for recording.""",
    )
    mbgroup.add_argument(
        "-m",
        "--mainboard",
        dest="mboards",
        action=Extend,
        help="""Mainboard address. (default: first device found)""",
    )

    mbgroup.add_argument(
        "--clock_rate",
        dest="clock_rates",
        action=Extend,
        type=noneorfloat,
        help="""Master clock rate for mainboard. Can be 'None'/'' to use
                device default or a value in Hz. (default: None)""",
    )
    mbgroup.add_argument(
        "--clock_source",
        dest="clock_sources",
        action=Extend,
        type=noneorstr,
        help="""Clock source (i.e. 10 MHz REF) for mainboard. Can be 'None'/''
                to use default (do not set if --nolock, otherwise 'external')
                or a string like 'external' or 'internal'. (default: '')""",
    )
    mbgroup.add_argument(
        "--time_source",
        dest="time_sources",
        action=Extend,
        type=noneorstr,
        help="""Time source (i.e. PPS) for mainboard. Can be 'None'/''
                to use default (do not set if --nosync, otherwise 'external')
                or a string like 'external' or 'internal'. (default: '')""",
    )
    return parser


def _add_receiver_group(parser):
    recgroup = parser.add_argument_group(title="receiver")
    recgroup.add_argument(
        "-r",
        "--samplerate",
        dest="samplerate",
        type=evalfloat,
        help="""Sample rate in Hz. (default: 1e6)""",
    )
    recgroup.add_argument(
        "-d",
        "--buffersize",
        dest="buff_size",
        type=evalint,
        help="""Size of buffer for the pluto.""",
    )
    # recgroup.add_argument(
    #     "-A",
    #     "--devargs",
    #     dest="dev_args",
    #     action=Extend,
    #     help="""Device arguments, e.g. "master_clock_rate=30e6".
    #             (default: 'recv_buff_size=100000000,num_recv_frames=512')""",
    # )

    recgroup.add_argument(
        "--stop_on_dropped",
        dest="stop_on_dropped",
        action="store_true",
        help="""Stop on dropped packet. (default: %(default)s)""",
    )
    recgroup.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        help="""Enable realtime scheduling if possible.
                (default: %(default)s)""",
    )
    recgroup.add_argument(
        "--notest",
        dest="test_settings",
        action="store_false",
        help="""Do not test USRP settings until experiment start.
                (default: False)""",
    )
    return parser


def _add_rchannel_group(parser):
    chgroup = parser.add_argument_group(title="receiver channel")
    chgroup.add_argument(
        "-f",
        "--centerfreq",
        dest="centerfreqs",
        action=Extend,
        type=evalfloat,
        help="""Center frequency in Hz. (default: 2.4e9)""",
    )
    chgroup.add_argument(
        "--quad_track",
        dest="quad_track",
        action=Extend,
        type=noneorboolorcomplex,
        help="""Enable quadrature DC tracking correction (default: False)""",
    )
    chgroup.add_argument(
        "--rfdc_track",
        dest="rfdc_track",
        action=Extend,
        type=noneorboolorcomplex,
        help="""Enable RF DC tracking correction (default: False)""",
    )
    chgroup.add_argument(
        "--bbdc_track",
        dest="bbdc_track",
        action=Extend,
        type=noneorboolorcomplex,
        help="""Enable baseband DC tracking correction (default: False)""",
    )

    chgroup.add_argument(
        "-g",
        "--gain",
        dest="gains",
        action=Extend,
        type=evalfloat,
        help="""Gain in dB. (default: 73)""",
    )
    chgroup.add_argument(
        "-b",
        "--bandwidth",
        dest="bandwidths",
        action=Extend,
        type=evalfloat,
        help="""Frontend bandwidth in Hz. (default: 20e6)""",
    )

    return parser


def _add_ochannel_group(parser):
    chgroup = parser.add_argument_group(title="output channel")
    chgroup.add_argument(
        "+c",
        "-c",
        "--channel",
        dest="chs",
        action=Extend,
        type=intstrtuple,
        help="""Output channel specification, including names and mapping from
                receiver channels. Each output channel must be specified here
                and given a unique name. Specifications are given as a receiver
                channel number and name pair, e.g. "0:ch0". The number and
                colon are optional; if omitted, any unused receiver channels
                will be assigned to output channels in the supplied name order.
                (default: "ch0")""",
    )
    chgroup.add_argument(
        "+r",
        "--ch_samplerate",
        dest="ch_samplerates",
        action=Extend,
        type=noneorfloat,
        help="""Output channel sample rate in Hz. If 'None'/'', use the
                receiver sample rate. Filtering and resampling will be
                performed to achieve the desired rate (set filter specs with
                lpf_* options). Must be less than or equal to the receiver
                sample rate. (default: None)""",
    )
    # deprecated by ch_samplerate in 2.6
    # if used, all ch_samplerate arguments will be ignored
    chgroup.add_argument(
        "-i",
        "--dec",
        dest="decimations",
        action=Extend,
        type=evalint,
        help=argparse.SUPPRESS,
    )
    chgroup.add_argument(
        "+f",
        "--ch_centerfreq",
        dest="ch_centerfreqs",
        action=Extend,
        type=noneorboolorfloat,
        help="""Output channel center frequency in Hz. Can be 'True'/'auto' to
                use the receiver channel target frequency (correcting for
                actual tuner offset), 'False' to use the receiver channel
                frequency unchanged, or a float value. (default: False)""",
    )
    chgroup.add_argument(
        "+k",
        "--scale",
        dest="ch_scalings",
        action=Extend,
        type=evalfloat,
        help="""Scale output channel by this factor. (default: 1)""",
    )
    chgroup.add_argument(
        "+n",
        "--subchannels",
        dest="ch_nsubchannels",
        action=Extend,
        type=evalint,
        help="""Number of subchannels for channelizing the output. A polyphase
                filter bank will be applied after the otherwise specified
                resampling and frequency shifting to further decimate the
                output and divide it into this many equally-spaced channels.
                (default: 1)""",
    )
    chgroup.add_argument(
        "--lpf_cutoff",
        dest="ch_lpf_cutoffs",
        action=Extend,
        type=evalfloat,
        help="""Normalized low-pass filter cutoff frequency (start of
                transition band), where a value of 1 indicates half the
                *output* sampling rate. Value in Hz is therefore
                (cutoff * out_sample_rate / 2.0). (default: 0.9)""",
    )
    chgroup.add_argument(
        "--lpf_transition_width",
        dest="ch_lpf_transition_widths",
        action=Extend,
        type=evalfloat,
        help="""Normalized width (in frequency) of low-pass filter transition
                region from pass band to stop band, where a value of 1
                indicates half the *output* sampling rate. Value in Hz is
                therefore (transition_width * out_sample_rate / 2.0).
                (default: 0.2)""",
    )
    chgroup.add_argument(
        "--lpf_attenuation",
        dest="ch_lpf_attenuations",
        action=Extend,
        type=evalfloat,
        help="""Minimum attenuation of the low-pass filter stop band in dB.
                (default: 80)""",
    )
    chgroup.add_argument(
        "--lpf_pass_ripple",
        dest="ch_lpf_pass_ripples",
        action=Extend,
        type=noneorfloat,
        help="""Maximum ripple of the low-pass filter pass band in dB. If
                'None', use the same value as `lpf_attenuation`.
                (default: None)""",
    )
    chgroup.add_argument(
        "+t",
        "--type",
        dest="ch_out_types",
        action=Extend,
        type=noneorstr,
        help="""Output channel data type to convert to ('scXX' for complex
                integer and 'fcXX' for complex float with XX bits). Use 'None'
                to skip conversion and use the USRP or filter output type.
                Conversion from float to integer will map a magnitude of 1.0
                (after any scaling) to the maximum integer value.
                (default: None)""",
    )
    return parser


def _add_drf_group(parser):

    drfgroup = parser.add_argument_group(title="digital_rf")
    drfgroup.add_argument(
        "-n",
        "--file_cadence_ms",
        dest="file_cadence_ms",
        type=evalint,
        help="""Number of milliseconds of data per file.
                (default: 1000)""",
    )
    drfgroup.add_argument(
        "-N",
        "--subdir_cadence_s",
        dest="subdir_cadence_s",
        type=evalint,
        help="""Number of seconds of data per subdirectory.
                (default: 3600)""",
    )
    drfgroup.add_argument(
        "--metadata",
        action=Extend,
        metavar="{KEY}={VALUE}",
        help="""Key, value metadata pairs to include with data.
                (default: "")""",
    )
    drfgroup.add_argument(
        "--uuid",
        dest="uuid",
        help="""Unique ID string for this data collection.
                (default: random)""",
    )
    return parser


def _add_time_group(parser):
    timegroup = parser.add_argument_group(title="time")
    timegroup.add_argument(
        "-s",
        "--starttime",
        dest="starttime",
        help="""Start time of the experiment as datetime (if in ISO8601 format:
                2016-01-01T15:24:00Z) or Unix time (if float/int).
                (default: start ASAP)""",
    )
    timegroup.add_argument(
        "-e",
        "--endtime",
        dest="endtime",
        help="""End time of the experiment as datetime (if in ISO8601 format:
                2016-01-01T16:24:00Z) or Unix time (if float/int).
                (default: wait for Ctrl-C)""",
    )
    timegroup.add_argument(
        "-l",
        "--duration",
        dest="duration",
        type=evalint,
        help="""Duration of experiment in seconds. When endtime is not given,
                end this long after start time. (default: wait for Ctrl-C)""",
    )
    timegroup.add_argument(
        "-p",
        "--cycle-length",
        dest="period",
        type=evalint,
        help="""Repeat time of experiment cycle. Align to start of next cycle
                if start time has passed. (default: 10)""",
    )
    return parser


def _build_thor_parser(Parser, *args):
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "THORPLUTO (The Haystack Observatory Recorder) for the Pluto"
    copyright = "Copyright (c) 2020 Massachusetts Institute of Technology"
    shortdesc = "Record data from ADALM Pluto SDRs in DigitalRF format."
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

    usage = (
        "%(prog)s [-m MBOARD] [-c CH]  [-f FREQ]"
        " [-F OFFSET] \\\n"
        "{0:8}[-g GAIN] [-b BANDWIDTH] [-r RATE] [options] DIR\n".format("")
    )

    epi_pars = [
        """\
        Arguments in the "mainboard", "receiver channel", and "output channel"
        groups accept multiple values, allowing multiple mainboards and
        channels to be specified. Multiple arguments can be provided by
        repeating the argument flag, by passing a comma-separated list of
        values, or both. Within each argument group, parameters will be grouped
        in the order in which they are given to form the complete set of
        parameters for each mainboard/channel. For any argument with fewer
        values given than the number of mainboards/channels, its values will be
        extended by repeatedly cycling through the values given up to the
        needed number.

        """,
        """\
        Arguments in other groups apply to all mainboards/channels (including
        the receiver sample rate).
        """,
        """\
        Example usage:
        """,
    ]
    epi_pars = [fill(dedent(s), width) for s in epi_pars]

    egtw = TextWrapper(
        width=(width - 2),
        break_long_words=False,
        break_on_hyphens=False,
        subsequent_indent=" " * (len(scriptname) + 1),
    )

    egs = [
        """\
        {0} -m usb:20.27.5 -c hpol -f 2.4e9 -r 100e6/24
        /data/test
        """,
        """\
        {0} -m 192.168.10.2 -c ch0 -f 440e6 -g 64
        -b 800e3 -r 1e6 /data/test
        """,
        """\
        {0} -c ch0 -f 437.5e6 -r 5e6 /data/test
        """,
    ]

    egs = [" \\\n".join(egtw.wrap(dedent(s.format(scriptname)))) for s in egs]
    epi = "\n" + "\n\n".join(epi_pars + egs) + "\n"

    # parse options
    parser = Parser(
        description=desc,
        usage=usage,
        epilog=epi,
        prefix_chars="-+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="THOR 3.1, using digital_rf {0}".format(drf.__version__),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        help="""Reduce text output to the screen. (default: False)""",
    )

    parser = _add_dir_group(parser)
    parser = _add_mainboard_group(parser)
    parser = _add_receiver_group(parser)
    parser = _add_rchannel_group(parser)
    parser = _add_ochannel_group(parser)
    parser = _add_drf_group(parser)
    parser = _add_time_group(parser)

    parser.set_defaults(func=_run_thor)

    return parser


def _run_thor(args):
    import signal

    if args.datadir is None:
        args.datadir = args.outdir
    del args.outdir

    # handle deprecated decimation argument, converting it to sample rate
    if args.decimations is not None:
        if args.samplerate is None:
            args.samplerate = 1e6
        args.ch_samplerates = [args.samplerate / d for d in args.decimations]
    del args.decimations

    # separate args.chs (num, name) tuples into args.channels and
    # args.channel_names
    if args.chs is not None:
        args.channels, args.channel_names = map(list, zip(*args.chs))
    del args.chs

    # convert metadata strings to a dictionary
    if args.metadata is not None:
        metadata_dict = {}
        for a in args.metadata:
            try:
                k, v = a.split("=")
            except ValueError:
                k = None
                v = a
            try:
                v = literal_eval(v)
            except ValueError:
                pass
            if k is None:
                metadata_dict.setdefault("metadata", []).append(v)
            else:
                metadata_dict[k] = v
        args.metadata = metadata_dict

    # ignore test_settings option if no starttime is set (starting right now)
    if args.starttime is None:
        args.test_settings = False

    options = {k: v for k, v in args._get_kwargs() if v is not None}
    runopts = {
        k: options.pop(k)
        for k in list(options.keys())
        if k in ("starttime", "endtime", "duration", "period")
    }
    del options["func"]

    # handle SIGTERM (getting killed) gracefully by calling sys.exit
    def sigterm_handler(signal, frame):
        print("Killed")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    thor = Thorpluto(**options)
    thor.run(**runopts)


if __name__ == "__main__":
    parser = _build_thor_parser(argparse.ArgumentParser)
    args = parser.parse_args()
    args.func(args)
