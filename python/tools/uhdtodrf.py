#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2020 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Directly record to Digital RF using UHD python API."""

import argparse
import ast
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import dedent, fill, TextWrapper

import digital_rf as drf
import numpy as np
import pytz
import scipy.signal as sig
from six.moves import queue
import uhd

# UHD globals not included in python uhd wrapper.
ALL_MBOARDS = 18446744073709551615
ALL_LOS = "all"
ALL_GAINS = ""
ALL_CHANS = 18446744073709551615


def equiripple_lpf(cutoff=0.8, transition_width=0.2, attenuation=80, pass_ripple=None):
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
    ampl = [1, 0]
    error_weight = [10 ** ((pass_ripple - attenuation) / 20.0), 1]

    # get estimate for the filter order (Oppenheim + Schafer 2nd ed, 7.104)
    M = ((attenuation + pass_ripple) / 2.0 - 13) / 2.324 / (np.pi * transition_width)
    # round up to nearest even-order (Type I) filter
    M = int(np.ceil(M / 2.0)) * 2 + 1

    for _attempts in range(20):
        # get taps for order M
        try:
            taps = sig.remez(M, bands, ampl, error_weight, Hz=2.0)

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
        act_atten = -20 * np.log10(np.max(np.abs(ampl[1] - np.abs(stopband))))

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


class Recorder(object):
    """Record data from a USRP to digital rf through the uhd python library."""

    def __init__(self, datadir, **kwargs):
        options = dict(
            verbose=True,
            # mainboard group (num: len of mboards)
            mboards=[],
            subdevs=["A:A"],
            clock_rates=[None],
            clock_sources=[""],
            time_sources=[""],
            # receiver group (apply to all)
            samplerate=1e6,
            dev_args=["recv_buff_size=100000000", "num_recv_frames=512"],
            stream_args=[],
            tune_args=[],
            time_sync=True,
            wait_for_lock=True,
            stop_on_dropped=False,
            realtime=False,
            test_settings=True,
            # receiver ch. group (num: matching channels from mboards/subdevs)
            centerfreqs=[100e6],
            lo_offsets=[0],
            lo_sources=[""],
            lo_exports=[None],
            dc_offsets=[False],
            iq_balances=[None],
            gains=[0],
            bandwidths=[0],
            antennas=[""],
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
        # Set up buffer and create a secondary buffer for future use
        # HACK may need to come up with way to set this better
        BUFFLEN = int(op.samplerate)

        self.bufflen = BUFFLEN
        nbuff = 2
        self.bufflist = [
            np.empty((op.nrchs, BUFFLEN), dtype=op.cpu_dtype) for i in range(nbuff)
        ]
        self.pntlist = [0 for i in range(nbuff)]
        self.nbuff = nbuff
        self.act_buff = 0

        # test usrp device settings, release device when done
        if op.test_settings:
            if op.verbose:
                print("Initialization: testing device settings.")
            self._usrp_setup()

            # finalize options (for settings that depend on USRP setup)
            self._finalize_options()

    @staticmethod
    def _parse_options(**kwargs):
        """Put all keyword options in a namespace and normalize them."""
        op = argparse.Namespace(**kwargs)

        # check that subdevice specifications are unique per-mainboard
        for sd in op.subdevs:
            sds = sd.split()
            if len(set(sds)) != len(sds):
                errstr = (
                    'Invalid subdevice specification: "{0}". '
                    "Each subdevice specification for a given mainboard must "
                    "be unique."
                )
                raise ValueError(errstr.format(sd))

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
            op.cpu_dtype = np.dtype([(str("r"), np.int16), (str("i"), np.int16)])
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
            op.cpu_dtype = np.dtype("complex64")
            # get full specification for output types
            supported_out_types = {
                "sc8": dict(
                    convert="float_to_char",
                    convert_kwargs=dict(vlen=2, scale=float(2 ** 7 - 1)),
                    dtype=np.dtype([(str("r"), np.int8), (str("i"), np.int8)]),
                    name="sc8",
                ),
                "sc16": dict(
                    convert="float_to_short",
                    convert_kwargs=dict(vlen=2, scale=float(2 ** 15 - 1)),
                    dtype=np.dtype([(str("r"), np.int16), (str("i"), np.int16)]),
                    name="sc16",
                ),
                "sc32": dict(
                    convert="float_to_int",
                    convert_kwargs=dict(vlen=2, scale=float(2 ** 31 - 1)),
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
        for mb_arg in ("subdevs", "clock_rates", "clock_sources", "time_sources"):
            val = getattr(op, mb_arg)
            mbval = list(islice(cycle(val), 0, op.nmboards))
            setattr(op, mb_arg, mbval)

        # get number of receiver channels by total number of subdevices over
        # all mainboards
        op.mboards_bychan = []
        op.subdevs_bychan = []
        op.mboardnum_bychan = []
        mboards = op.mboards if op.mboards else ["default"]
        for mbnum, (mb, sd) in enumerate(zip(mboards, op.subdevs)):
            sds = sd.split()
            mbs = list(repeat(mb, len(sds)))
            mbnums = list(repeat(mbnum, len(sds)))
            op.mboards_bychan.extend(mbs)
            op.subdevs_bychan.extend(sds)
            op.mboardnum_bychan.extend(mbnums)

        # repeat receiver channel arguments as necessary
        op.nrchs = len(op.subdevs_bychan)
        for rch_arg in (
            "antennas",
            "bandwidths",
            "centerfreqs",
            "dc_offsets",
            "iq_balances",
            "lo_offsets",
            "lo_sources",
            "lo_exports",
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
            if re.match(r"[^0-9]+=.+", mb):
                idtype, mb = mb.split("=")
            elif re.match(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", mb):
                idtype = "addr"
            elif (
                re.match(r"usrp[123]", mb)
                or re.match(r"b2[01]0", mb)
                or re.match(r"x3[01]0", mb)
            ):
                idtype = "type"
            elif re.match(r"[0-9A-Fa-f]{1,}", mb):
                idtype = "serial"
            else:
                idtype = "name"
            if len(op.mboards) == 1:
                # do not use identifier numbering if only using one mainboard
                s = "{type}={mb}".format(type=idtype, mb=mb.strip())
            else:
                s = "{type}{n}={mb}".format(type=idtype, n=n, mb=mb.strip())
            op.mboard_strs.append(s)

        if op.verbose:
            opstr = (
                dedent(
                    """\
                Main boards: {mboard_strs}
                Subdevices: {subdevs}
                Clock rates: {clock_rates}
                Clock sources: {clock_sources}
                Time sources: {time_sources}
                Sample rate: {samplerate}
                Device arguments: {dev_args}
                Stream arguments: {stream_args}
                Tune arguments: {tune_args}
                Antenna: {antennas}
                Bandwidth: {bandwidths}
                Frequency: {centerfreqs}
                LO frequency offset: {lo_offsets}
                LO source: {lo_sources}
                LO export: {lo_exports}
                Gain: {gains}
                DC offset: {dc_offsets}
                IQ balance: {iq_balances}
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

    def _finalize_options(self):
        """Apply changes to op object to deal with the sub banding."""
        op = self.op

        op.ch_samplerates_frac = []
        op.resampling_ratios = []
        op.resampling_filter_taps = []
        op.resampling_filter_delays = []
        op.channelizer_filter_taps = []
        op.channelizer_filter_taps_list = []
        op.channelizer_filter_delays = []
        op.total_filter_delay = []
        op.rotator = []
        op.max_filter = 0
        for ko, (osr, nsc) in enumerate(zip(op.ch_samplerates, op.ch_nsubchannels)):
            # get output sample rate fraction
            # (op.samplerate_frac final value is set in _usrp_setup
            #  so can't get output sample rate until after that is done)
            if osr is None:
                ch_samplerate_frac = op.samplerate_frac
            else:
                ch_samplerate_frac = Fraction(osr).limit_denominator()
            op.ch_samplerates_frac.append(ch_samplerate_frac)

            # get resampling ratio
            ratio = ch_samplerate_frac / op.samplerate_frac
            op.resampling_ratios.append(ratio)
            op.rotator.append(False)
            # get resampling low-pass filter taps
            if ratio == 1:
                op.resampling_filter_taps.append(np.zeros(0))
                op.resampling_filter_delays.append(0)
            else:
                # filter taps need to be designed for the highest rate
                # (i.e. after interpolation but before decimation)
                taps = equiripple_lpf(
                    cutoff=float(op.ch_lpf_cutoffs[ko]) / ratio.denominator,
                    transition_width=(
                        float(op.ch_lpf_transition_widths[ko]) / ratio.denominator
                    ),
                    attenuation=op.ch_lpf_attenuations[ko],
                    pass_ripple=op.ch_lpf_pass_ripples[ko],
                )
                # for unit gain in passband, need to multiply taps by
                # interpolation rate
                taps = ratio.numerator * taps
                op.resampling_filter_taps.append(taps)
                # calculate filter delay in same way as pfb_arb_resampler
                # (overall taps are applied at interpolated rate, but delay is
                #  still in terms of input rate, i.e. the taps per filter
                #  after being split into the polyphase filter bank)
                taps_per_filter = int(np.ceil(float(len(taps)) / ratio.numerator))
                op.resampling_filter_delays.append(Fraction(taps_per_filter - 1, 2))

            # get channelizer low-pass filter taps

            if nsc > 1:
                taps = equiripple_lpf(
                    cutoff=(op.ch_lpf_cutoffs[ko] / nsc),
                    transition_width=(op.ch_lpf_transition_widths[ko] / nsc),
                    attenuation=op.ch_lpf_attenuations[ko],
                    pass_ripple=op.ch_lpf_pass_ripples[ko],
                )
                op.channelizer_filter_taps.append(taps)
                op.channelizer_filter_taps_list.append([taps])
                op.channelizer_filter_delays.append(Fraction(len(taps) - 1, 2))
            else:
                op.channelizer_filter_taps.append(np.zeros(0))
                op.channelizer_filter_taps_list.append([np.zeros(0)])
                op.channelizer_filter_delays.append(0)
            m_rs = op.resampling_filter_taps[-1].shape[0]
            m_cf = op.channelizer_filter_taps[-1].shape[0]
            # Delay for filter without downsampling
            hlen_rs = max(0, (m_rs - 1) // 2)
            hlen_cf = max(0, (m_cf - 1) // 2)
            # Delay after downsampling
            rs_del = hlen_rs // ratio.denominator + bool(hlen_rs % ratio.denominator)
            ch_del = (rs_del + hlen_cf) // nsc + bool((rs_del + hlen_cf) % nsc)
            op.total_filter_delay.append(ch_del)
            overlap0 = ch_del * ratio.denominator * nsc * 2 + 1
            overlapout = overlap0 // ratio.numerator + bool(overlap0 % ratio.numerator)
            op.max_filter = max(overlapout, op.max_filter)
            op.rotator.append(False)

    def _usrp_setup(self):
        """Create, set up, and return USRP source and streamer objects.

        Using the op object set up uhd.usrp.MultiUSRP object and create a rx_streamer.
        to get all of the data.

        Returns
        -------
        usrp : MultiUSRP
            Object for the radios.

        rx_streamer : RX_streamer
            Streamer object for getting data.

        """
        op = self.op
        # create usrp source block
        op.otw_format = "sc16"
        usrp = uhd.usrp.MultiUSRP(",".join(chain(op.mboard_strs, op.dev_args)))

        # set mainboard options
        for mb_num in range(op.nmboards):
            usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(op.subdevs[mb_num]), mb_num)

            # set master clock rate
            clock_rate = op.clock_rates[mb_num]
            if clock_rate is not None:
                usrp.set_master_clock_rate(clock_rate, mb_num)

            # set clock source
            clock_source = op.clock_sources[mb_num]
            if not clock_source and op.wait_for_lock:
                clock_source = "external"
            if clock_source:
                try:
                    usrp.set_clock_source(clock_source, mb_num)
                except RuntimeError:
                    errstr = (
                        "Setting mainboard {0} clock_source to '{1}' failed."
                        " Must be one of {2}. If setting is valid, check that"
                        " the source (REF) is operational."
                    ).format(mb_num, clock_source, usrp.get_clock_sources(mb_num))
                    raise ValueError(errstr)

            # set time source
            time_source = op.time_sources[mb_num]
            if not time_source and op.time_sync:
                time_source = "external"
            if time_source:
                try:
                    usrp.set_time_source(time_source, mb_num)
                except RuntimeError:
                    errstr = (
                        "Setting mainboard {0} time_source to '{1}' failed."
                        " Must be one of {2}. If setting is valid, check that"
                        " the source (PPS) is operational."
                    ).format(mb_num, time_source, usrp.get_time_sources(mb_num))
                    raise ValueError(errstr)

        # check for ref lock
        mbnums_with_ref = [
            mb_num
            for mb_num in range(op.nmboards)
            if "ref_locked" in usrp.get_mboard_sensor_names(mb_num)
        ]
        if op.wait_for_lock and mbnums_with_ref:
            if op.verbose:
                sys.stdout.write("Waiting for reference lock...")
                sys.stdout.flush()
            timeout = 0
            if op.wait_for_lock is True:
                timeout_thresh = 30
            else:
                timeout_thresh = op.wait_for_lock
            while not all(
                usrp.get_mboard_sensor("ref_locked", mb_num).to_bool()
                for mb_num in mbnums_with_ref
            ):
                if op.verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                time.sleep(1)
                timeout += 1
                if timeout > timeout_thresh:
                    if op.verbose:
                        sys.stdout.write("failed\n")
                        sys.stdout.flush()
                    unlocked_mbs = [
                        mb_num
                        for mb_num in mbnums_with_ref
                        if usrp.get_mboard_sensor("ref_locked", mb_num).to_bool()
                    ]
                    errstr = (
                        "Failed to lock to 10 MHz reference on mainboards {0}."
                        " To skip waiting for lock, set `wait_for_lock` to"
                        " False (pass --nolock on the command line)."
                    ).format(unlocked_mbs)
                    raise RuntimeError(errstr)
            if op.verbose:
                sys.stdout.write("locked\n")
                sys.stdout.flush()

        # set global options
        # sample rate (set after clock rate so it can be calculated correctly)
        usrp.set_rx_rate(float(op.samplerate))
        time.sleep(0.25)
        # read back actual mainboard options
        # (clock rate can be affected by setting sample rate)
        for mb_num in range(op.nmboards):
            op.clock_rates[mb_num] = usrp.get_master_clock_rate(mb_num)
            op.clock_sources[mb_num] = usrp.get_clock_source(mb_num)
            op.time_sources[mb_num] = usrp.get_time_source(mb_num)

        # read back actual sample rate value
        samplerate = usrp.get_rx_rate()
        # calculate longdouble precision/rational sample rate
        # (integer division of clock rate)
        cr = op.clock_rates[0]
        srdec = int(round(cr / samplerate))
        samplerate_ld = np.longdouble(cr) / srdec
        op.samplerate = samplerate_ld
        op.samplerate_frac = Fraction(cr).limit_denominator() / srdec

        # set per-channel options
        # set command time so settings are synced
        COMMAND_DELAY = 0.2
        cmd_time = usrp.get_time_now() + uhd.types.TimeSpec(COMMAND_DELAY)
        usrp.set_command_time(cmd_time, ALL_MBOARDS)  # defaults to all mboards
        for ch_num in range(op.nrchs):
            # local oscillator sharing settings
            lo_source = op.lo_sources[ch_num]
            if lo_source:
                try:
                    usrp.set_rx_lo_source(lo_source, ALL_LOS, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown LO source option: '{0}'. Must be one of {1},"
                        " or it may not be possible to set the LO source on"
                        " this daughterboard."
                    ).format(lo_source, usrp.get_rx_lo_sources(ALL_LOS, ch_num))
                    raise ValueError(errstr)
            lo_export = op.lo_exports[ch_num]
            if lo_export is not None:
                if not lo_source:
                    errstr = (
                        "Channel {0}: must set an LO source in order to set"
                        " LO export."
                    ).format(ch_num)
                    raise ValueError(errstr)
                usrp.set_rx_lo_export_enabled(True, ALL_LOS, ch_num)
            # center frequency and tuning offset
            # HACK TuneRequest constructor does not take tune args as input.
            #   Need to set args afterward.
            tune_req = uhd.types.TuneRequest(
                op.centerfreqs[ch_num], op.lo_offsets[ch_num]
            )
            tune_req.args = uhd.types.DeviceAddr(",".join(op.tune_args))
            tune_res = usrp.set_rx_freq(tune_req, ch_num)
            time.sleep(0.5)

            # store actual values from tune result
            op.centerfreqs[ch_num] = tune_res.actual_rf_freq - tune_res.actual_dsp_freq
            op.lo_offsets[ch_num] = tune_res.actual_dsp_freq
            # dc offset
            dc_offset = op.dc_offsets[ch_num]
            if dc_offset is True:
                usrp.set_rx_dc_offset(True, ch_num)
            elif dc_offset is False:
                usrp.set_rx_dc_offset(False, ch_num)
            elif dc_offset is not None:
                usrp.set_rx_dc_offset(dc_offset, ch_num)
            # iq balance
            iq_balance = op.iq_balances[ch_num]
            if iq_balance is True:
                usrp.set_rx_iq_balance(True, ch_num)
            elif iq_balance is False:
                usrp.set_rx_iq_balance(False, ch_num)
            elif iq_balance is not None:
                usrp.set_rx_iq_balance(iq_balance, ch_num)
            # gain
            usrp.set_rx_gain(op.gains[ch_num], ch_num)
            # bandwidth
            bw = op.bandwidths[ch_num]
            if bw:
                usrp.set_rx_bandwidth(bw, ch_num)
            # antenna
            ant = op.antennas[ch_num]
            if ant:
                try:
                    usrp.set_rx_antenna(ant, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown RX antenna option: '{0}'. Must be one of {1}."
                    ).format(ant, usrp.get_antennas(ch_num))
                    raise ValueError(errstr)

        # commands are done, clear time
        usrp.clear_command_time(ALL_MBOARDS)
        time.sleep(COMMAND_DELAY)
        st_args = uhd.usrp.StreamArgs(op.cpu_format, op.otw_format)
        st_args.channels = np.unique(op.channels).tolist()
        rx_streamer = usrp.get_rx_stream(st_args)
        # read back actual channel settings
        op.usrpinfo = []
        for ch_num in range(op.nrchs):
            if op.lo_sources[ch_num]:
                op.lo_sources[ch_num] = usrp.get_rx_lo_sources(ALL_LOS, ch_num)
            if op.lo_exports[ch_num] is not None:
                op.lo_exports[ch_num] = usrp.get_rx_lo_export_enabled(ALL_LOS, ch_num)
            op.gains[ch_num] = usrp.get_rx_gain(ch_num)
            op.bandwidths[ch_num] = usrp.get_rx_bandwidth(ch_num)
            op.antennas[ch_num] = usrp.get_rx_antenna(ch_num)
            op.usrpinfo.append(dict(usrp.get_usrp_rx_info(ch_num)))

        if op.verbose:
            print("Using the following devices:")
            chinfostrs = [
                "Motherboard: {mb_id} ({mb_addr}) | Daughterboard: {db_name}",
                "Subdev: {sub} | Antenna: {ant} | Gain: {gain} | Rate: {sr}",
                "Frequency: {freq:.3f} ({lo_off:+.3f}) | Bandwidth: {bw}",
            ]
            if any(op.lo_sources) or any(op.lo_exports):
                chinfostrs.append("LO source: {lo_source} | LO export: {lo_export}")
            chinfo = "\n".join(["  " + l for l in chinfostrs])
            for ch_num in range(op.nrchs):
                header = "---- receiver channel {0} ".format(ch_num)
                header += "-" * (78 - len(header))
                print(header)
                usrpinfo = op.usrpinfo[ch_num]
                info = {}
                info["mb_id"] = usrpinfo["mboard_id"]
                mba = op.mboards_bychan[ch_num]
                if mba == "default":
                    mba = usrpinfo["mboard_serial"]
                info["mb_addr"] = mba
                info["db_name"] = usrpinfo["rx_subdev_name"]
                info["sub"] = op.subdevs_bychan[ch_num]
                info["ant"] = op.antennas[ch_num]
                info["bw"] = op.bandwidths[ch_num]
                info["freq"] = op.centerfreqs[ch_num]
                info["gain"] = op.gains[ch_num]
                info["lo_off"] = op.lo_offsets[ch_num]
                info["lo_source"] = op.lo_sources[ch_num]
                info["lo_export"] = op.lo_exports[ch_num]
                info["sr"] = op.samplerate
                print(chinfo.format(**info))
                print("-" * 78)

        return usrp, rx_streamer

    def run(self, starttime=None, endtime=None, duration=None, period=10):
        """Launch threads that run the buffering, processing and recording.

        This function sets launches the threads and final set up for the timeing.

        Parameters
        ----------
        starttime : str, optional
            Start time string. Defaults to None.

        endtime : str, optional
            End time string. Defaults to None.

        duration : float, optional
            Recording duration. Defaults to None.

        period : int, optional
            Cycle repeat period. Defaults to 10.

        """
        op = self.op

        # window in seconds that we allow for setup time so that we don't
        # issue a start command that's in the past when the recording starts
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

        # if op.realtime:
        #     r = gr.enable_realtime_scheduling()
        #
        #     if op.verbose:
        #         if r == gr.RT_OK:
        #             print("Realtime scheduling enabled")
        #         else:
        #             print("Note: failed to enable realtime scheduling")

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
        usrp, stream = self._usrp_setup()
        # finalize options (for settings that depend on USRP setup)
        self._finalize_options()

        # set device time
        tt = time.time()
        if op.time_sync:
            # wait until time 0.2 to 0.5 past full second, then latch
            # we have to trust NTP to be 0.2 s accurate
            while tt - math.floor(tt) < 0.2 or tt - math.floor(tt) > 0.3:
                time.sleep(0.01)
                tt = time.time()
            if op.verbose:
                print("Latching at " + str(tt))
            # waits for the next pps to happen
            # (at time math.ceil(tt))
            # then sets the time for the subsequent pps
            # (at time math.ceil(tt) + 1.0)
            usrp.set_time_unknown_pps(uhd.types.TimeSpec(math.ceil(tt) + 1.0))
        else:
            usrp.set_time_now(uhd.types.TimeSpec(tt), ALL_MBOARDS)

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
        # command launch time
        ct_td = lt - drf.util.epoch
        ct_secs = ct_td.total_seconds() // 1.0
        ct_frac = ct_td.microseconds / 1000000.0

        # Craft and send the Stream Command
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        # Need to set this to False if using Multiple receive channels. This is
        # buried pretty deep in uhd.
        stream_cmd.stream_now = op.nrchs == 1
        stream_cmd.time_spec = uhd.types.TimeSpec(ct_secs) + uhd.types.TimeSpec(ct_frac)

        stream.issue_stream_cmd(stream_cmd)

        # Set up drf writers
        drfObjs = []
        # ko is for output channel, kr is the radio channel
        for ko, kr in enumerate(op.channels):
            # make channelizer if necessary
            nsc = op.ch_nsubchannels[ko]
            mbnum = op.mboardnum_bychan[kr]

            cpath = os.path.join(op.datadir, op.channel_names[ko])
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
                    # create band-pass filter (complex taps)
                    f_shift = ch_centerfreq - op.centerfreqs[kr]
                    phase_inc = 2 * np.pi * f_shift / op.samplerate
                    rotator = np.exp(phase_inc * 1j * np.arange(len(rs_taps)))
                    rs_taps = (rs_taps * rotator).astype("complex64")
                    op.resampling_filter_taps[ko] = rs_taps

                else:
                    # save low-pass filter (float taps)
                    op.resampling_filter_taps[ko] = rs_taps

            else:
                conv_scaling = scaling

            # make frequency shift rotator if necessary
            if ch_centerfreq is not False:
                f_shift = ch_centerfreq - op.centerfreqs[kr]
                phase_inc = -2 * np.pi * f_shift / ch_samplerate_frac
                op.rotator[ko] = phase_inc
            else:
                ch_centerfreq = op.centerfreqs[kr]
            # make channelizer if necessary
            if nsc > 1:
                sc_taps = op.channelizer_filter_taps[ko]
                n_h = np.arange(len(sc_taps))
                f_frac = np.arange(nsc, dtype=float) / nsc

                tap_list = []
                for i_f in f_frac:
                    f_vec = np.exp(2j * np.pi * i_f * n_h)
                    tmp_taps = f_vec * sc_taps
                    tap_list.append(tmp_taps.astype("complex64"))

                op.channelizer_filter_taps_list[ko] = tap_list

                # # declare sample delay for the filter block so that tags are
                # # propagated to the correct sample
                # # (for channelized, delay is applied for each filter in the
                # #  polyphase bank, so this needs to be the output sample delay)
                # filt.declare_sample_delay(int(op.channelizer_filter_delays[ko] / nsc))

                # adjust start sample to account for filter delay so first
                # sample going to output is shifted to an earlier time
                # (adjustment is in terms of filter output samples, so need to
                #  take the input filter delay and account for the output rate)
                # start_sample_adjust = int(
                #     (start_sample_adjust - op.channelizer_filter_delays[ko]) / nsc
                # )

                # modify output settings accordingly
                ch_centerfreq = ch_centerfreq + np.fft.fftfreq(
                    nsc, 1 / float(ch_samplerate_frac)
                )
                ch_samplerate_frac = ch_samplerate_frac / nsc
            else:
                ch_centerfreq = np.array([ch_centerfreq])

            # make conversion block if necessary
            ot_dict = op.ch_out_specs[ko]
            converter = ot_dict["convert"]
            if converter is not None:
                kw = ot_dict["convert_kwargs"]
                # increase vector length of input due to channelizer
                # incorporate any scaling into type conversion block
                conv_scaling *= kw["scale"]

            op.ch_out_specs[ko]["conv_scaling"] = conv_scaling
            # get start sample
            ch_samplerate_ld = np.longdouble(
                ch_samplerate_frac.numerator
            ) / np.longdouble(ch_samplerate_frac.denominator)
            start_sample = int(np.uint64(ltts * ch_samplerate_ld)) + start_sample_adjust
            metadata = dict(
                # receiver metadata for USRP
                center_frequencies=ch_centerfreq,
                receiver=dict(
                    description="UHD USRP source using pyuhd",
                    info=op.usrpinfo[kr],
                    antenna=op.antennas[kr],
                    bandwidth=op.bandwidths[kr],
                    center_freq=op.centerfreqs[kr],
                    clock_rate=op.clock_rates[mbnum],
                    clock_source=op.clock_sources[mbnum],
                    dc_offset=op.dc_offsets[kr],
                    gain=op.gains[kr],
                    id=op.mboards_bychan[kr],
                    iq_balance=op.iq_balances[kr],
                    lo_export=op.lo_exports[kr],
                    lo_offset=op.lo_offsets[kr],
                    lo_source=op.lo_sources[kr],
                    otw_format=op.otw_format,
                    samp_rate=usrp.get_rx_rate(),
                    stream_args=",".join(op.stream_args),
                    subdev=op.subdevs_bychan[kr],
                    time_source=op.time_sources[mbnum],
                ),
                processing=dict(  # Filtering info
                    channelizer_filter_taps=op.channelizer_filter_taps[ko],
                    decimation=op.resampling_ratios[ko].denominator,
                    interpolation=op.resampling_ratios[ko].numerator,
                    resampling_filter_taps=op.resampling_filter_taps[ko],
                    scaling=op.ch_scalings[ko],
                ),
                **op.metadata
            )
            # Metadata writer and write at first record sample
            mdata_path = os.path.join(cpath, "metadata")
            if not os.path.isdir(mdata_path):
                os.makedirs(mdata_path)
            mdatawrite = drf.DigitalMetadataWriter(
                mdata_path,
                3600,
                60,
                ch_samplerate_frac.numerator,
                ch_samplerate_frac.denominator,
                "uhdtodrf",
            )
            mdatawrite.write(start_sample, metadata)
            drf_out = drf.DigitalRFWriter(
                cpath,
                op.ch_out_specs[ko]["dtype"],
                op.subdir_cadence_s,
                op.file_cadence_ms,
                start_sample,
                ch_samplerate_frac.numerator,
                ch_samplerate_frac.denominator,
                op.uuid,
                compression_level=0,
                checksum=False,
                is_complex=True,
                num_subchannels=nsc,
                is_continuous=True,
                marching_periods=True,
            )
            drfObjs.append(drf_out)

        # Lists for processing and write threads.
        proc_threads = []
        write_threads = []
        radfifo = queue.Queue()  # receive queue
        write_fifo = queue.Queue()
        proc_count = 0
        end_rec = threading.Event()
        if et is None and duration is not None:
            et = lt + timedelta(seconds=duration)
        try:
            # Start the buffering thread
            read_th = threading.Thread(
                target=self.buff_func, args=(stream, radfifo, end_rec, start_sample)
            )
            read_th.start()
            read_th.setName("Read Thread")
            while not end_rec.is_set():

                if et is not None:
                    stop_bool = pytz.utc.localize(datetime.utcnow()) >= et - timedelta(
                        seconds=1
                    )
                    if stop_bool:
                        end_rec.set()

                if not radfifo.empty():
                    d1 = radfifo.get()
                    tmp = self.bufflist[d1[0]][:, : self.pntlist[d1[0]]]
                    cur_pt = threading.Thread(
                        target=self.procsamples, args=(write_fifo, tmp, d1[1])
                    )
                    cur_pt.start()
                    cur_pt.setName("Proc{0}".format(proc_count))
                    proc_threads.append(cur_pt)
                    proc_count += 1
                if proc_threads:
                    if not proc_threads[0].isAlive():
                        write_data = write_fifo.get()
                        cur_wt = threading.Thread(
                            target=write_samples, args=(drfObjs, write_data)
                        )
                        cur_wt.start()
                        cur_wt.setName("Save to drf")
                        write_threads.append(cur_wt)
                        proc_threads.pop(0)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                if write_threads:
                    if not write_threads[0].isAlive():
                        write_threads.pop(0)

            time.sleep(1)

        except RuntimeError as ex:
            print("Runtime error in receive: %s", ex)
        # Handle the error codes

        except KeyboardInterrupt:
            end_rec.set()
            time.sleep(1)

        finally:

            while write_threads or proc_threads or (not radfifo.empty()):
                if not radfifo.empty():
                    d1 = radfifo.get()
                    tmp = self.bufflist[d1[0]][:, : self.pntlist[d1[0]]]
                    cur_pt = threading.Thread(
                        target=self.procsamples, args=(write_fifo, tmp, d1[1])
                    )
                    cur_pt.start()
                    cur_pt.setName("Proc{0}".format(proc_count))
                    proc_threads.append(cur_pt)
                    proc_count += 1
                if proc_threads:
                    if not proc_threads[0].isAlive():
                        write_data = write_fifo.get()

                        cur_wt = threading.Thread(
                            target=write_samples, args=(drfObjs, write_data)
                        )
                        cur_wt.start()
                        cur_wt.setName("Save to drf")
                        write_threads.append(cur_wt)
                        proc_threads.pop(0)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                if write_threads:
                    if not write_threads[0].isAlive():
                        write_threads.pop(0)

                time.sleep(0.1)
            for iobj in drfObjs:
                iobj.close()

            print("done")
            sys.stdout.flush()

    def buff_func(self, stream, radfifo, end_rec, start_sample):
        """Call the receive command for the streamer and places the data in a buffer.

        This function repeatly calls the recv command from the streamer and places
        the data in a buffer inside the bufflist object. Once a buffer is full
        the function cycles through to the next buffer in the list in a round
        robin style assuming that the processing and recording threads have
        copied the old data.

        Parameters
        ----------
        stream : RX_streamer
            Streamer that has already started taking data.

        radfifo : Queue
            Will be filled for the next steps.

        end_rec : Threading.Event
            If set will stop taking data.

        start_sample :long
            Start sample in number of samples in Posix Epoch.

        """
        try:
            op = self.op

            # To estimate the number of dropped samples in an overflow situation,
            # we need the following:
            #   On the first overflow, set had_an_overflow and record the time
            #   On the next ERROR_CODE_NONE, calculate how long its been since
            #       the recorded time, and use the tick rate to estimate the
            #       number of dropped samples. Also, reset the tracking variables.
            had_an_overflow = False
            last_overflow = uhd.types.TimeSpec(0)

            samp_num = 0
            m_overlap = op.max_filter
            sps = int(op.samplerate)
            # Make a receive buffer
            num_channels = stream.get_num_channels()
            max_samps_per_packet = stream.get_max_num_samps()
            # receive  buffer from
            # HACK need a parameter for the dytpe of the cpu format
            recv_buffer = np.empty(
                (num_channels, max_samps_per_packet), dtype=op.cpu_dtype
            )

            radio_meta = uhd.types.RXMetadata()
            num_rx_dropped = 0
            while not end_rec.is_set():
                num_rx = stream.recv(recv_buffer, radio_meta, 1.0)

                rec_samp = radio_meta.time_spec.to_ticks(sps)  # +np.arange(num_rx)

                # Logic for how to deal with samples before start time.
                if rec_samp + num_rx < start_sample:
                    continue
                # Go through error checks
                if radio_meta.error_code == uhd.types.RXMetadataErrorCode.none:
                    # Reset the overflow flag
                    if had_an_overflow:
                        had_an_overflow = False

                        num_rx_dropped = uhd.types.TimeSpec(
                            radio_meta.time_spec.get_real_secs()
                            - last_overflow.get_real_secs()
                        ).to_ticks(sps)
                        # Break out of loop if dropped sample.
                        if op.stop_on_dropped:
                            end_rec.set()
                elif radio_meta.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    had_an_overflow = True
                    last_overflow = radio_meta.time_spec
                    continue
                    # If we had a sequence error, record it
                    if radio_meta.out_of_sequence:
                        end_rec.set()
                        break

                else:
                    print("Receiver error: %s", radio_meta.strerror())
                    break

                # Write the current set of samples to memory.
                # Put this after the error checks to put in nans for overflows.

                a = self.act_buff
                bpnt = self.pntlist[a]
                if self.pntlist[a] + num_rx + num_rx_dropped >= self.bufflen:
                    b = (a + 1) % self.nbuff
                    cpnt = self.pntlist[a]

                    if m_overlap > 0:
                        self.bufflist[b][:, :m_overlap] = self.bufflist[b][
                            :, cpnt - m_overlap : cpnt
                        ]
                    radfifo.put([a, samp_num])
                    samp_num += bpnt
                    self.pntlist[b] = m_overlap
                    self.act_buff = b
                    a = self.act_buff
                    bpnt = self.pntlist[a]

                if num_rx_dropped:

                    end_pnt = min(self.bufflen, bpnt + num_rx_dropped)
                    self.bufflist[a][:, bpnt:end_pnt] = np.nan
                    self.pntlist[a] += num_rx_dropped
                    num_rx_dropped = 0

                beg_samp = max(0, start_sample - rec_samp)
                a = self.act_buff
                bpnt = self.pntlist[a]
                self.bufflist[a][:, bpnt : bpnt + num_rx - beg_samp] = recv_buffer[
                    :, beg_samp:num_rx
                ]
                self.pntlist[a] += num_rx - beg_samp

        except Exception as error:
            print("Radio buffer caught this error: " + repr(error))
            end_rec.set()
        finally:
            # After we get the signal to stop, issue a stop command
            stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
            # Set the last buffer to be read
            a = self.act_buff

            radfifo.put([a, samp_num])

    def procsamples(self, out_que, data_samples, start_sample):
        """Perform resampling, channelization, scaling and setting data types.

        Parameters
        ----------
        out_que : Queue
            Output for the processed data.

        data_samples : array
            Input data samples in nrec x N array.

        start_sample : long
            Time of the first sample.

        """
        try:
            op = self.op

            nchans = len(op.channels)
            outlist = [0 for i in range(nchans)]

            # HACK Make this a class method so I don't need to make dictionary.
            for ko, kr in enumerate(op.channels):
                rec_data = data_samples[kr]
                n_data = rec_data.shape[0]
                m_over = op.max_filter
                h_len = (m_over - 1) / 2
                rs = op.resampling_ratios[ko]
                rs_taps = op.resampling_filter_taps[ko]

                rot = op.rotator[ko]
                nsc = op.ch_nsubchannels[ko]
                n_del = (rs.numerator * h_len) // (rs.denominator * nsc) + bool(
                    (rs.numerator * h_len) % (rs.denominator * nsc)
                )
                n_out = n_data - (m_over - 1)
                n_out = n_out * rs.numerator // rs.denominator + bool(
                    (n_out * rs.numerator) % rs.denominator
                )
                n_out = n_out // nsc + bool(n_out % nsc)

                sc_taps = op.channelizer_filter_taps_list[ko]
                ot_dict = op.ch_out_specs[ko]

                conv_scaling = ot_dict["conv_scaling"]
                convert = ot_dict["convert"]
                # Resampling
                if rs != 1:
                    rec_data = sig.resample_poly(
                        rec_data, rs.numerator, rs.denominator, window=rs_taps
                    )
                # frequency rotation with no resampling
                elif rot:
                    rec_data = rec_data * np.exp(rot * 1j * np.arange(len(rec_data)))
                    rec_data = rec_data * np.exp(rot * 1j * start_sample)
                # sub banding
                if nsc > 1:
                    n_ds = rec_data.shape[0]
                    n_ch = n_ds // nsc + bool(n_ds % nsc)
                    xout = np.empty((n_ch, nsc), dtype=rec_data.dtype)
                    for isc in range(nsc):
                        cur_tap = sc_taps[isc]
                        xout[:, isc] = sig.resample_poly(
                            rec_data, 1, nsc, window=cur_tap
                        )
                    rec_data = xout

                rec_data = rec_data[n_del : n_out + n_del]
                # scaling
                if conv_scaling != 1:
                    rec_data = conv_scaling * rec_data
                # HACK Need to make a set of functions to do all of the translations.
                # conversion of number type
                if not (convert is None):
                    tmp_data = np.empty(rec_data.shape, dtype=ot_dict["dtype"])
                    tmp_data["r"] = np.round(rec_data.real)
                    tmp_data["i"] = np.round(rec_data.imag)
                    rec_data = tmp_data
                outlist[ko] = rec_data
            out_que.put(outlist)
        except Exception as error:
            print("Processor caught this error: " + repr(error))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


def write_samples(drfObj, data_samples):
    """Write out data buffer, typically in a separate thread.

    Parameters
    ----------
    drfObj : list
        List of digital rf writers for each channel.

    data_samples : array, shape (nchan, nsample)
        Array that will be saved.

    """
    try:

        for i_num, iobj in enumerate(drfObj):
            iobj.rf_write(data_samples[i_num])
    except Exception as error:
        print("write function caught this error: " + repr(error))


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
        "-m",
        "--mainboard",
        dest="mboards",
        action=Extend,
        help="""Mainboard address. (default: first device found)""",
    )
    mbgroup.add_argument(
        "-d",
        "--subdevice",
        dest="subdevs",
        action=Extend,
        help="""USRP subdevice string. (default: "A:A")""",
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
        "-A",
        "--devargs",
        dest="dev_args",
        action=Extend,
        help="""Device arguments, e.g. "master_clock_rate=30e6".
                (default: 'recv_buff_size=100000000,num_recv_frames=512')""",
    )
    recgroup.add_argument(
        "-a",
        "--streamargs",
        dest="stream_args",
        action=Extend,
        help="""Stream arguments, e.g. "peak=0.125,fullscale=1.0".
                (default: '')""",
    )
    recgroup.add_argument(
        "-T",
        "--tuneargs",
        dest="tune_args",
        action=Extend,
        help="""Tune request arguments, e.g. "mode_n=integer,int_n_step=100e3".
                (default: '')""",
    )
    # kept for backward compatibility,
    # replaced by clock_source/time_source in 2.6
    recgroup.add_argument("--sync_source", dest="sync_source", help=argparse.SUPPRESS)
    recgroup.add_argument(
        "--nosync",
        dest="time_sync",
        action="store_false",
        help="""Skip syncing with reference time. (default: False)""",
    )
    recgroup.add_argument(
        "--nolock",
        dest="wait_for_lock",
        action="store_false",
        help="""Don't wait for reference clock to lock. (default: False)""",
    )
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
        help="""Center frequency in Hz. (default: 100e6)""",
    )
    chgroup.add_argument(
        "-F",
        "--lo_offset",
        dest="lo_offsets",
        action=Extend,
        type=evalfloat,
        help="""Frontend tuner offset from center frequency, in Hz.
                (default: 0)""",
    )
    chgroup.add_argument(
        "--lo_source",
        dest="lo_sources",
        action=Extend,
        type=noneorstr,
        help="""Local oscillator source. Typically 'None'/'' (do not set),
                'internal' (e.g. LO1 for CH1, LO2 for CH2),
                'companion' (e.g. LO2 for CH1, LO1 for CH2), or
                'external' (neighboring board via connector).
                (default: '')""",
    )
    chgroup.add_argument(
        "--lo_export",
        dest="lo_exports",
        action=Extend,
        type=noneorbool,
        help="""Whether to export the LO's source to the external connector.
                Can be 'None'/'' to skip the channel, otherwise it can be
                'True' or 'False' provided the LO source is set.
                (default: None)""",
    )
    chgroup.add_argument(
        "--dc_offset",
        dest="dc_offsets",
        action=Extend,
        type=noneorboolorcomplex,
        help="""DC offset correction to use. Can be 'None'/'' to keep device
                default, 'True'/'auto' to enable automatic correction, 'False'
                to disable automatic correction, or a complex value
                (e.g. "1+1j"). (default: False)""",
    )
    chgroup.add_argument(
        "--iq_balance",
        dest="iq_balances",
        action=Extend,
        type=noneorboolorcomplex,
        help="""IQ balance correction to use. Can be 'None'/'' to keep device
                default, 'True'/'auto' to enable automatic correction, 'False'
                to disable automatic correction, or a complex value
                (e.g. "1+1j"). (default: None)""",
    )
    chgroup.add_argument(
        "-g",
        "--gain",
        dest="gains",
        action=Extend,
        type=evalfloat,
        help="""Gain in dB. (default: 0)""",
    )
    chgroup.add_argument(
        "-b",
        "--bandwidth",
        dest="bandwidths",
        action=Extend,
        type=evalfloat,
        help="""Frontend bandwidth in Hz. (default: 0 == frontend default)""",
    )
    chgroup.add_argument(
        "-y",
        "--antenna",
        dest="antennas",
        action=Extend,
        type=noneorstr,
        help="""Name of antenna to select on the frontend.
                (default: frontend default))""",
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


def _build_parser(Parser, *args):
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "uhdtodrf"
    copyright = "Copyright (c) 2017 Massachusetts Institute of Technology"
    shortdesc = "Record data from synchronized USRPs in DigitalRF format."
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
        "%(prog)s [-m MBOARD] [-d SUBDEV] [-c CH] [-y ANT] [-f FREQ]"
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
        {0} -m 192.168.20.2 -d "A:A A:B" -c h,v -f 95e6 -r 100e6/24
        /data/test
        """,
        """\
        {0} -m 192.168.10.2 -d "A:0" -c ch1 -y "TX/RX" -f 20e6 -F 10e3 -g 20
        -b 0 -r 1e6 /data/test
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

    # parser.set_defaults(func=_run_thor)

    return parser


def _ops_setup(args):

    if args.datadir is None:
        args.datadir = args.outdir
    del args.outdir

    # handle deprecated decimation argument, converting it to sample rate
    if args.decimations is not None:
        if args.samplerate is None:
            args.samplerate = 1e6
        args.ch_samplerates = [args.samplerate / d for d in args.decimations]
    del args.decimations

    # handle deprecated sync_source argument, converting it to clock_sources
    # and time_sources
    if args.sync_source is not None:
        if args.clock_sources is None:
            args.clock_sources = [args.sync_source]
        if args.time_sources is None:
            args.time_sources = [args.sync_source]
    del args.sync_source

    # separate args.chs (num, name) tuples into args.channels and
    # args.channel_names
    if args.chs is not None:
        args.channels, args.channel_names = map(list, zip(*args.chs))
    del args.chs

    # remove redundant arguments in dev_args, stream_args, tune_args
    if args.dev_args is not None:
        try:
            dev_args_dict = dict([a.split("=") for a in args.dev_args])
        except ValueError:
            raise ValueError("Device arguments must be {KEY}={VALUE} pairs.")
        args.dev_args = ["{0}={1}".format(k, v) for k, v in dev_args_dict.items()]
    if args.stream_args is not None:
        try:
            stream_args_dict = dict([a.split("=") for a in args.stream_args])
        except ValueError:
            raise ValueError("Stream arguments must be {KEY}={VALUE} pairs.")
        args.stream_args = ["{0}={1}".format(k, v) for k, v in stream_args_dict.items()]
    if args.tune_args is not None:
        try:
            tune_args_dict = dict([a.split("=") for a in args.tune_args])
        except ValueError:
            raise ValueError("Tune request arguments must be {KEY}={VALUE} pairs.")
        args.tune_args = ["{0}={1}".format(k, v) for k, v in tune_args_dict.items()]

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
                v = ast.literal_eval(v)
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
    return options, runopts


if __name__ == "__main__":

    parser = _build_parser(argparse.ArgumentParser)
    args = parser.parse_args()
    options, runopts = _ops_setup(args)
    import signal

    # handle SIGTERM (getting killed) gracefully by calling sys.exit
    def sigterm_handler(signal, frame):
        print("Killed")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    rec1 = Recorder(**options)
    rec1.run(**runopts)
