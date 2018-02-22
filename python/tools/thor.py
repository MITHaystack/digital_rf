#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Record data from synchronized USRPs in Digital RF format."""
from __future__ import print_function

import math
import os
import re
import sys
import time
from argparse import (
    Action, ArgumentParser, Namespace, RawDescriptionHelpFormatter,
)
from ast import literal_eval
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import TextWrapper, dedent, fill

import numpy as np
import pytz
from gnuradio import blocks, gr, uhd
from gnuradio.filter import firdes, freq_xlating_fir_filter_ccf

import digital_rf as drf
import gr_digital_rf as gr_drf


class Thor(object):
    """Record data from synchronized USRPs in DigitalRF format."""

    def __init__(
        self, datadir, verbose=True,
        # mainboard group (num: len of mboards)
        mboards=[], subdevs=['A:A'],
        # receiver group (apply to all)
        samplerate=1e6,
        dev_args=['recv_buff_size=100000000', 'num_recv_frames=512'],
        stream_args=[], tune_args=[],
        sync=True, sync_source='external',
        stop_on_dropped=False, realtime=False, test_settings=True,
        # receiver channel group (num: matching channels from mboards/subdevs)
        centerfreqs=[100e6],
        lo_offsets=[0], lo_sources=[''], lo_exports=[None],
        dc_offsets=[False], iq_balances=[None],
        gains=[0], bandwidths=[0], antennas=[''],
        # output channel group (num: len of channel_names)
        channel_names=['ch0'], channels=[None], decimations=[1],
        scalings=[1.0], out_types=[None],
        # digital_rf group (apply to all)
        file_cadence_ms=1000, subdir_cadence_s=3600, metadata={}, uuid=None,
    ):
        options = locals()
        del options['self']
        op = self._parse_options(**options)
        self.op = op

        # test usrp device settings, release device when done
        if op.test_settings:
            if op.verbose:
                print('Initialization: testing device settings.')
            u = self._usrp_setup()
            del u

    @staticmethod
    def _parse_options(**kwargs):
        """Put all keyword options in a namespace and normalize them."""
        op = Namespace(**kwargs)

        # check that subdevice specifications are unique per-mainboard
        for sd in op.subdevs:
            sds = sd.split()
            if len(set(sds)) != len(sds):
                errstr = (
                    'Invalid subdevice specification: "{0}". '
                    'Each subdevice specification for a given mainboard must '
                    'be unique.'
                )
                raise ValueError(errstr.format(sd))

        # get USRP cpu_format based on output type and decimation requirements
        if (all(ot is None or ot == 'sc16' for ot in op.out_types)
                and all(d == 1 for d in op.decimations)
                and all(s == 1 for s in op.scalings)):
            # with only sc16 output and no processing, can use sc16 as cpu
            # format and disable conversion
            op.cpu_format = 'sc16'
            op.out_specs = [dict(
                convert=None,
                convert_kwargs=None,
                dtype=np.dtype([('r', np.int16), ('i', np.int16)]),
                name='sc16',
            )]
        else:
            op.cpu_format = 'fc32'
            # get full specification for output types
            supported_out_types = {
                'sc8': dict(
                    convert='float_to_char',
                    convert_kwargs=dict(vlen=2, scale=float(2**7-1)),
                    dtype=np.dtype([('r', np.int8), ('i', np.int8)]),
                    name='sc8',
                ),
                'sc16': dict(
                    convert='float_to_short',
                    convert_kwargs=dict(vlen=2, scale=float(2**15-1)),
                    dtype=np.dtype([('r', np.int16), ('i', np.int16)]),
                    name='sc16',
                ),
                'sc32': dict(
                    convert='float_to_int',
                    convert_kwargs=dict(vlen=2, scale=float(2**31-1)),
                    dtype=np.dtype([('r', np.int32), ('i', np.int32)]),
                    name='sc32',
                ),
                'fc32': dict(
                    convert=None,
                    convert_kwargs=None,
                    dtype=np.dtype('complex64'),
                    name='fc32',
                ),
            }
            supported_out_types[None] = supported_out_types['fc32']
            type_dicts = []
            for ot in op.out_types:
                try:
                    type_dict = supported_out_types[ot]
                except KeyError:
                    errstr = (
                        'Output type {0} is not supported. Must be one of {1}.'
                    ).format(ot, supported_out_types.keys())
                    raise ValueError(errstr)
                else:
                    type_dicts.append(type_dict)
            op.out_specs = type_dicts
        # replace out_types to fill in None values with type name
        op.out_types = [os['name'] for os in op.out_specs]

        # repeat mainboard arguments as necessary
        op.nmboards = len(op.mboards) if len(op.mboards) > 0 else 1
        op.subdevs = list(islice(cycle(op.subdevs), 0, op.nmboards))

        # get number of receiver channels by total number of subdevices over
        # all mainboards
        op.mboards_bychan = []
        op.subdevs_bychan = []
        op.mboardnum_bychan = []
        mboards = op.mboards if op.mboards else ['default']
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
            'antennas', 'bandwidths', 'centerfreqs', 'dc_offsets',
            'iq_balances', 'lo_offsets', 'lo_sources', 'lo_exports', 'gains',
        ):
            val = getattr(op, rch_arg)
            rval = list(islice(cycle(val), 0, op.nrchs))
            setattr(op, rch_arg, rval)

        # repeat output channel arguments as necessary
        op.nochs = len(op.channel_names)
        for och_arg in (
            'channels', 'decimations', 'out_specs', 'out_types', 'scalings',
        ):
            val = getattr(op, och_arg)
            rval = list(islice(cycle(val), 0, op.nochs))
            setattr(op, och_arg, rval)

        # fill in unspecified (None) channels values
        rchannels = set(range(op.nrchs))
        ochannels = set(c for c in op.channels if c is not None)
        if not ochannels.issubset(rchannels):
            errstr = (
                'Invalid channel specification. Output channel uses'
                ' non-existent receiver channel: {0}.'
            )
            raise ValueError(errstr.format(list(ochannels - rchannels)))
        avail = sorted(rchannels - ochannels)
        try:
            op.channels = [
                c if c is not None else avail.pop(0) for c in op.channels
            ]
        except IndexError:
            errstr = (
                'No remaining receiver channels left to assign to unspecified'
                ' (None) output channel. You probably need to explicitly'
                ' specify the receiver channels to output.'
            )
            raise ValueError(errstr)
        unused_rchs = set(range(op.nrchs)) - set(op.channels)
        if unused_rchs:
            errstr = (
                'Receiver channels {0} are unused in the output. Either'
                ' remove them from the mainboard/subdevice specification or'
                ' correct the output channel specification.'
            )
            raise ValueError(errstr.format(unused_rchs))

        # create device_addr string to identify the requested device(s)
        op.mboard_strs = []
        for n, mb in enumerate(op.mboards):
            if re.match(r'[^0-9]+=.+', mb):
                idtype, mb = mb.split('=')
            elif re.match(
                r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', mb
            ):
                idtype = 'addr'
            elif (
                re.match(r'usrp[123]', mb) or re.match(r'b2[01]0', mb)
                or re.match(r'x3[01]0', mb)
            ):
                idtype = 'type'
            elif re.match(r'[0-9A-Fa-f]{1,}', mb):
                idtype = 'serial'
            else:
                idtype = 'name'
            if len(op.mboards) == 1:
                # do not use identifier numbering if only using one mainboard
                s = '{type}={mb}'.format(type=idtype, mb=mb.strip())
            else:
                s = '{type}{n}={mb}'.format(type=idtype, n=n, mb=mb.strip())
            op.mboard_strs.append(s)

        if op.verbose:
            opstr = dedent('''\
                Main boards: {mboard_strs}
                Subdevices: {subdevs}
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
                Channel names: {channel_names}
                Output decimation: {decimations}
                Output scaling: {scalings}
                Output type: {out_types}
                Data dir: {datadir}
                Metadata: {metadata}
                UUID: {uuid}
            ''').strip().format(**op.__dict__)
            print(opstr)

        return op

    def _usrp_setup(self):
        """Create, set up, and return USRP source object."""
        op = self.op
        # create usrp source block
        op.otw_format = 'sc16'
        u = uhd.usrp_source(
            device_addr=','.join(chain(op.mboard_strs, op.dev_args)),
            stream_args=uhd.stream_args(
                cpu_format=op.cpu_format,
                otw_format=op.otw_format,
                channels=range(op.nrchs),
                args=','.join(op.stream_args)
            ),
        )

        # set clock and time source if synced
        if op.sync:
            try:
                u.set_clock_source(op.sync_source, uhd.ALL_MBOARDS)
                u.set_time_source(op.sync_source, uhd.ALL_MBOARDS)
            except RuntimeError:
                errstr = (
                    "Setting sync_source to '{0}' failed. Must be one of {1}."
                    " If setting is valid, check that the source (REF, PPS) is"
                    " operational."
                ).format(op.sync_source, u.get_clock_sources(0))
                raise ValueError(errstr)

        # check for ref lock
        mbnums_with_ref = [
            mb_num for mb_num in range(op.nmboards)
            if 'ref_locked' in u.get_mboard_sensor_names(mb_num)
        ]
        if mbnums_with_ref:
            if op.verbose:
                sys.stdout.write('Waiting for reference lock...')
                sys.stdout.flush()
            timeout = 0
            while not all(
                u.get_mboard_sensor('ref_locked', mb_num).to_bool()
                for mb_num in mbnums_with_ref
            ):
                if op.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                time.sleep(1)
                timeout += 1
                if timeout > 30:
                    if op.verbose:
                        sys.stdout.write('failed\n')
                        sys.stdout.flush()
                    raise RuntimeError(
                        'Failed to lock to 10 MHz reference.'
                    )
            if op.verbose:
                sys.stdout.write('locked\n')
                sys.stdout.flush()

        # set mainboard options
        for mb_num in range(op.nmboards):
            u.set_subdev_spec(op.subdevs[mb_num], mb_num)
        # set global options
        # sample rate
        u.set_samp_rate(float(op.samplerate))
        # read back actual value
        samplerate = u.get_samp_rate()
        # calculate longdouble precision sample rate
        # (integer division of clock rate)
        cr = u.get_clock_rate()
        srdec = int(round(cr / samplerate))
        samplerate_ld = np.longdouble(cr) / srdec
        op.samplerate = samplerate_ld
        sr_rat = Fraction(cr).limit_denominator() / srdec
        op.samplerate_num = sr_rat.numerator
        op.samplerate_den = sr_rat.denominator

        # set per-channel options
        # set command time so settings are synced
        COMMAND_DELAY = 0.2
        cmd_time = u.get_time_now() + uhd.time_spec(COMMAND_DELAY)
        u.set_command_time(cmd_time, uhd.ALL_MBOARDS)
        for ch_num in range(op.nrchs):
            # local oscillator sharing settings
            lo_source = op.lo_sources[ch_num]
            if lo_source:
                try:
                    u.set_lo_source(lo_source, uhd.ALL_LOS, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown LO source option: '{0}'. Must be one of {1},"
                        " or it may not be possible to set the LO source on"
                        " this daughterboard."
                    ).format(lo_source, u.get_lo_sources(uhd.ALL_LOS, ch_num))
                    raise ValueError(errstr)
            lo_export = op.lo_exports[ch_num]
            if lo_export is not None:
                if not lo_source:
                    errstr = (
                        'Channel {0}: must set an LO source in order to set'
                        ' LO export.'
                    ).format(ch_num)
                    raise ValueError(errstr)
                u.set_lo_export_enabled(lo_export, uhd.ALL_LOS, ch_num)
            # center frequency and tuning offset
            tune_res = u.set_center_freq(
                uhd.tune_request(
                    op.centerfreqs[ch_num], op.lo_offsets[ch_num],
                    args=uhd.device_addr(','.join(op.tune_args)),
                ),
                ch_num,
            )
            # store actual values from tune result
            op.centerfreqs[ch_num] = (
                tune_res.actual_rf_freq - tune_res.actual_dsp_freq
            )
            op.lo_offsets[ch_num] = tune_res.actual_dsp_freq
            # dc offset
            dc_offset = op.dc_offsets[ch_num]
            if dc_offset is True:
                u.set_auto_dc_offset(True, ch_num)
            elif dc_offset is False:
                u.set_auto_dc_offset(False, ch_num)
            elif dc_offset is not None:
                u.set_auto_dc_offset(False, ch_num)
                u.set_dc_offset(dc_offset, ch_num)
            # iq balance
            iq_balance = op.iq_balances[ch_num]
            if iq_balance is True:
                u.set_auto_iq_balance(True, ch_num)
            elif iq_balance is False:
                u.set_auto_iq_balance(False, ch_num)
            elif iq_balance is not None:
                u.set_auto_iq_balance(False, ch_num)
                u.set_iq_balance(iq_balance, ch_num)
            # gain
            u.set_gain(op.gains[ch_num], ch_num)
            # bandwidth
            bw = op.bandwidths[ch_num]
            if bw:
                u.set_bandwidth(bw, ch_num)
            # antenna
            ant = op.antennas[ch_num]
            if ant:
                try:
                    u.set_antenna(ant, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown RX antenna option: '{0}'. Must be one of {1}."
                    ).format(ant, u.get_antennas(ch_num))
                    raise ValueError(errstr)

        # commands are done, clear time
        u.clear_command_time(uhd.ALL_MBOARDS)
        time.sleep(COMMAND_DELAY)

        # read back actual channel settings
        for ch_num in range(op.nrchs):
            if op.lo_sources[ch_num]:
                op.lo_sources[ch_num] = u.get_lo_source(uhd.ALL_LOS, ch_num)
            if op.lo_exports[ch_num] is not None:
                op.lo_exports[ch_num] = u.get_lo_export_enabled(
                    uhd.ALL_LOS, ch_num,
                )
            op.gains[ch_num] = u.get_gain(ch_num)
            op.bandwidths[ch_num] = u.get_bandwidth(chan=ch_num)
            op.antennas[ch_num] = u.get_antenna(chan=ch_num)

        if op.verbose:
            print('Using the following devices:')
            chinfostrs = [
                'Motherboard: {mb_id} ({mb_addr}) | Daughterboard: {db_name}',
                'Subdev: {sub} | Antenna: {ant} | Gain: {gain} | Rate: {sr}',
                'Frequency: {freq:.3f} ({lo_off:+.3f}) | Bandwidth: {bw}',
            ]
            if any(op.lo_sources) or any(op.lo_exports):
                chinfostrs.append(
                    'LO source: {lo_source} | LO export: {lo_export}'
                )
            chinfo = '\n'.join(['  ' + l for l in chinfostrs])
            for ch_num in range(op.nrchs):
                header = '---- receiver channel {0} '.format(ch_num)
                header += '-' * (78 - len(header))
                print(header)
                usrpinfo = dict(u.get_usrp_info(chan=ch_num))
                info = {}
                info['mb_id'] = usrpinfo['mboard_id']
                mba = op.mboards_bychan[ch_num]
                if mba == 'default':
                    mba = usrpinfo['mboard_serial']
                info['mb_addr'] = mba
                info['db_name'] = usrpinfo['rx_subdev_name']
                info['sub'] = op.subdevs_bychan[ch_num]
                info['ant'] = op.antennas[ch_num]
                info['bw'] = op.bandwidths[ch_num]
                info['freq'] = op.centerfreqs[ch_num]
                info['gain'] = op.gains[ch_num]
                info['lo_off'] = op.lo_offsets[ch_num]
                info['lo_source'] = op.lo_sources[ch_num]
                info['lo_export'] = op.lo_exports[ch_num]
                info['sr'] = op.samplerate
                print(chinfo.format(**info))
                print('-' * 78)

        return u

    def run(self, starttime=None, endtime=None, duration=None, period=10):
        op = self.op

        # window in seconds that we allow for setup time so that we don't
        # issue a start command that's in the past when the flowgraph starts
        SETUP_TIME = 10

        # print current time and NTP status
        if op.verbose and sys.platform.startswith('linux'):
            try:
                call(('timedatectl', 'status'))
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
                ststr = st.strftime('%a %b %d %H:%M:%S %Y')
                stts = (st - drf.util.epoch).total_seconds()
                print('Start time: {0} ({1})'.format(ststr, stts))

        et = drf.util.parse_identifier_to_time(endtime, ref_datetime=st)
        if et is not None:
            if op.verbose:
                etstr = et.strftime('%a %b %d %H:%M:%S %Y')
                etts = (et - drf.util.epoch).total_seconds()
                print('End time: {0} ({1})'.format(etstr, etts))

            if ((et < (pytz.utc.localize(datetime.utcnow())
                       + timedelta(seconds=SETUP_TIME)))
               or (st is not None and et <= st)):
                raise ValueError('End time is before launch time!')

        if op.realtime:
            r = gr.enable_realtime_scheduling()

            if op.verbose:
                if r == gr.RT_OK:
                    print('Realtime scheduling enabled')
                else:
                    print('Note: failed to enable realtime scheduling')

        # create data directory so ringbuffer code can be started while waiting
        # to launch
        if not os.path.isdir(op.datadir):
            os.makedirs(op.datadir)

        # wait for the start time if it is not past
        while (st is not None) and (
            (st - pytz.utc.localize(datetime.utcnow())) >
                timedelta(seconds=SETUP_TIME)
        ):
            ttl = int((
                st - pytz.utc.localize(datetime.utcnow())
            ).total_seconds())
            if (ttl % 10) == 0:
                print('Standby {0} s remaining...'.format(ttl))
                sys.stdout.flush()
            time.sleep(1)

        # get UHD USRP source
        u = self._usrp_setup()

        # force creation of the RX streamer ahead of time with a start/stop
        # (after setting time/clock sources, before setting the
        # device time)
        # this fixes timing with the B210
        u.start()
        # need to wait >0.1 s (constant in usrp_source_impl.c) for start/stop
        # to actually take effect, so sleep a bit, 0.5 s seems more reliable
        time.sleep(0.5)
        u.stop()
        time.sleep(0.2)

        # set device time
        tt = time.time()
        if op.sync:
            # wait until time 0.2 to 0.5 past full second, then latch
            # we have to trust NTP to be 0.2 s accurate
            while tt - math.floor(tt) < 0.2 or tt - math.floor(tt) > 0.3:
                time.sleep(0.01)
                tt = time.time()
            if op.verbose:
                print('Latching at ' + str(tt))
            # waits for the next pps to happen
            # (at time math.ceil(tt))
            # then sets the time for the subsequent pps
            # (at time math.ceil(tt) + 1.0)
            u.set_time_unknown_pps(uhd.time_spec(math.ceil(tt) + 1.0))
        else:
            u.set_time_now(uhd.time_spec(tt), uhd.ALL_MBOARDS)

        # set launch time
        # (at least 1 second out so USRP start time can be set properly and
        #  there is time to set up flowgraph)
        if st is not None:
            lt = st
        else:
            now = pytz.utc.localize(datetime.utcnow())
            # launch on integer second by default for convenience (ceil + 1)
            lt = now.replace(microsecond=0) + timedelta(seconds=2)
        ltts = (lt - drf.util.epoch).total_seconds()
        # adjust launch time forward so it falls on an exact sample since epoch
        lt_rsamples = np.ceil(ltts * op.samplerate)
        ltts = lt_rsamples / op.samplerate
        lt = drf.util.sample_to_datetime(lt_rsamples, op.samplerate)
        if op.verbose:
            ltstr = lt.strftime('%a %b %d %H:%M:%S.%f %Y')
            print('Launch time: {0} ({1})'.format(ltstr, repr(ltts)))
        # command launch time
        ct_td = lt - drf.util.epoch
        ct_secs = ct_td.total_seconds() // 1.0
        ct_frac = ct_td.microseconds / 1000000.0
        u.set_start_time(
            uhd.time_spec(ct_secs) + uhd.time_spec(ct_frac)
        )

        # populate flowgraph one channel at a time
        fg = gr.top_block()
        for ko in range(op.nochs):
            # receiver channel number corresponding to this output channel
            kr = op.channels[ko]
            # mainboard number corresponding to this receiver's channel
            mbnum = op.mboardnum_bychan[kr]

            # get output settings
            ot_dict = op.out_specs[ko]
            converter = ot_dict['convert']
            osample_dtype = ot_dict['dtype']
            scaling = conv_scaling = op.scalings[ko]
            dec = op.decimations[ko]
            samplerate_out = op.samplerate / dec
            samplerate_out_fr = Fraction(
                op.samplerate_num, op.samplerate_den * dec,
            )
            samplerate_num_out = samplerate_out_fr.numerator
            samplerate_den_out = samplerate_out_fr.denominator
            start_sample = int(np.uint64(ltts * samplerate_out))
            if dec > 1:
                # (integrate scaling into filter taps)
                taps = firdes.low_pass_2(
                    scaling, float(op.samplerate),
                    float(samplerate_out / 2.0), float(0.2 * samplerate_out),
                    80.0, window=firdes.WIN_BLACKMAN_HARRIS,
                )
                conv_scaling = 1.0
                # create low-pass filter
                lpf = freq_xlating_fir_filter_ccf(
                    dec, taps, 0.0, float(op.samplerate)
                )
            else:
                lpf = None

            if converter is not None:
                kw = ot_dict['convert_kwargs']
                # incorporate any scaling into type conversion block
                kw['scale'] *= conv_scaling
                convert = getattr(blocks, converter)(**kw)
            elif conv_scaling != 1:
                convert = blocks.multiply_const_vcc(conv_scaling)
            else:
                convert = None

            # create digital RF sink
            dst = gr_drf.digital_rf_channel_sink(
                channel_dir=os.path.join(op.datadir, op.channel_names[ko]),
                dtype=osample_dtype,
                subdir_cadence_secs=op.subdir_cadence_s,
                file_cadence_millisecs=op.file_cadence_ms,
                sample_rate_numerator=samplerate_num_out,
                sample_rate_denominator=samplerate_den_out,
                start=start_sample,
                ignore_tags=False,
                is_complex=True,
                num_subchannels=1,
                uuid_str=op.uuid,
                center_frequencies=op.centerfreqs[kr],
                metadata=dict(
                    # receiver metadata for USRP
                    receiver=dict(
                        description='UHD USRP source using GNU Radio',
                        info=dict(u.get_usrp_info(chan=kr)),
                        antenna=op.antennas[kr],
                        bandwidth=op.bandwidths[kr],
                        center_freq=op.centerfreqs[kr],
                        clock_rate=u.get_clock_rate(mboard=mbnum),
                        clock_source=u.get_clock_source(mboard=mbnum),
                        dc_offset=op.dc_offsets[kr],
                        gain=op.gains[kr],
                        id=op.mboards_bychan[kr],
                        iq_balance=op.iq_balances[kr],
                        lo_offset=op.lo_offsets[kr],
                        otw_format=op.otw_format,
                        samp_rate=u.get_samp_rate(),
                        stream_args=','.join(op.stream_args),
                        subdev=op.subdevs_bychan[kr],
                        time_source=u.get_time_source(mboard=mbnum),
                    ),
                    processing=dict(
                        decimation=op.decimations[ko],
                        scaling=op.scalings[ko],
                    ),
                ),
                is_continuous=True,
                compression_level=0,
                checksum=False,
                marching_periods=True,
                stop_on_skipped=op.stop_on_dropped,
                debug=op.verbose,
            )

            connections = [(u, kr)]
            if lpf is not None:
                connections.append((lpf, 0))
            if convert is not None:
                connections.append((convert, 0))
            connections.append((dst, 0))
            connections = tuple(connections)

            # make channel connections in flowgraph
            fg.connect(*connections)

        # start the flowgraph once we are near the launch time
        # (start too soon and device buffers might not yet be flushed)
        # (start too late and device might not be able to start in time)
        while ((lt - pytz.utc.localize(datetime.utcnow()))
                > timedelta(seconds=1.2)):
            time.sleep(0.1)
        fg.start()

        # wait until end time or until flowgraph stops
        if et is None and duration is not None:
            et = lt + timedelta(seconds=duration)
        try:
            if et is None:
                fg.wait()
            else:
                # sleep until end time nears
                while(pytz.utc.localize(datetime.utcnow()) <
                        et - timedelta(seconds=2)):
                    time.sleep(1)
                else:
                    # issue stream stop command at end time
                    ct_td = et - drf.util.epoch
                    ct_secs = ct_td.total_seconds() // 1.0
                    ct_frac = ct_td.microseconds / 1000000.0
                    u.set_command_time(
                        (uhd.time_spec(ct_secs) + uhd.time_spec(ct_frac)),
                        uhd.ALL_MBOARDS,
                    )
                    stop_enum = uhd.stream_cmd.STREAM_MODE_STOP_CONTINUOUS
                    u.issue_stream_cmd(uhd.stream_cmd(stop_enum))
                    u.clear_command_time(uhd.ALL_MBOARDS)
                    # sleep until after end time
                    time.sleep(2)
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        fg.stop()
        # need to wait for the flowgraph to clean up, otherwise it won't exit
        fg.wait()
        print('done')
        sys.stdout.flush()


def evalint(s):
    """Evaluate string to an integer."""
    return int(eval(s, {}, {}))


def evalfloat(s):
    """Evaluate string to a float."""
    return float(eval(s, {}, {}))


def intstrtuple(s):
    """Get (int, string) tuple from int:str strings."""
    parts = [p.strip() for p in s.split(':', 1)]
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    else:
        return None, parts[0]


def noneorstr(s):
    """Turn empty or 'none' string to None."""
    if s.lower() in ('', 'none'):
        return None
    else:
        return s


def noneorbool(s):
    """Turn empty or 'none' string to None, all others to boolean."""
    if s.lower() in ('', 'none'):
        return None
    elif s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        return False


def noneorboolorcomplex(s):
    """Turn empty or 'none' to None, else evaluate to a boolean or complex."""
    if s.lower() in ('', 'none'):
        return None
    elif s.lower() in ('auto', 'true', 't', 'yes', 'y'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n'):
        return False
    else:
        return complex(eval(s, {}, {}))


class Extend(Action):
    """Action to split comma-separated arguments and add to a list."""

    def __init__(self, option_strings, dest, type=None, **kwargs):
        if type is not None:
            itemtype = type
        else:
            def itemtype(s):
                return s

        def split_string_and_cast(s):
            return [itemtype(a.strip()) for a in s.strip().split(',')]

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
        'datadir', nargs='?', default=None,
        help='''Data directory, to be filled with channel subdirectories.''',
    )
    dirgroup.add_argument(
        '-o', '--out', dest='outdir', default=None,
        help='''Data directory, to be filled with channel subdirectories.''',
    )
    return parser


def _add_mainboard_group(parser):
    mbgroup = parser.add_argument_group(title='mainboard')
    mbgroup.add_argument(
        '-m', '--mainboard', dest='mboards', action=Extend,
        help='''Mainboard address. (default: first device found)''',
    )
    mbgroup.add_argument(
        '-d', '--subdevice', dest='subdevs', action=Extend,
        help='''USRP subdevice string. (default: "A:A")''',
    )
    return parser


def _add_receiver_group(parser):
    recgroup = parser.add_argument_group(title='receiver')
    recgroup.add_argument(
        '-r', '--samplerate', dest='samplerate', type=evalfloat,
        help='''Sample rate in Hz. (default: 1e6)''',
    )
    recgroup.add_argument(
        '-A', '--devargs', dest='dev_args', action=Extend,
        help='''Device arguments, e.g. "master_clock_rate=30e6".
                (default: 'recv_buff_size=100000000,num_recv_frames=512')''',
    )
    recgroup.add_argument(
        '-a', '--streamargs', dest='stream_args', action=Extend,
        help='''Stream arguments, e.g. "peak=0.125,fullscale=1.0".
                (default: '')''',
    )
    recgroup.add_argument(
        '-T', '--tuneargs', dest='tune_args', action=Extend,
        help='''Tune request arguments, e.g. "mode_n=integer,int_n_step=100e3".
                (default: '')''',
    )
    recgroup.add_argument(
        '--sync_source', dest='sync_source',
        help='''Clock and time source for all mainboards.
                (default: external)''',
    )
    recgroup.add_argument(
        '--nosync', dest='sync', action='store_false',
        help='''No syncing with external clock. (default: False)''',
    )
    recgroup.add_argument(
        '--stop_on_dropped', dest='stop_on_dropped', action='store_true',
        help='''Stop on dropped packet. (default: %(default)s)''',
    )
    recgroup.add_argument(
        '--realtime', dest='realtime', action='store_true',
        help='''Enable realtime scheduling if possible.
                (default: %(default)s)''',
    )
    recgroup.add_argument(
        '--notest', dest='test_settings', action='store_false',
        help='''Do not test USRP settings until experiment start.
                (default: False)''',
    )
    return parser


def _add_rchannel_group(parser):
    chgroup = parser.add_argument_group(title='receiver channel')
    chgroup.add_argument(
        '-f', '--centerfreq', dest='centerfreqs', action=Extend,
        type=evalfloat,
        help='''Center frequency in Hz. (default: 100e6)''',
    )
    chgroup.add_argument(
        '-F', '--lo_offset', dest='lo_offsets', action=Extend, type=evalfloat,
        help='''Frontend tuner offset from center frequency, in Hz.
                (default: 0)''',
    )
    chgroup.add_argument(
        '--lo_source', dest='lo_sources', action=Extend, type=noneorstr,
        help='''Local oscillator source. Typically 'None'/'' (do not set),
                'internal' (e.g. LO1 for CH1, LO2 for CH2),
                'companion' (e.g. LO2 for CH1, LO1 for CH2), or
                'external' (neighboring board via connector).
                (default: '')''',
    )
    chgroup.add_argument(
        '--lo_export', dest='lo_exports', action=Extend, type=noneorbool,
        help='''Whether to export the LO's source to the external connector.
                Can be 'None'/'' to skip the channel, otherwise it can be
                'True' or 'False' provided the LO source is set.
                (default: None)''',
    )
    chgroup.add_argument(
        '--dc_offset', dest='dc_offsets', action=Extend,
        type=noneorboolorcomplex,
        help='''DC offset correction to use. Can be 'None'/'' to keep device
                default, 'True'/'auto' to enable automatic correction, 'False'
                to disable automatic correction, or a complex value
                (e.g. "1+1j"). (default: False)''',
    )
    chgroup.add_argument(
        '--iq_balance', dest='iq_balances', action=Extend,
        type=noneorboolorcomplex,
        help='''IQ balance correction to use. Can be 'None'/'' to keep device
                default, 'True'/'auto' to enable automatic correction, 'False'
                to disable automatic correction, or a complex value
                (e.g. "1+1j"). (default: None)''',
    )
    chgroup.add_argument(
        '-g', '--gain', dest='gains', action=Extend, type=evalfloat,
        help='''Gain in dB. (default: 0)''',
    )
    chgroup.add_argument(
        '-b', '--bandwidth', dest='bandwidths', action=Extend, type=evalfloat,
        help='''Frontend bandwidth in Hz. (default: 0 == frontend default)''',
    )
    chgroup.add_argument(
        '-y', '--antenna', dest='antennas', action=Extend, type=noneorstr,
        help='''Name of antenna to select on the frontend.
                (default: frontend default))''',
    )
    return parser


def _add_ochannel_group(parser):
    chgroup = parser.add_argument_group(title='output channel')
    chgroup.add_argument(
        '-c', '--channel', dest='chs', action=Extend, type=intstrtuple,
        help='''Output channel specification, including names and mapping from
                receiver channels. Each output channel must be specified here
                and given a unique name. Specifications are given as a receiver
                channel number and name pair, e.g. "0:ch0". The number and
                colon are optional; if omitted, any unused receiver channels
                will be assigned to output channels in the supplied name order.
                (default: "ch0")''',
    )
    chgroup.add_argument(
        '-i', '--dec', '--decimate', dest='decimations', action=Extend,
        type=evalint,
        help='''Integrate and decimate by an output channel by this factor
                using a low-pass filter. (default: 1)''',
    )
    chgroup.add_argument(
        '--scale', dest='scalings', action=Extend, type=evalfloat,
        help='''Scale an output channel by this factor. (default: 1)''',
    )
    chgroup.add_argument(
        '--type', dest='out_types', action=Extend, type=noneorstr,
        help='''Output channel data type to convert to ('scXX' for complex
                integer and 'fcXX' for complex float with XX bits). Use 'None'
                to skip conversion and use the USRP or filter output type.
                Conversion from float to integer will map a magnitude of 1.0
                (after any scaling) to the maximum integer value.
                (default: None)''',
    )
    return parser


def _add_drf_group(parser):
    drfgroup = parser.add_argument_group(title='digital_rf')
    drfgroup.add_argument(
        '-n', '--file_cadence_ms', dest='file_cadence_ms', type=evalint,
        help='''Number of milliseconds of data per file.
                (default: 1000)''',
    )
    drfgroup.add_argument(
        '-N', '--subdir_cadence_s', dest='subdir_cadence_s', type=evalint,
        help='''Number of seconds of data per subdirectory.
                (default: 3600)''',
    )
    drfgroup.add_argument(
        '--metadata', action=Extend, metavar='{KEY}={VALUE}',
        help='''Key, value metadata pairs to include with data.
                (default: "")''',
    )
    drfgroup.add_argument(
        '--uuid', dest='uuid',
        help='''Unique ID string for this data collection.
                (default: random)''',
    )
    return parser


def _add_time_group(parser):
    timegroup = parser.add_argument_group(title='time')
    timegroup.add_argument(
        '-s', '--starttime', dest='starttime',
        help='''Start time of the experiment as datetime (if in ISO8601 format:
                2016-01-01T15:24:00Z) or Unix time (if float/int).
                (default: start ASAP)''',
    )
    timegroup.add_argument(
        '-e', '--endtime', dest='endtime',
        help='''End time of the experiment as datetime (if in ISO8601 format:
                2016-01-01T16:24:00Z) or Unix time (if float/int).
                (default: wait for Ctrl-C)''',
    )
    timegroup.add_argument(
        '-l', '--duration', dest='duration', type=evalint,
        help='''Duration of experiment in seconds. When endtime is not given,
                end this long after start time. (default: wait for Ctrl-C)''',
    )
    timegroup.add_argument(
        '-p', '--cycle-length', dest='period', type=evalint,
        help='''Repeat time of experiment cycle. Align to start of next cycle
                if start time has passed. (default: 10)''',
    )
    return parser


def _build_thor_parser(Parser, *args):
    scriptname = os.path.basename(sys.argv[0])

    formatter = RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = 'THOR (The Haystack Observatory Recorder)'
    copyright = 'Copyright (c) 2017 Massachusetts Institute of Technology'
    shortdesc = 'Record data from synchronized USRPs in DigitalRF format.'
    desc = '\n'.join((
        '*'*width,
        '*{0:^{1}}*'.format(title, width-2),
        '*{0:^{1}}*'.format(copyright, width-2),
        '*{0:^{1}}*'.format('', width-2),
        '*{0:^{1}}*'.format(shortdesc, width-2),
        '*'*width,
    ))

    usage = (
        '%(prog)s [-m MBOARD] [-d SUBDEV] [-c CH] [-y ANT] [-f FREQ]'
        ' [-F OFFSET] \\\n'
        '{0:8}[-g GAIN] [-b BANDWIDTH] [-r RATE] [options] DIR\n'.format('')
    )

    epi_pars = [
        '''\
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
        ''',
        '''\
        Arguments in other groups apply to all mainboards/channels (including
        the receiver sample rate).
        ''',
        '''\
        Example usage:
        ''',
    ]
    epi_pars = [fill(dedent(s), width) for s in epi_pars]

    egtw = TextWrapper(
        width=(width - 2), break_long_words=False, break_on_hyphens=False,
        subsequent_indent=' ' * (len(scriptname) + 1),
    )
    egs = [
        '''\
        {0} -m 192.168.20.2 -d "A:A A:B" -c h,v -f 95e6 -r 100e6/24
        /data/test
        ''',
        '''\
        {0} -m 192.168.10.2 -d "A:0" -c ch1 -y "TX/RX" -f 20e6 -F 10e3 -g 20
        -b 0 -r 1e6 /data/test
        ''',
    ]
    egs = [' \\\n'.join(egtw.wrap(dedent(s.format(scriptname)))) for s in egs]
    epi = '\n' + '\n\n'.join(epi_pars + egs) + '\n'

    # parse options
    parser = Parser(
        description=desc, usage=usage, epilog=epi,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--version', action='version',
        version='THOR 3.1, using digital_rf {0}'.format(drf.__version__),
    )
    parser.add_argument(
        '-q', '--quiet', dest='verbose', action='store_false',
        help='''Reduce text output to the screen. (default: False)''',
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
    if args.datadir is None:
        args.datadir = args.outdir
    del args.outdir

    # separate args.chs (num, name) tuples into args.channels and
    # args.channel_names
    if args.chs is not None:
        args.channels, args.channel_names = map(list, zip(*args.chs))
    del args.chs

    # remove redundant arguments in dev_args, stream_args, tune_args
    if args.dev_args is not None:
        try:
            dev_args_dict = dict([a.split('=') for a in args.dev_args])
        except ValueError:
            raise ValueError(
                'Device arguments must be {KEY}={VALUE} pairs.'
            )
        args.dev_args = [
            '{0}={1}'.format(k, v) for k, v in dev_args_dict.iteritems()
        ]
    if args.stream_args is not None:
        try:
            stream_args_dict = dict([a.split('=') for a in args.stream_args])
        except ValueError:
            raise ValueError(
                'Stream arguments must be {KEY}={VALUE} pairs.'
            )
        args.stream_args = [
            '{0}={1}'.format(k, v) for k, v in stream_args_dict.iteritems()
        ]
    if args.tune_args is not None:
        try:
            tune_args_dict = dict([a.split('=') for a in args.tune_args])
        except ValueError:
            raise ValueError(
                'Tune request arguments must be {KEY}={VALUE} pairs.'
            )
        args.tune_args = [
            '{0}={1}'.format(k, v) for k, v in tune_args_dict.iteritems()
        ]

    # convert metadata strings to a dictionary
    if args.metadata is not None:
        metadata_dict = {}
        for a in args.metadata:
            try:
                k, v = a.split('=')
            except ValueError:
                k = None
                v = a
            try:
                v = literal_eval(v)
            except ValueError:
                pass
            if k is None:
                metadata_dict.setdefault('metadata', []).append(v)
            else:
                metadata_dict[k] = v
        args.metadata = metadata_dict

    # ignore test_settings option if no starttime is set (starting right now)
    if args.starttime is None:
        args.test_settings = False

    options = {k: v for k, v in args._get_kwargs() if v is not None}
    runopts = {
        k: options.pop(k) for k in list(options.keys())
        if k in ('starttime', 'endtime', 'duration', 'period')
    }
    del options['func']
    thor = Thor(**options)
    thor.run(**runopts)


if __name__ == '__main__':
    parser = _build_thor_parser(ArgumentParser)
    args = parser.parse_args()
    args.func(args)
