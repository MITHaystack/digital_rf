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
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from ast import literal_eval
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import TextWrapper, dedent, fill

import numpy as np
import pytz
from gnuradio import filter, gr, uhd
from gnuradio.filter import firdes

import digital_rf as drf
import gr_drf


class Thor(object):
    """Record data from synchronized USRPs in DigitalRF format."""

    def __init__(
        self, datadir, mboards=[], subdevs=['A:A'],
        chs=['ch0'], centerfreqs=[100e6], lo_offsets=[0],
        lo_sources=[''], lo_exports=[None],
        gains=[0], bandwidths=[0], antennas=[''],
        samplerate=1e6, dec=1,
        dev_args=['recv_buff_size=100000000', 'num_recv_frames=512'],
        stream_args=[],
        sync=True, sync_source='external',
        stop_on_dropped=False, realtime=False,
        file_cadence_ms=1000, subdir_cadence_s=3600, metadata={}, uuid=None,
        verbose=True, test_settings=True,
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

        op.nmboards = len(op.mboards) if len(op.mboards) > 0 else 1
        op.nchs = len(op.chs)
        # repeat arguments as necessary
        op.subdevs = list(islice(cycle(op.subdevs), 0, op.nmboards))
        op.centerfreqs = list(islice(cycle(op.centerfreqs), 0, op.nchs))
        op.lo_offsets = list(islice(cycle(op.lo_offsets), 0, op.nchs))
        op.lo_sources = list(islice(cycle(op.lo_sources), 0, op.nchs))
        op.lo_exports = list(islice(cycle(op.lo_exports), 0, op.nchs))
        op.gains = list(islice(cycle(op.gains), 0, op.nchs))
        op.bandwidths = list(islice(cycle(op.bandwidths), 0, op.nchs))
        op.antennas = list(islice(cycle(op.antennas), 0, op.nchs))

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
                Channel names: {chs}
                Frequency: {centerfreqs}
                LO frequency offset: {lo_offsets}
                LO source: {lo_sources}
                LO export: {lo_exports}
                Gain: {gains}
                Bandwidth: {bandwidths}
                Antenna: {antennas}
                Device arguments: {dev_args}
                Stream arguments: {stream_args}
                Sample rate: {samplerate}
                Data dir: {datadir}
                Metadata: {metadata}
                UUID: {uuid}
            ''').strip().format(**op.__dict__)
            print(opstr)

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

        # sanity check: # of total subdevs should be same as # of channels
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
        if len(op.subdevs_bychan) != op.nchs:
            raise ValueError(
                'Number of device channels does not match the number of '
                'channel names provided.'
            )

        return op

    def _usrp_setup(self):
        """Create, set up, and return USRP source object."""
        op = self.op
        # create usrp source block
        if op.dec > 1:
            op.cpu_format = 'fc32'
        else:
            op.cpu_format = 'sc16'
        op.otw_format = 'sc16'
        u = uhd.usrp_source(
            device_addr=','.join(chain(op.mboard_strs, op.dev_args)),
            stream_args=uhd.stream_args(
                cpu_format=op.cpu_format,
                otw_format=op.otw_format,
                channels=range(len(op.chs)),
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
                    "Unknown sync_source option: '{0}'. Must be one of {1}."
                ).format(op.sync_source, u.get_clock_sources(0))
                raise ValueError(errstr)

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
        for ch_num in range(op.nchs):
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
                ),
                ch_num,
            )
            # store actual values from tune result
            op.centerfreqs[ch_num] = (
                tune_res.actual_rf_freq - tune_res.actual_dsp_freq
            )
            op.lo_offsets[ch_num] = tune_res.actual_dsp_freq
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
        for ch_num in range(op.nchs):
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
                'Frequency: {freq} (+{lo_off}) | Bandwidth: {bw}',
            ]
            if any(op.lo_sources) or any(op.lo_exports):
                chinfostrs.append(
                    'LO source: {lo_source} | LO export: {lo_export}'
                )
            chinfo = '\n'.join(['  ' + l for l in chinfostrs])
            for ch_num in range(op.nchs):
                header = '---- {0} '.format(op.chs[ch_num])
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
        if op.verbose:
            call(('timedatectl', 'status'))

        # parse time arguments
        st = drf.util.parse_identifier_to_time(
            starttime, samples_per_second=op.samplerate,
        )
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

        et = drf.util.parse_identifier_to_time(
            endtime, samples_per_second=op.samplerate, ref_datetime=st,
        )
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

        # force creation of the RX streamer ahead of time with a finite
        # acquisition (after setting time/clock sources, before setting the
        # device time)
        # this fixes timing with the B210
        u.finite_acquisition_v(16384)

        # wait until time 0.2 to 0.5 past full second, then latch
        # we have to trust NTP to be 0.2 s accurate
        tt = time.time()
        while tt - math.floor(tt) < 0.2 or tt - math.floor(tt) > 0.3:
            time.sleep(0.01)
            tt = time.time()
        if op.verbose:
            print('Latching at ' + str(tt))
        if op.sync:
            # waits for the next pps to happen
            # (at time math.ceil(tt))
            # then sets the time for the subsequent pps
            # (at time math.ceil(tt) + 1.0)
            u.set_time_unknown_pps(uhd.time_spec(math.ceil(tt) + 1.0))
        else:
            u.set_time_now(uhd.time_spec(tt))
        # reset device stream and flush buffer to clear leftovers from finite
        # acquisition
        u.stop()

        # get output settings that depend on decimation rate
        samplerate_out = op.samplerate / op.dec
        samplerate_num_out = op.samplerate_num
        samplerate_den_out = op.samplerate_den * op.dec
        if op.dec > 1:
            sample_dtype = '<c8'

            taps = firdes.low_pass_2(
                1.0, float(op.samplerate), float(samplerate_out / 2.0),
                float(0.2 * samplerate_out), 80.0,
                window=firdes.WIN_BLACKMAN_hARRIS
            )
        else:
            sample_dtype = np.dtype([('r', '<i2'), ('i', '<i2')])

        # set launch time
        # (at least 1 second out so USRP time is set, time to set up flowgraph)
        if st is not None:
            lt = st
        else:
            now = pytz.utc.localize(datetime.utcnow())
            lt = now.replace(microsecond=0) + timedelta(seconds=2)
        ltts = (lt - drf.util.epoch).total_seconds()
        # adjust launch time forward so it falls on an exact sample since epoch
        lt_samples = np.ceil(ltts * samplerate_out)
        ltts = lt_samples / samplerate_out
        lt = drf.util.sample_to_datetime(lt_samples, op.samplerate)
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
        for k in range(op.nchs):
            mbnum = op.mboardnum_bychan[k]
            # create digital RF sink
            dst = gr_drf.digital_rf_channel_sink(
                channel_dir=os.path.join(op.datadir, op.chs[k]),
                dtype=sample_dtype,
                subdir_cadence_secs=op.subdir_cadence_s,
                file_cadence_millisecs=op.file_cadence_ms,
                sample_rate_numerator=samplerate_num_out,
                sample_rate_denominator=samplerate_den_out,
                start=lt_samples,
                ignore_tags=False,
                is_complex=True,
                num_subchannels=1,
                uuid_str=op.uuid,
                center_frequencies=op.centerfreqs[k],
                metadata=dict(
                    # receiver metadata for USRP
                    receiver=dict(
                        description='UHD USRP source using GNU Radio',
                        info=dict(u.get_usrp_info(chan=k)),
                        antenna=op.antennas[k],
                        bandwidth=op.bandwidths[k],
                        center_freq=op.centerfreqs[k],
                        clock_rate=u.get_clock_rate(mboard=mbnum),
                        clock_source=u.get_clock_source(mboard=mbnum),
                        gain=op.gains[k],
                        id=op.mboards_bychan[k],
                        lo_offset=op.lo_offsets[k],
                        otw_format=op.otw_format,
                        samp_rate=u.get_samp_rate(),
                        stream_args=','.join(op.stream_args),
                        subdev=op.subdevs_bychan[k],
                        time_source=u.get_time_source(mboard=mbnum),
                    ),
                ),
                is_continuous=True,
                compression_level=0,
                checksum=False,
                marching_periods=True,
                stop_on_skipped=op.stop_on_dropped,
                debug=op.verbose,
            )

            if op.dec > 1:
                # create low-pass filter
                lpf = filter.freq_xlating_fir_filter_ccf(
                    op.dec, taps, 0.0, float(op.samplerate)
                )

                # connections for usrp->lpf->drf
                connections = ((u, k), (lpf, 0), (dst, 0))
            else:
                # connections for usrp->drf
                connections = ((u, k), (dst, 0))

            # make channel connections in flowgraph
            fg.connect(*connections)

        # start to receive data
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


if __name__ == '__main__':
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
        Arguments in the "mainboard" and "channel" groups accept multiple
        values, allowing multiple mainboards and channels to be specified.
        Multiple arguments can be provided by repeating the argument flag, by
        passing a comma-separated list of values, or both. Within each argument
        group, parameters will be grouped in the order in which they are given
        to form the complete set of parameters for each mainboard/channel. For
        any argument with fewer values given than the number of
        mainboards/channels, its values will be extended by repeatedly cycling
        through the values given up to the needed number.
        ''',
        '''\
        Arguments in other groups apply to all mainboards/channels (including
        the sample rate and decimation factor).
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
    parser = ArgumentParser(description=desc, usage=usage, epilog=epi,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        '--version', action='version',
        version='THOR 3.1, using digital_rf {0}'.format(drf.__version__),
    )
    parser.add_argument(
        '-q', '--quiet', dest='verbose', action='store_false',
        help='''Reduce text output to the screen. (default: False)''',
    )
    parser.add_argument(
        '--notest', dest='test_settings', action='store_false',
        help='''Do not test USRP settings until experiment start.
                (default: False)''',
    )

    dirgroup = parser.add_mutually_exclusive_group(required=True)
    dirgroup.add_argument(
        'datadir', nargs='?', default=None,
        help='''Data directory, to be filled with channel subdirectories.''',
    )
    dirgroup.add_argument(
        '-o', '--out', dest='outdir', default=None,
        help='''Data directory, to be filled with channel subdirectories.''',
    )

    mbgroup = parser.add_argument_group(title='mainboard')
    mbgroup.add_argument(
        '-m', '--mainboard', dest='mboards', action='append',
        help='''Mainboard address. (default: first device found)''',
    )
    mbgroup.add_argument(
        '-d', '--subdevice', dest='subdevs', action='append',
        help='''USRP subdevice string. (default: "A:A")''',
    )

    chgroup = parser.add_argument_group(title='channel')
    chgroup.add_argument(
        '-c', '--channel', dest='chs', action='append',
        help='''Channel names to use in data directory. (default: "ch0")''',
    )
    chgroup.add_argument(
        '-f', '--centerfreq', dest='centerfreqs', action='append',
        help='''Center frequency in Hz. (default: 100e6)''',
    )
    chgroup.add_argument(
        '-F', '--lo_offset', dest='lo_offsets', action='append',
        help='''Frontend tuner offset from center frequency, in Hz.
                (default: 0)''',
    )
    chgroup.add_argument(
        '--lo_source', dest='lo_sources', action='append',
        help='''Local oscillator source. Typically 'None'/'' (do not set),
                'internal' (e.g. LO1 for CH1, LO2 for CH2),
                'companion' (e.g. LO2 for CH1, LO1 for CH2), or
                'external' (neighboring board via connector).
                (default: '')''',
    )
    chgroup.add_argument(
        '--lo_export', dest='lo_exports', action='append',
        help='''Whether to export the LO's source to the external connector.
                Can be 'None'/'' to skip the channel, otherwise it can be
                'True' or 'False' provided the LO source is set.
                (default: None)''',
    )
    chgroup.add_argument(
        '-g', '--gain', dest='gains', action='append',
        help='''Gain in dB. (default: 0)''',
    )
    chgroup.add_argument(
        '-b', '--bandwidth', dest='bandwidths', action='append',
        help='''Frontend bandwidth in Hz. (default: 0 == frontend default)''',
    )
    chgroup.add_argument(
        '-y', '--antenna', dest='antennas', action='append',
        help='''Name of antenna to select on the frontend.
                (default: frontend default))''',
    )

    recgroup = parser.add_argument_group(title='receiver')
    recgroup.add_argument(
        '-r', '--samplerate', dest='samplerate',
        default='1e6',
        help='''Sample rate in Hz. (default: %(default)s)''',
    )
    recgroup.add_argument(
        '-i', '--dec', dest='dec',
        default=1, type=int,
        help='''Integrate and decimate by this factor.
                (default: %(default)s)''',
    )
    recgroup.add_argument(
        '-A', '--devargs', dest='dev_args', action='append',
        default=['recv_buff_size=100000000', 'num_recv_frames=512'],
        help='''Device arguments, e.g. "master_clock_rate=30e6".
                (default: %(default)s)''',
    )
    recgroup.add_argument(
        '-a', '--streamargs', dest='stream_args', action='append',
        help='''Stream arguments, e.g. "peak=0.125,fullscale=1.0".
                (default: %(default)s)''',
    )
    recgroup.add_argument(
        '--sync_source', dest='sync_source', default='external',
        help='''Clock and time source for all mainboards.
                (default: %(default)s)''',
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

    timegroup = parser.add_argument_group(title='time')
    timegroup.add_argument(
        '-s', '--starttime', dest='starttime',
        help='''Start time of the experiment as sample index (if int), Unix
                time (if float), or datetime (if in ISO8601 format:
                2016-01-01T15:24:00Z) (default: %(default)s)''',
    )
    timegroup.add_argument(
        '-e', '--endtime', dest='endtime',
        help='''End time of the experiment as sample index (if int), Unix
                time (if float), or datetime (if in ISO8601 format:
                2016-01-01T16:24:00Z (default: %(default)s)''',
    )
    timegroup.add_argument(
        '-l', '--duration', dest='duration',
        default=None,
        help='''Duration of experiment in seconds. When endtime is not given,
                end this long after start time. (default: %(default)s)''',
    )
    timegroup.add_argument(
        '-p', '--cycle-length', dest='period',
        default=10, type=int,
        help='''Repeat time of experiment cycle. Align to start of next cycle
                if start time has passed. (default: %(default)s)''',
    )

    drfgroup = parser.add_argument_group(title='digital_rf')
    drfgroup.add_argument(
        '-n', '--file_cadence_ms', dest='file_cadence_ms',
        default=1000, type=int,
        help='''Number of milliseconds of data per file.
                (default: %(default)s)''',
    )
    drfgroup.add_argument(
        '-N', '--subdir_cadence_s', dest='subdir_cadence_s',
        default=3600, type=int,
        help='''Number of seconds of data per subdirectory.
                (default: %(default)s)''',
    )
    drfgroup.add_argument(
        '--metadata', action='append', metavar='{KEY}={VALUE}',
        help='''Key, value metadata pairs to include with data.
                (default: "")''',
    )
    drfgroup.add_argument(
        '--uuid', dest='uuid',
        default=None,
        help='''Unique ID string for this data collection.
                (default: random)''',
    )

    op = parser.parse_args()

    if op.datadir is None:
        op.datadir = op.outdir
    del op.outdir

    if op.mboards is None:
        op.mboards = []
    if op.subdevs is None:
        op.subdevs = ['A:A']
    if op.chs is None:
        op.chs = ['ch0']
    if op.centerfreqs is None:
        op.centerfreqs = ['100e6']
    if op.lo_offsets is None:
        op.lo_offsets = ['0']
    if op.lo_sources is None:
        op.lo_sources = ['']
    if op.lo_exports is None:
        op.lo_exports = ['None']
    if op.gains is None:
        op.gains = ['0']
    if op.bandwidths is None:
        # use 0 bandwidth as special case to set frontend default
        op.bandwidths = ['0']
    if op.antennas is None:
        op.antennas = ['']
    if op.dev_args is None:
        op.dev_args = []
    if op.stream_args is None:
        op.stream_args = []
    if op.metadata is None:
        op.metadata = []

    # separate any combined arguments
    # e.g. op.mboards = ['192.168.10.2,192.168.10.3']
    #      becomes ['192.168.10.2', '192.168.10.3']
    op.mboards = [b.strip() for a in op.mboards for b in a.strip().split(',')]
    op.subdevs = [b.strip() for a in op.subdevs for b in a.strip().split(',')]
    op.chs = [b.strip() for a in op.chs for b in a.strip().split(',')]
    op.centerfreqs = [
        float(b.strip()) for a in op.centerfreqs for b in a.strip().split(',')
    ]
    op.lo_offsets = [
        float(b.strip()) for a in op.lo_offsets for b in a.strip().split(',')
    ]
    op.lo_sources = [
        b.strip() if b.strip().lower() not in ('', 'none') else None
        for a in op.lo_sources for b in a.strip().split(',')
    ]
    op.lo_exports = [
        b.strip().lower() in ('true', 't', 'yes', 'y', '1')
        if b.strip().lower() not in ('', 'none') else None
        for a in op.lo_exports for b in a.strip().split(',')
    ]
    op.gains = [
        float(b.strip()) for a in op.gains for b in a.strip().split(',')
    ]
    op.bandwidths = [
        float(b.strip()) for a in op.bandwidths for b in a.strip().split(',')
    ]
    op.antennas = [
        b.strip() for a in op.antennas for b in a.strip().split(',')
    ]
    op.dev_args = [
        b.strip() for a in op.dev_args for b in a.strip().split(',')
    ]
    op.stream_args = [
        b.strip() for a in op.stream_args for b in a.strip().split(',')
    ]
    op.metadata = [
        b.strip() for a in op.metadata for b in a.strip().split(',')
    ]

    # remove redundant arguments in dev_args and stream_args
    try:
        dev_args_dict = dict([a.split('=') for a in op.dev_args])
        stream_args_dict = dict([a.split('=') for a in op.stream_args])
    except ValueError:
        raise ValueError(
            'Device and stream arguments must be {KEY}={VALUE} pairs.'
        )
    op.dev_args = [
        '{0}={1}'.format(k, v) for k, v in dev_args_dict.iteritems()
    ]
    op.stream_args = [
        '{0}={1}'.format(k, v) for k, v in stream_args_dict.iteritems()
    ]

    # evaluate samplerate to float
    op.samplerate = float(eval(op.samplerate))
    # evaluate duration to int
    if op.duration is not None:
        op.duration = int(eval(op.duration))

    # convert metadata strings to a dictionary
    metadata_dict = {}
    for a in op.metadata:
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
    op.metadata = metadata_dict

    options = dict(op._get_kwargs())
    starttime = options.pop('starttime')
    endtime = options.pop('endtime')
    duration = options.pop('duration')
    period = options.pop('period')
    thor = Thor(**options)
    thor.run(starttime, endtime, duration, period)
