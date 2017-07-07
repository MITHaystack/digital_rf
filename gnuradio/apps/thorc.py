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

import datetime
import math
import os
import re
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from ast import literal_eval
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import TextWrapper, dedent, fill

import dateutil.parser
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

        if op.uuid is None:
            # generate random UUID
            op.uuid = uuid.uuid4().hex

        op.nmboards = len(op.mboards) if len(op.mboards) > 0 else 1
        op.nchs = len(op.chs)
        # repeat arguments as necessary
        op.subdevs = list(islice(cycle(op.subdevs), 0, op.nmboards))
        op.centerfreqs = list(islice(cycle(op.centerfreqs), 0, op.nchs))
        op.lo_offsets = list(islice(cycle(op.lo_offsets), 0, op.nchs))
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
            elif re.match(r'[0-9]{1,}', mb):
                idtype = 'serial'
            elif (
                re.match(r'usrp[123]', mb) or re.match(r'b2[01]0', mb)
                or re.match(r'x3[01]0', mb)
            ):
                idtype = 'type'
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
                Frequency offset: {lo_offsets}
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
                errstr = 'Unknown sync_source option: {0}. Must be one of {1}.'
                errstr = errstr.format(op.sync_source, u.get_clock_sources(0))
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
        for ch_num in range(op.nchs):
            # center frequency and tuning offset
            tune_res = u.set_center_freq(
                uhd.tune_request(
                    op.centerfreqs[ch_num], op.lo_offsets[ch_num],
                ),
                ch_num,
            )
            # read back actual values
            op.centerfreqs[ch_num] = u.get_center_freq(chan=ch_num)
            op.lo_offsets[ch_num] = tune_res.actual_dsp_freq
            # gain
            u.set_gain(op.gains[ch_num], ch_num)
            # read back actual value
            op.gains[ch_num] = u.get_gain(ch_num)
            # bandwidth
            bw = op.bandwidths[ch_num]
            if bw:
                u.set_bandwidth(bw, ch_num)
            # read back actual value
            op.bandwidths[ch_num] = u.get_bandwidth(chan=ch_num)
            # antenna
            ant = op.antennas[ch_num]
            if ant != '':
                try:
                    u.set_antenna(ant, ch_num)
                except RuntimeError:
                    errstr = 'Unknown RX antenna option: {0}.'
                    errstr += ' Must be one of {1}.'
                    errstr = errstr.format(ant, u.get_antennas(ch_num))
                    raise ValueError(errstr)
            # read back actual value
            op.antennas[ch_num] = u.get_antenna(chan=ch_num)

        if op.verbose:
            print('Using the following devices:')
            chinfo = '  Motherboard: {mb_id} ({mb_addr})\n'
            chinfo += '  Daughterboard: {db_subdev}\n'
            chinfo += '  Subdev: {subdev}\n'
            chinfo += '  Antenna: {ant}'
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
                info['db_subdev'] = usrpinfo['rx_subdev_name']
                info['subdev'] = op.subdevs_bychan[ch_num]
                info['ant'] = op.antennas[ch_num]
                print(chinfo.format(**info))
                print('-' * 78)

        return u

    def run(self, starttime=None, endtime=None, duration=None, period=10):
        op = self.op

        # print current time and NTP status
        if op.verbose:
            call(('timedatectl', 'status'))

        # parse time arguments
        if starttime is None:
            st = None
        else:
            dtst = dateutil.parser.parse(starttime)
            epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            st = int((dtst - epoch).total_seconds())

            # find next suitable start time by cycle repeat period
            soon = int(math.ceil(time.time())) + 5
            periods_until_next = (max(soon - st, 0) - 1) // period + 1
            st = st + periods_until_next * period

            if op.verbose:
                dtst = datetime.datetime.utcfromtimestamp(st)
                dtststr = dtst.strftime('%a %b %d %H:%M:%S %Y')
                print('Start time: {0} ({1})'.format(dtststr, st))

        if endtime is None:
            et = None
        else:
            dtet = dateutil.parser.parse(endtime)
            epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            et = int((dtet - epoch).total_seconds())

            if op.verbose:
                dtetstr = dtet.strftime('%a %b %d %H:%M:%S %Y')
                print('End time: {0} ({1})'.format(dtetstr, et))

        if et is not None:
            if (et < time.time() + 5) or (st is not None and et <= st):
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
        while (st is not None) and (st - time.time()) > 10:
            ttl = int(math.floor(st - time.time()))
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
            sample_size = gr.sizeof_gr_complex
            sample_dtype = '<f4'

            taps = firdes.low_pass_2(
                1.0, float(op.samplerate), float(samplerate_out / 2.0),
                float(0.2 * samplerate_out), 80.0,
                window=firdes.WIN_BLACKMAN_hARRIS
            )
        else:
            sample_size = 2 * gr.sizeof_short
            sample_dtype = '<i2'

        # set launch time
        # (at least 1 second out so USRP time is set, time to set up flowgraph)
        if st is not None:
            lt = st
        else:
            lt = int(math.ceil(time.time() + 1.0))
        # adjust launch time forward so it falls on an exact sample since epoch
        lt_samples = np.ceil(lt * samplerate_out)
        lt = lt_samples / samplerate_out
        if op.verbose:
            dtlt = datetime.datetime.utcfromtimestamp(lt)
            dtltstr = dtlt.strftime('%a %b %d %H:%M:%S.%f %Y')
            print('Launch time: {0} ({1})'.format(dtltstr, repr(lt)))
        # command time
        ct_samples = lt_samples
        # splitting ct into secs/frac lets us set a more accurate time_spec
        ct_secs = ct_samples // samplerate_out
        ct_frac = (ct_samples % samplerate_out) / samplerate_out
        u.set_start_time(
            uhd.time_spec(float(ct_secs)) + uhd.time_spec(float(ct_frac))
        )

        # populate flowgraph one channel at a time
        fg = gr.top_block()
        for k in range(op.nchs):
            # create digital RF sink
            chdir = os.path.join(op.datadir, op.chs[k])
            dst = gr_drf.digital_rf_sink_c(
                dir=chdir, sample_size=sample_size,
                subdir_cadence_s=op.subdir_cadence_s,
                file_cadence_ms=op.file_cadence_ms,
                sample_rate_numerator=samplerate_num_out,
                sample_rate_denominator=samplerate_den_out,
                uuid=op.uuid, is_complex=True, num_subchannels=1,
                stop_on_dropped_packet=op.stop_on_dropped,
                start_sample_index=int(lt * samplerate_out),
                ignore_tags=False,
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

        # write metadata one channel at a time
        for k in range(op.nchs):
            mbnum = op.mboardnum_bychan[k]
            # create metadata dir, dmd object, and write channel metadata
            mddir = os.path.join(op.datadir, op.chs[k], 'metadata')
            if not os.path.exists(mddir):
                os.makedirs(mddir)
            mdo = drf.DigitalMetadataWriter(
                metadata_dir=mddir,
                subdir_cadence_secs=op.subdir_cadence_s,
                file_cadence_secs=1,
                sample_rate_numerator=samplerate_num_out,
                sample_rate_denominator=samplerate_den_out,
                file_name='metadata',
            )
            md = op.metadata.copy()
            md.update(
                # standard metadata by convention
                uuid_str=op.uuid,
                sample_rate_numerator=samplerate_num_out,
                sample_rate_denominator=samplerate_den_out,
                # put in a list because we want the data to be a 1-D array
                # and it would be a single value if we didn't
                center_frequencies=[np.array([op.centerfreqs[k]])],

                # additional receiver metadata for USRP
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
            )
            mdo.write(
                samples=int(lt * samplerate_out),
                data=md,
            )

        # wait until end time or until flowgraph stops
        if et is None and duration is not None:
            et = lt + duration
        try:
            if et is None:
                fg.wait()
            else:
                # sleep until end time nears
                while(time.time() < et - 1):
                    time.sleep(1)
                else:
                    # issue stream stop command at end time
                    ct_secs = et // 1
                    ct_frac = et % 1
                    u.set_command_time(
                        (uhd.time_spec(float(ct_secs)) +
                            uhd.time_spec(float(ct_frac))),
                        uhd.ALL_MBOARDS,
                    )
                    stop_enum = uhd.stream_cmd.STREAM_MODE_STOP_CONTINUOUS
                    u.issue_stream_cmd(uhd.stream_cmd(stop_enum))
                    u.clear_command_time(uhd.ALL_MBOARDS)
                    # sleep until after end time
                    time.sleep(1)
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        fg.stop()
        print('done')
        sys.stdout.flush()


if __name__ == '__main__':
    scriptname = os.path.basename(sys.argv[0])

    formatter = RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    desc = 'Record data from synchronized USRPs in DigitalRF format.'

    usage = '%(prog)s [options] [-o DIR | DIR]'

    epi_pars = [
        '''\
        Multiple mainboards can be specified by repeating the mainboard and
        channel arguments. The subdevice, centerfreq, and gain arguments can
        also be repeated to set those properties per mainboard, but those
        values will be duplicated if necessary to match the number of specified
        mainboards.
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
        {0} -m 192.168.10.2 -d "A:0" -y "TX/RX" -c ch1 -f 20e6 -F 10e3 -g 20
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
        version=(
            'THOR version 3.0 (The Haystack Observatory Recorder)\n'
            '\n'
            'Copyright (c) 2017 Massachusetts Institute of Technology'
        )
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
        help='''Start time of the experiment in ISO8601 format:
                2016-01-01T15:24:00Z (default: %(default)s)''',
    )
    timegroup.add_argument(
        '-e', '--endtime', dest='endtime',
        help='''End time of the experiment in ISO8601 format:
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
