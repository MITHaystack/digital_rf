#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
import datetime
import math
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from itertools import chain
from textwrap import dedent

import dateutil.parser
import numpy as np
import pytz
from gnuradio import blocks, gr, uhd


class Tx(object):
    """Transmit data in binary format from a single USRP."""

    def __init__(
        self, codefile, mainboard=None, subdevice=None,
        centerfreq=3.6e6, lo_offset=0,
        gain=0, amplitude=0.25, bandwidth=0, antenna=None,
        samplerate=1e6,
        dev_args=['send_buff_size=1000000'],
        stream_args=[],
        sync=True, sync_source='external',
        realtime=False, verbose=True, test_settings=True,
    ):
        options = locals()
        del options['self']
        op = self._parse_options(**options)
        self.op = op

        if op.test_settings:
            # try reading code from file
            code_vector = np.fromfile(op.codefile, dtype=np.complex64)
            # test usrp device settings, release device when done
            u = self._usrp_setup()
            if op.verbose:
                print('Using the following device:')
                chinfo = '  Motherboard: {mb_id} ({mb_addr})\n'
                chinfo += '  Daughterboard: {db_subdev}\n'
                chinfo += '  Subdev: {subdev}\n'
                chinfo += '  Antenna: {ant}'
                print('-' * 78)
                usrpinfo = dict(u.get_usrp_info(chan=0))
                info = {}
                info['mb_id'] = usrpinfo['mboard_id']
                mba = op.mainboard
                if mba is None:
                    mba = usrpinfo['mboard_serial']
                info['mb_addr'] = mba
                info['db_subdev'] = usrpinfo['tx_subdev_name']
                info['subdev'] = usrpinfo['tx_subdev_spec']
                info['ant'] = usrpinfo['tx_antenna']
                print(chinfo.format(**info))
                print('-' * 78)
            del u

    @staticmethod
    def _parse_options(**kwargs):
        """Put all keyword options in a namespace and normalize them."""
        op = Namespace(**kwargs)

        op.mboard_strs = []
        if op.mainboard is not None:
            mb = op.mainboard
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
            s = '{type}={mb}'.format(type=idtype, mb=mb.strip())
            op.mboard_strs.append(s)

        if op.verbose:
            opstr = dedent('''\
                Main board: {mboard_strs}
                Subdevice: {subdevice}
                Frequency: {centerfreq}
                Frequency offset: {lo_offset}
                Gain: {gain}
                Amplitude: {amplitude}
                Bandwidth: {bandwidth}
                Antenna: {antenna}
                Device arguments: {dev_args}
                Stream arguments: {stream_args}
                Sample rate: {samplerate}
                Code file: {codefile}
            ''').strip().format(**op.__dict__)
            print(opstr)

        return op

    def _usrp_setup(self):
        """Create, set up, and return USRP sink object."""
        op = self.op
        # create usrp sink block
        u = uhd.usrp_sink(
            device_addr=','.join(chain(op.mboard_strs, op.dev_args)),
            stream_args=uhd.stream_args(
                cpu_format='fc32',
                otw_format='sc16',
                channels=range(1),
                args=','.join(op.stream_args)
            )
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
        if op.subdevice is not None:
            u.set_subdev_spec(op.subdevice, 0)
        # set global options
        u.set_samp_rate(float(op.samplerate))
        samplerate = u.get_samp_rate()  # may be different than desired
        # calculate longdouble precision sample rate
        # (integer division of clock rate)
        cr = u.get_clock_rate()
        srdec = int(round(cr / samplerate))
        samplerate_ld = np.longdouble(cr) / srdec
        op.samplerate = samplerate_ld
        # set per-channel options
        u.set_center_freq(uhd.tune_request(op.centerfreq, op.lo_offset), 0)
        u.set_gain(op.gain, 0)
        if op.bandwidth:
            u.set_bandwidth(op.bandwidth, 0)
        if op.antenna:
            try:
                u.set_antenna(op.antenna, 0)
            except RuntimeError:
                errstr = 'Unknown RX antenna option: {0}.'
                errstr += ' Must be one of {1}.'
                errstr = errstr.format(op.antenna, u.get_antennas(0))
                raise ValueError(errstr)
        return u

    def run(self, starttime=None, endtime=None, duration=None, period=10):
        op = self.op

        # print current time and NTP status
        if op.verbose:
            subprocess.call(('timedatectl', 'status'))

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

        # wait for the start time if it is not past
        while (st is not None) and (st - time.time()) > 10:
            ttl = int(math.floor(st - time.time()))
            if (ttl % 10) == 0:
                print('Standby {0} s remaining...'.format(ttl))
                sys.stdout.flush()
            time.sleep(1)

        # get UHD USRP source
        u = self._usrp_setup()

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

        # populate flowgraph
        fg = gr.top_block()
        code_source = blocks.file_source(
            gr.sizeof_gr_complex, op.codefile, repeat=True,
        )
        multiply = blocks.multiply_const_vcc((op.amplitude,))
        fg.connect(code_source, multiply, u)

        # set launch time
        if st is not None:
            lt = st
        else:
            lt = int(math.ceil(time.time() + 0.5))
        # adjust launch time forward so it falls on an exact sample since epoch
        lt_samples = np.ceil(lt * op.samplerate)
        lt = lt_samples / op.samplerate
        if op.verbose:
            dtlt = datetime.datetime.utcfromtimestamp(lt)
            dtltstr = dtlt.strftime('%a %b %d %H:%M:%S.%f %Y')
            print('Launch time: {0} ({1})'.format(dtltstr, repr(lt)))
        # command time
        ct_samples = lt_samples
        # splitting ct into secs/frac lets us set a more accurate time_spec
        ct_secs = ct_samples // op.samplerate
        ct_frac = (ct_samples % op.samplerate) / op.samplerate
        u.set_start_time(
            uhd.time_spec(float(ct_secs)) + uhd.time_spec(float(ct_frac))
        )

        # start to transmit data
        fg.start()

        # wait until end time or until flowgraph stops
        if et is None and duration is not None:
            et = lt + duration
        try:
            # sleep until start time
            time.sleep(st - time.time())
            # sleep until end time nears
            while(et is None or time.time() < et):
                time.sleep(1)
                sys.stdout.write('.')
                sys.stdout.flush()
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        fg.stop()
        print('done')
        sys.stdout.flush()


if __name__ == '__main__':
    scriptname = os.path.basename(sys.argv[0])
    desc = 'Transmit a waveform from a binary file on a loop using a USRP.'
    usage = '%(prog)s [options] CODE_FILE'
    epilog = '''\
        e.g.: {0} -m 192.168.10.2 -d "A:A" -f 3.6e6 -G 0.25 -g 0 -r 1e6
        code.bin
    '''
    parser = ArgumentParser(
        description=desc, usage=usage, epilog=epilog.format(scriptname),
    )

    parser.add_argument(
        'codefile', help='''Transmit code binary file.''',
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

    txgroup = parser.add_argument_group(title='receiver')
    txgroup.add_argument(
        '-m', '--mainboard', default=None,
        help='''Mainboard address. (default: first device found)''',
    )
    txgroup.add_argument(
        '-d', '--subdevice', default=None,
        help='''USRP subdevice string. (default: mainboard default)''',
    )

    txgroup.add_argument(
        '-f', '--centerfreq', default='3.6e6',
        help='''Center frequency in Hz. (default: %(default)s)''',
    )
    txgroup.add_argument(
        '-F', '--lo_offset', default='0',
        help='''Frontend tuner offset from center frequency, in Hz.
                (default: %(default)s)''',
    )
    txgroup.add_argument(
        '-G', '--amplitude', type=float, default=0.25,
        help='''Waveform amplitude multiplier. (default: %(default)s)''',
    )
    txgroup.add_argument(
        '-g', '--gain', type=float, default=0,
        help='''USRP gain in dB. (default: %(default)s)''',
    )
    txgroup.add_argument(
        '-b', '--bandwidth', default='0',
        help='''Frontend bandwidth in Hz. (default: 0 == frontend default)''',
    )
    txgroup.add_argument(
        '-y', '--antenna', default=None,
        help='''Name of antenna to select on the frontend.
                (default: frontend default))''',
    )

    txgroup.add_argument(
        '-r', '--samplerate', default='1e6',
        help='''Sample rate in Hz. (default: %(default)s)''',
    )

    txgroup.add_argument(
        '-A', '--devargs', dest='dev_args', action='append',
        default=['send_buff_size=1000000'],
        help='''Device arguments, e.g. "send_buff_size=1000000".
                (default: %(default)s)''',
    )
    txgroup.add_argument(
        '-a', '--streamargs', dest='stream_args', action='append',
        default=[],
        help='''Stream arguments, e.g. "fullscale=1.0".
                (default: %(default)s)''',
    )
    txgroup.add_argument(
        '--sync_source', default='external',
        help='''Clock and time source for all mainboards.
                (default: %(default)s)''',
    )
    txgroup.add_argument(
        '--nosync', dest='sync', action='store_false',
        help='''No syncing with external clock. (default: False)''',
    )
    txgroup.add_argument(
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

    op = parser.parse_args()

    op.centerfreq = float(op.centerfreq)
    op.lo_offset = float(op.lo_offset)
    op.bandwidth = float(op.bandwidth)
    op.samplerate = float(eval(op.samplerate))
    if op.duration is not None:
        op.duration = int(eval(op.duration))

    # separate any combined arguments
    # e.g. op.mboards = ['192.168.10.2,192.168.10.3']
    #      becomes ['192.168.10.2', '192.168.10.3']
    op.dev_args = [
        b.strip() for a in op.dev_args for b in a.strip().split(',')
    ]
    op.stream_args = [
        b.strip() for a in op.stream_args for b in a.strip().split(',')
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

    options = dict(op._get_kwargs())
    starttime = options.pop('starttime')
    endtime = options.pop('endtime')
    duration = options.pop('duration')
    period = options.pop('period')

    tx = Tx(**options)
    tx.run(starttime, endtime, duration, period)
