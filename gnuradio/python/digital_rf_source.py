# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module defining a Digital RF Source block."""
import ast
import datetime
import os

import dateutil.parser
import gnuradio.blocks
import h5py
import pmt
import numpy as np
import pytz
import six
from gnuradio import gr

from digital_rf import DigitalRFReader

H5T_LOOKUP = {
    # (class, itemsize, is_complex): {name, dtype}
    (h5py.h5t.INTEGER, 1, False): dict(name='s8', dtype=np.int8),
    (h5py.h5t.INTEGER, 2, False): dict(name='s16', dtype=np.int16),
    (h5py.h5t.INTEGER, 4, False): dict(name='s32', dtype=np.int32),
    (h5py.h5t.INTEGER, 8, False): dict(name='s64', dtype=np.int64),
    (h5py.h5t.FLOAT, 4, False): dict(name='f32', dtype=np.float32),
    (h5py.h5t.FLOAT, 8, False): dict(name='f64', dtype=np.float64),
    (h5py.h5t.INTEGER, 1, True): dict(
        name='sc8', dtype=np.dtype([('r', np.int8), ('i', np.int8)]),
    ),
    (h5py.h5t.INTEGER, 2, True): dict(
        name='sc16', dtype=np.dtype([('r', np.int16), ('i', np.int16)]),
    ),
    (h5py.h5t.INTEGER, 4, True): dict(
        name='sc32', dtype=np.dtype([('r', np.int32), ('i', np.int32)]),
    ),
    (h5py.h5t.INTEGER, 8, True): dict(
        name='sc64', dtype=np.dtype([('r', np.int64), ('i', np.int64)]),
    ),
    (h5py.h5t.FLOAT, 4, True): dict(name='fc32', dtype=np.complex64),
    (h5py.h5t.FLOAT, 8, True): dict(name='fc64', dtype=np.complex128),
}


def get_h5type(cls, size, is_complex):
    try:
        typedict = H5T_LOOKUP[(cls, size, is_complex)]
    except KeyError:
        raise ValueError('HDF5 data type not supported for reading.')
    return typedict


class digital_rf_channel_source(gr.sync_block):
    """
    docstring for block digital_rf_channel_source
    """
    def __init__(
        self, channel_dir, start=None, end=None, repeat=False,
    ):
        """Initialize source to directory containing Digital RF channels.

        Parameters
        ----------

        channel_dir : string
            Either a single channel directory containing 'drf_properties.h5'
            and timestamped subdirectories with Digital RF files, or a list of
            such. A directory can be a file system path or a url, where the url
            points to a channel directory. Each must be a local path, or start
            with 'http://'', 'file://'', or 'ftp://''.


        Other Parameters
        ----------------

        start : None | int/long | float | string
            A value giving the start of the channel's playback.
            If None or '', the start of the channel's available data is used.
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a timestamp (seconds since epoch).
            If a string, three forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a string beginning with '+' and followed by an integer
                    (float) expression, interpreted as samples (seconds) from
                    the start of the data, and
                3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

        end : None | int/long | float | string
            A value giving the end of the channel's playback.
            If None or '', the end of the channel's available data is used.
            See `start` for a description of how this value is interpreted.

        repeat : bool
            If True, loop the data continuously from the start after the end
            is reached. If False, stop after the data is read once.


        Notes
        -----

        A channel directory must contain subdirectories/files in the format:
            <YYYY-MM-DDTHH-MM-SS/rf@<seconds>.<%03i milliseconds>.h5

        Each directory provided is considered the same channel. An error is
        raised if their sample rates differ, or if their time periods overlap.

        """
        if isinstance(channel_dir, six.string_types):
            channel_dir = [channel_dir]
        # eventually, we should re-factor DigitalRFReader and associated so
        # that reading from a list of channel directories is possible
        # with a DigitalRFChannelReader class or similar
        # until then, split the path and use existing DigitalRFReader
        top_level_dirs = []
        chs = set()
        for ch_dir in channel_dir:
            top_level_dir, ch = os.path.split(ch_dir)
            top_level_dirs.append(top_level_dir)
            chs.add(ch)
        if len(chs) == 1:
            ch = chs.pop()
        else:
            raise ValueError('Channel directories must have the same name.')
        self._ch = ch

        self._Reader = DigitalRFReader(top_level_dirs)
        self._properties = self._Reader.get_properties(self._ch)

        typeclass = self._properties['H5Tget_class']
        itemsize = self._properties['H5Tget_size']
        is_complex = self._properties['is_complex']
        vlen = self._properties['num_subchannels']
        sr = self._properties['samples_per_second']

        self._itemsize = itemsize
        self._sample_rate = sr
        self._sample_rate_pmt = pmt.from_double(float(sr))

        # determine output signature from HDF5 type metadata
        typedict = get_h5type(typeclass, itemsize, is_complex)
        self._outtype = typedict['name']
        self._itemtype = typedict['dtype']
        if vlen == 1:
            out_sig = [self._itemtype]
        else:
            out_sig = [(self._itemtype, vlen)]

        gr.sync_block.__init__(
            self,
            name="digital_rf_channel_source",
            in_sig=None,
            out_sig=out_sig,
        )

        self.message_port_register_out(pmt.intern('metadata'))
        self._id = pmt.intern(self._ch)
        self._tag_queue = {}

        self._start = start
        self._end = end
        self._repeat = repeat

        try:
            self._DMDReader = self._Reader.get_digital_metadata(self._ch)
        except IOError:
            self._DMDReader = None

        # FIXME: should not be necessary, sets a large output buffer so that
        # we don't underrun on frequent calls to work
        self.set_output_multiple(int(sr))

    @staticmethod
    def _parse_sample_identifier(iden, sample_rate=None, ref_index=None):
        """Get a sample index from different forms of identifiers.

        Parameters
        ----------

        iden : None | int/long | float | string
            If None or '', None is returned to indicate that the index should
            be automatically determined.
            If an integer, it is returned as the sample index.
            If a float, it is interpreted as a timestamp (seconds since epoch)
            and the corresponding sample index is returned.
            If a string, three forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a string beginning with '+' and followed by an integer
                    (float) expression, interpreted as samples (seconds) from
                    `ref_index`, and
                3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

        sample_rate : numpy.longdouble, required for float and time `iden`
            Sample rate in Hz used to convert a time to a sample index.

        ref_index : int/long, required for '+' string form of `iden`
            Reference index from which string `iden` beginning with '+' are
            offset.


        Returns
        -------

        sample_index : long | None
            Index to the identified sample given in the number of samples since
            the epoch (time_since_epoch*sample_rate).

        """
        is_relative = False
        if iden is None or iden == '':
            return None
        elif isinstance(iden, six.string_types):
            if iden.startswith('+'):
                is_relative = True
                iden = iden.lstrip('+')
            try:
                # int/long or float
                iden = ast.literal_eval(iden)
            except (ValueError, SyntaxError):
                # convert datetime to float
                dt = dateutil.parser.parse(iden)
                epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
                iden = (dt - epoch).total_seconds()

        if isinstance(iden, float):
            if sample_rate is None:
                raise ValueError(
                    'sample_rate required when time identifier is used.'
                )
            idx = long(np.uint64(iden*sample_rate))
        else:
            idx = long(iden)

        if is_relative:
            if ref_index is None:
                raise ValueError(
                    'ref_index required when relative "+" identifier is used.'
                )
            return idx + ref_index
        else:
            return idx

    def _queue_tags(self, sample, tags):
        """Queue stream tags to be attached to data in the work function.

        In addition to the tags specified in the `tags` dictionary, this will
        add `rx_time` and `rx_rate` tags giving the sample time and rate.


        Parameters
        ----------

        sample : int | long
            Sample index for the sample to tag, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        tags : dict
            Dictionary containing the tags to add with keys specifying the tag
            name. The value is cast as an appropriate pmt type, while the name
            will be turned into a pmt string in the work function.

        """
        # add to current queued tags for sample if applicable
        tag_dict = self._tag_queue.get(sample, {})
        if not tag_dict:
            # add time and rate tags
            time = sample/self._sample_rate
            tag_dict['rx_time'] = pmt.make_tuple(
                pmt.from_uint64(long(np.uint64(time))),
                pmt.from_double(float(time % 1)),
            )
            tag_dict['rx_rate'] = self._sample_rate_pmt
        for k, v in tags.items():
            tag_dict[k] = pmt.to_pmt(v)
        self._tag_queue[sample] = tag_dict

    def start(self):
        self._bounds = self._Reader.get_bounds(self._ch)
        self._start_sample = self._parse_sample_identifier(
            self._start, self._sample_rate, self._bounds[0],
        )
        self._end_sample = self._parse_sample_identifier(
            self._end, self._sample_rate, self._bounds[0],
        )
        if self._start_sample is None:
            self._global_index = self._bounds[0]
        else:
            self._global_index = self._start_sample
        # add default tags to first sample
        self._queue_tags(self._global_index, {})
        # replace longdouble samples_per_second with float for pmt conversion
        message_metadata = self._properties.copy()
        message_metadata['samples_per_second'] = \
            float(message_metadata['samples_per_second'])
        self.message_port_pub(
            pmt.intern('metadata'), pmt.to_pmt({self._ch: message_metadata}),
        )
        return super(digital_rf_channel_source, self).start()

    def work(self, input_items, output_items):
        out = output_items[0]
        nsamples = len(out)
        nout = 0
        # repeat reading until we succeed or return
        while nout < nsamples:
            start_sample = self._global_index
            # end_sample is inclusive, hence the -1
            end_sample = self._global_index + (nsamples - nout) - 1
            # creating a read function that has an output argument so data can
            # be copied directly would be nice
            # also should move EOFError checking into reader once watchdog
            # bounds functionality is implemented
            try:
                if self._end_sample is None:
                    if end_sample > self._bounds[1]:
                        self._bounds = self._Reader.get_bounds(self._ch)
                        end_sample = min(end_sample, self._bounds[1])
                else:
                    if end_sample > self._end_sample:
                        end_sample = self._end_sample
                if start_sample > end_sample:
                    raise EOFError
                data_dict = self._Reader.read(
                    start_sample, end_sample, self._ch,
                )
                for sample, data in data_dict.items():
                    # index into out starts at number of previously read
                    # samples plus the offset from what we requested
                    ks = nout + (sample - start_sample)
                    ke = ks + data.shape[0]
                    # out is zeroed, so only have to write samples we have
                    out[ks:ke] = data.squeeze()
                # now read corresponding metadata
                if self._DMDReader is not None:
                    meta_dict = self._DMDReader.read(
                        start_sample, end_sample,
                    )
                    for sample, meta in meta_dict.items():
                        # add center frequency tag from metadata
                        # (in addition to default time and rate tags)
                        tags = dict(
                            rx_freq=meta['center_frequencies'].ravel()[0]
                        )
                        self._queue_tags(sample, tags)
                # add queued tags to stream
                for sample, tag_dict in self._tag_queue.items():
                    offset = (
                        self.nitems_written(0) + nout +
                        (sample - start_sample)
                    )
                    for name, val in tag_dict.items():
                        self.add_item_tag(
                            0, offset, pmt.intern(name), val, self._id,
                        )
                self._tag_queue.clear()
                # no errors, so we read all the samples we wanted
                # (end_sample is inclusive, hence the +1)
                nread = (end_sample + 1 - start_sample)
                nout += nread
                self._global_index += nread
            except EOFError:
                if self._repeat:
                    if self._start_sample is None:
                        self._global_index = self._bounds[0]
                    else:
                        self._global_index = self._start_sample
                    self._queue_tags(self._global_index, {})
                    continue
                else:
                    break
        if nout == 0:
            # return WORK_DONE
            return -1
        return nout


class digital_rf_source(gr.hier_block2):
    """
    docstring for block digital_rf_source
    """
    def __init__(
        self, top_level_dir, channels=None, start=None, end=None,
        repeat=False, throttle=False,
    ):
        """Initialize source to directory containing Digital RF channels.

        Parameters
        ----------

        top_level_dir : string
            Either a single top level directory containing Digital RF channel
            directories, or a list of such. A directory can be a file system
            path or a url, where the url points to a top level directory. Each
            must be a local path, or start with 'http://'', 'file://'', or
            'ftp://''.


        Other Parameters
        ----------------

        channels : None | string | int | iterable of previous
            If None, use all available channels in alphabetical order.
            Otherwise, use the channels in the order specified in the given
            iterable (a string or int is taken as a single-element iterable).
            A string is used to specify the channel name, while an int is used
            to specify the channel index in the sorted list of available
            channel names.

        start : None | string | long | iterable of previous
            Can be a single value or an iterable of values corresponding to
            `channels` giving the start of the channel's playback.
            If None or '', the start of the channel's available data is used.
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a timestamp (seconds since epoch).
            If a string, three forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a string beginning with '+' and followed by an integer
                    (float) expression, interpreted as samples (seconds) from
                    the start of the data, and
                3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

        end : None | string | long | iterable of previous
            Can be a single value or an iterable of values corresponding to
            `channels` giving the end of the channel's playback.
            If None or '', the end of the channel's available data is used.
            See `start` for a description of how this value is interpreted.

        repeat : bool
            If True, loop the data continuously from the start after the end
            is reached. If False, stop after the data is read once.

        throttle : bool
            If True, playback the samples at their recorded sample rate. If
            False, read samples as quickly as possible.

        Notes
        -----

        A top level directory must contain files in the format:
            <channel>/<YYYY-MM-DDTHH-MM-SS/rf@<seconds>.<%03i milliseconds>.h5

        If more than one top level directory contains the same channel_name
        subdirectory, this is considered the same channel. An error is raised
        if their sample rates differ, or if their time periods overlap.

        """
        Reader = DigitalRFReader(top_level_dir)
        available_channel_names = Reader.get_channels()
        self._channel_names = self._get_channel_names(
            channels, available_channel_names,
        )

        if start is None:
            start = [None]*len(self._channel_names)
        try:
            s_iter = iter(start)
        except TypeError:
            s_iter = iter([start])
        if end is None:
            end = [None]*len(self._channel_names)
        try:
            e_iter = iter(end)
        except TypeError:
            e_iter = iter([end])

        # make sources for each channel
        self._channels = []
        for ch, s, e in zip(self._channel_names, s_iter, e_iter):
            chsrc = digital_rf_channel_source(
                os.path.join(top_level_dir, ch), start=s, end=e, repeat=repeat,
            )
            self._channels.append(chsrc)

        out_sig_dtypes = [src.out_sig()[0] for src in self._channels]
        out_sig = gr.io_signaturev(
            len(out_sig_dtypes), len(out_sig_dtypes),
            [s.itemsize for s in out_sig_dtypes],
        )
        in_sig = gr.io_signature(0, 0, 0)

        gr.hier_block2.__init__(
            self,
            name="digital_rf_source",
            input_signature=in_sig,
            output_signature=out_sig,
        )

        msg_port_name = pmt.intern('metadata')
        self.message_port_register_hier_out('metadata')

        for k, src in enumerate(self._channels):
            if throttle:
                throt = gnuradio.blocks.throttle(
                    src.out_sig()[0].itemsize, src._sample_rate,
                    ignore_tags=True,
                )
                self.connect(src, throt, (self, k))
            else:
                self.connect(src, (self, k))
            self.msg_connect(src, msg_port_name, self, msg_port_name)

    @staticmethod
    def _get_channel_names(channels, available_channel_names):
        # channels can be None, in which case we use all available
        if channels is None:
            return available_channel_names
        # or channels can be a string for a single channel
        if isinstance(channels, six.string_types):
            channels = [channels]
        unselected_channels = available_channel_names[:]
        channel_names = []
        # now channels should be an iterable of strings or indexes
        try:
            ch_iter = iter(channels)
        except TypeError:
            # unless channels is potentially a single index
            ch_iter = iter([channels])
        for ch in ch_iter:
            # make None and index ch into string channel name
            if ch is None or ch == '':
                # use first available channel (alphabetical)
                try:
                    ch_name = unselected_channels[0]
                except IndexError:
                    raise ValueError(
                        '"None" invalid for channel, all available '
                        'channels have been selected.'
                    )
            else:
                # try ch as a list index into available channels
                try:
                    ch_name = available_channel_names[int(ch)]
                except (TypeError, ValueError):
                    # not an index, that's fine
                    ch_name = ch
                except IndexError:
                    raise IndexError(
                        'Channel index {0} does not exist.'.format(ch)
                    )

            # now assume ch is a string, get from unselected channel list
            try:
                unselected_channels.remove(ch_name)
            except ValueError:
                errstr = (
                    'Channel {0} does not exist or has already been '
                    'selected.'
                )
                raise ValueError(errstr.format(ch_name))
            channel_names.append(ch_name)
        return channel_names
