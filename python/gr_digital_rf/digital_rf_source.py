# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module defining a Digital RF Source block."""
from __future__ import absolute_import, division, print_function

import os
import traceback

import gnuradio.blocks
import h5py
import numpy as np
import pmt
import six
from gnuradio import gr

from digital_rf import DigitalRFReader, util

H5T_LOOKUP = {
    # (class, itemsize, is_complex): {name, dtype, missingvalue}
    (h5py.h5t.INTEGER, 1, False): dict(
        name="s8", dtype=np.int8, missingvalue=np.iinfo(np.int8).min
    ),
    (h5py.h5t.INTEGER, 2, False): dict(
        name="s16", dtype=np.int16, missingvalue=np.iinfo(np.int16).min
    ),
    (h5py.h5t.INTEGER, 4, False): dict(
        name="s32", dtype=np.int32, missingvalue=np.iinfo(np.int32).min
    ),
    (h5py.h5t.INTEGER, 8, False): dict(
        name="s64", dtype=np.int64, missingvalue=np.iinfo(np.int64).min
    ),
    (h5py.h5t.FLOAT, 4, False): dict(name="f32", dtype=np.float32, missingvalue=np.nan),
    (h5py.h5t.FLOAT, 8, False): dict(name="f64", dtype=np.float64, missingvalue=np.nan),
    (h5py.h5t.INTEGER, 1, True): dict(
        name="sc8",
        dtype=np.dtype([("r", np.int8), ("i", np.int8)]),
        missingvalue=(np.iinfo(np.int8).min,) * 2,
    ),
    (h5py.h5t.INTEGER, 2, True): dict(
        name="sc16",
        dtype=np.dtype([("r", np.int16), ("i", np.int16)]),
        missingvalue=(np.iinfo(np.int16).min,) * 2,
    ),
    (h5py.h5t.INTEGER, 4, True): dict(
        name="sc32",
        dtype=np.dtype([("r", np.int32), ("i", np.int32)]),
        missingvalue=(np.iinfo(np.int32).min,) * 2,
    ),
    (h5py.h5t.INTEGER, 8, True): dict(
        name="sc64",
        dtype=np.dtype([("r", np.int64), ("i", np.int64)]),
        missingvalue=(np.iinfo(np.int64).min,) * 2,
    ),
    (h5py.h5t.FLOAT, 4, True): dict(
        name="fc32", dtype=np.complex64, missingvalue=(np.nan + np.nan * 1j)
    ),
    (h5py.h5t.FLOAT, 8, True): dict(
        name="fc64", dtype=np.complex128, missingvalue=(np.nan + np.nan * 1j)
    ),
}


def get_h5type(cls, size, is_complex):
    try:
        typedict = H5T_LOOKUP[(cls, size, is_complex)]
    except KeyError:
        raise ValueError("HDF5 data type not supported for reading.")
    return typedict


class digital_rf_channel_source(gr.sync_block):
    """Source block for reading a channel of Digital RF data."""

    def __init__(
        self,
        channel_dir,
        start=None,
        end=None,
        repeat=False,
        gapless=False,
        min_chunksize=None,
    ):
        """Read a channel of data from a Digital RF directory.

        In addition to outputting samples from Digital RF format data, this
        block also emits a 'properties' message containing inherent channel
        properties and adds stream tags using the channel's accompanying
        Digital Metadata. See the Notes section for details on what the
        messages and stream tags contain.


        Parameters
        ----------
        channel_dir : string | list of strings
            Either a single channel directory containing 'drf_properties.h5'
            and timestamped subdirectories with Digital RF files, or a list of
            such. A directory can be a file system path or a url, where the url
            points to a channel directory. Each must be a local path, or start
            with 'http://'', 'file://'', or 'ftp://''.


        Other Parameters
        ----------------
        start : None | int | float | string, optional
            A value giving the start of the channel's playback.
            If None or '', the start of the channel's available data is used.
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a UTC timestamp (seconds since
            epoch).
            If a string, four forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a string beginning with '+' and followed by an integer
                    (float) expression, interpreted as samples (seconds) from
                    the start of the data, and
                3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
                4) 'now' ('nowish'), indicating the current time (rounded up)

        end : None | int | float | string, optional
            A value giving the end of the channel's playback.
            If None or '', the end of the channel's available data is used.
            See `start` for a description of how this value is interpreted.

        repeat : bool, optional
            If True, loop the data continuously from the start after the end
            is reached. If False, stop after the data is read once.

        gapless : bool, optional
            If True, output default-filled samples for any missing data between
            start and end. If False, skip missing samples and add an `rx_time`
            stream tag to indicate the gap.

        min_chunksize : None | int, optional
            Minimum number of samples to output at once. This value can be used
            to adjust the source's performance to reduce underruns and
            processing time. If None, a sensible default will be used.


        Notes
        -----
        A channel directory must contain subdirectories/files in the format:
            [YYYY-MM-DDTHH-MM-SS]/rf@[seconds].[%03i milliseconds].h5

        Each directory provided is considered the same channel. An error is
        raised if their sample rates differ, or if their time periods overlap.

        Upon start, this block sends a 'properties' message on its output
        message port that contains a dictionary with one key, the channel's
        name, and a value which is a dictionary of properties found in the
        channel's 'drf_properties.h5' file.

        This block emits the following stream tags at the appropriate sample
        for each of the channel's accompanying Digital Metadata samples:

            rx_time : (int secs, float frac) tuple
                Time since epoch of the sample.

            rx_rate : float
                Sample rate in Hz.

            rx_freq : float | 1-D array of floats
                Center frequency or frequencies of the subchannels based on
                the 'center_frequencies' metadata field.

            metadata : dict
                Any additional Digital Metadata fields are added to this
                dictionary tag of metadata.

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
            raise ValueError("Channel directories must have the same name.")
        self._ch = ch

        self._Reader = DigitalRFReader(top_level_dirs)
        self._properties = self._Reader.get_properties(self._ch)

        typeclass = self._properties["H5Tget_class"]
        itemsize = self._properties["H5Tget_size"]
        is_complex = self._properties["is_complex"]
        vlen = self._properties["num_subchannels"]
        sr = self._properties["samples_per_second"]

        self._itemsize = itemsize
        self._sample_rate = sr
        self._sample_rate_pmt = pmt.from_double(float(sr))

        # determine output signature from HDF5 type metadata
        typedict = get_h5type(typeclass, itemsize, is_complex)
        self._outtype = typedict["name"]
        self._itemtype = typedict["dtype"]
        self._missingvalue = np.zeros((), dtype=self._itemtype)
        self._missingvalue[()] = typedict["missingvalue"]
        self._fillvalue = np.zeros((), dtype=self._itemtype)
        if np.issubdtype(self._itemtype, np.inexact) and np.isnan(self._missingvalue):
            self._ismissing = lambda a: np.isnan(a)
        else:
            self._ismissing = lambda a: a == self._missingvalue
        if vlen == 1:
            out_sig = [self._itemtype]
        else:
            out_sig = [(self._itemtype, vlen)]

        gr.sync_block.__init__(
            self, name="digital_rf_channel_source", in_sig=None, out_sig=out_sig
        )

        self.message_port_register_out(pmt.intern("properties"))
        self._id = pmt.intern(self._ch)
        self._tag_queue = {}

        self._start = start
        self._end = end
        self._repeat = repeat
        self._gapless = gapless
        if min_chunksize is None:
            # FIXME: it shouldn't have to be quite this high
            self._min_chunksize = int(sr)
        else:
            self._min_chunksize = min_chunksize

        # reduce CPU usage and underruns by setting a minimum number of samples
        # to handle at once
        # (really want to set_min_noutput_items, but no way to do that from
        #  Python)
        try:
            self.set_output_multiple(self._min_chunksize)
        except RuntimeError:
            traceback.print_exc()
            errstr = "Failed to set source block min_chunksize to {min_chunksize}."
            if min_chunksize is None:
                errstr += (
                    " This value was calculated automatically based on the sample rate."
                    " You may have to specify min_chunksize manually."
                )
            raise ValueError(errstr.format(min_chunksize=self._min_chunksize))

        try:
            self._DMDReader = self._Reader.get_digital_metadata(self._ch)
        except IOError:
            self._DMDReader = None

    def _queue_tags(self, sample, tags):
        """Queue stream tags to be attached to data in the work function.

        In addition to the tags specified in the `tags` dictionary, this will
        add `rx_time` and `rx_rate` tags giving the sample time and rate.


        Parameters
        ----------
        sample : int
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
            time = sample / self._sample_rate
            tag_dict["rx_time"] = pmt.make_tuple(
                pmt.from_uint64(int(np.uint64(time))), pmt.from_double(float(time % 1))
            )
            tag_dict["rx_rate"] = self._sample_rate_pmt
        for k, v in tags.items():
            try:
                pmt_val = pmt.to_pmt(v)
            except ValueError:
                traceback.print_exc()
                errstr = (
                    "Can't add tag for '{0}' because its value of {1} failed"
                    " to convert to a pmt value."
                )
                print(errstr.format(k, v))
            else:
                tag_dict[k] = pmt_val
        self._tag_queue[sample] = tag_dict

    def start(self):
        self._bounds = self._Reader.get_bounds(self._ch)
        self._start_sample = util.parse_identifier_to_sample(
            self._start, self._sample_rate, self._bounds[0]
        )
        self._end_sample = util.parse_identifier_to_sample(
            self._end, self._sample_rate, self._bounds[0]
        )
        if self._start_sample is None:
            self._read_start_sample = self._bounds[0]
        else:
            self._read_start_sample = self._start_sample
        # add default tags to first sample
        self._queue_tags(self._read_start_sample, {})
        # replace longdouble samples_per_second with float for pmt conversion
        properties_message = self._properties.copy()
        properties_message["samples_per_second"] = float(
            properties_message["samples_per_second"]
        )
        self.message_port_pub(
            pmt.intern("properties"), pmt.to_pmt({self._ch: properties_message})
        )
        return super(digital_rf_channel_source, self).start()

    def work(self, input_items, output_items):
        out = output_items[0]
        nsamples = len(out)
        next_index = 0
        # repeat reading until we succeed or return
        while next_index < nsamples:
            read_start = self._read_start_sample
            # read_end is inclusive, hence the -1
            read_end = self._read_start_sample + (nsamples - next_index) - 1
            # creating a read function that has an output argument so data can
            # be copied directly would be nice
            # also should move EOFError checking into reader once watchdog
            # bounds functionality is implemented
            try:
                if self._end_sample is None:
                    if read_end > self._bounds[1]:
                        self._bounds = self._Reader.get_bounds(self._ch)
                        read_end = min(read_end, self._bounds[1])
                else:
                    if read_end > self._end_sample:
                        read_end = self._end_sample
                if read_start > read_end:
                    raise EOFError
                # read data
                data_dict = self._Reader.read(read_start, read_end, self._ch)
                # handled all samples through read_end regardless of whether
                # they were written to the output vector
                self._read_start_sample = read_end + 1
                # early escape for no data
                if not data_dict:
                    if self._gapless:
                        # output empty samples if no data and gapless output
                        stop_index = next_index + read_end + 1 - read_start
                        out[next_index:stop_index] = self._fillvalue
                        next_index = stop_index
                    else:
                        # clear any existing tags
                        self._tag_queue.clear()
                        # add tag at next potential sample to indicate skip
                        self._queue_tags(self._read_start_sample, {})
                    continue
                # read corresponding metadata
                if self._DMDReader is not None:
                    meta_dict = self._DMDReader.read(read_start, read_end)
                    for sample, meta in meta_dict.items():
                        # add tags from Digital Metadata
                        # (in addition to default time and rate tags)
                        # eliminate sample_rate_* tags with duplicate info
                        meta.pop("sample_rate_denominator", None)
                        meta.pop("sample_rate_numerator", None)
                        # get center frequencies for rx_freq tag, squeeze()[()]
                        # to get single value if possible else pass as an array
                        cf = meta.pop("center_frequencies", None)
                        if cf is not None:
                            cf = cf.ravel().squeeze()[()]
                        tags = dict(
                            rx_freq=cf,
                            # all other metadata goes in metadata tag
                            metadata=meta,
                        )
                        self._queue_tags(sample, tags)

                # add data and tags to output
                next_continuous_sample = read_start
                for sample, data in data_dict.items():
                    # detect data skip
                    if sample > next_continuous_sample:
                        if self._gapless:
                            # advance output by skipped number of samples
                            nskipped = sample - next_continuous_sample
                            sample_index = next_index + nskipped
                            out[next_index:sample_index] = self._fillvalue
                            next_index = sample_index
                        else:
                            # emit new time tag at sample to indicate skip
                            self._queue_tags(sample, {})
                    # output data
                    n = data.shape[0]
                    stop_index = next_index + n
                    end_sample = sample + n
                    out_dest = out[next_index:stop_index]
                    data_arr = data.squeeze()
                    out_dest[:] = data_arr
                    # overwrite missing values with fill values
                    missing_val_idx = self._ismissing(data_arr)
                    out_dest[missing_val_idx] = self._fillvalue
                    # output tags
                    for tag_sample in sorted(self._tag_queue.keys()):
                        if tag_sample < sample:
                            # drop tags from before current data block
                            del self._tag_queue[tag_sample]
                            continue
                        elif tag_sample >= end_sample:
                            # wait to output tags from after current data block
                            break
                        offset = (
                            self.nitems_written(0)  # offset @ start of work
                            + next_index  # additional offset of data block
                            + (tag_sample - sample)
                        )
                        tag_dict = self._tag_queue.pop(tag_sample)
                        for name, val in tag_dict.items():
                            self.add_item_tag(
                                0, offset, pmt.intern(name), val, self._id
                            )
                    # advance next output index and continuous sample
                    next_index = stop_index  # <=== next_index += n
                    next_continuous_sample = end_sample
            except EOFError:
                if self._repeat:
                    if self._start_sample is None:
                        self._read_start_sample = self._bounds[0]
                    else:
                        self._read_start_sample = self._start_sample
                    self._queue_tags(self._read_start_sample, {})
                    continue
                else:
                    break
        if next_index == 0:
            # return WORK_DONE
            return -1
        return next_index

    def get_gapless(self):
        return self._gapless

    def set_gapless(self, gapless):
        self._gapless = gapless

    def get_repeat(self):
        return self._repeat

    def set_repeat(self, repeat):
        self._repeat = repeat


class digital_rf_source(gr.hier_block2):
    """Source block for reading Digital RF data."""

    def __init__(
        self,
        top_level_dir,
        channels=None,
        start=None,
        end=None,
        repeat=False,
        throttle=False,
        gapless=False,
        min_chunksize=None,
    ):
        """Read data from a directory containing Digital RF channels.

        In addition to outputting samples from Digital RF format data, this
        block also emits a 'properties' message containing inherent channel
        properties and adds stream tags using the channel's accompanying
        Digital Metadata. See the Notes section for details on what the
        messages and stream tags contain.


        Parameters
        ----------
        top_level_dir : string
            Either a single top-level directory containing Digital RF channel
            directories, or a list of such. A directory can be a file system
            path or a url, where the url points to a top level directory. Each
            must be a local path, or start with 'http://'', 'file://'', or
            'ftp://''.


        Other Parameters
        ----------------
        channels : None | string | int | iterable of previous, optional
            If None, use all available channels in alphabetical order.
            Otherwise, use the channels in the order specified in the given
            iterable (a string or int is taken as a single-element iterable).
            A string is used to specify the channel name, while an int is used
            to specify the channel index in the sorted list of available
            channel names.

        start : None | string | int | iterable of previous, optional
            Can be a single value or an iterable of values corresponding to
            `channels` giving the start of the channel's playback.
            If None or '', the start of the channel's available data is used.
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a UTC timestamp (seconds since
            epoch).
            If a string, four forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a string beginning with '+' and followed by an integer
                    (float) expression, interpreted as samples (seconds) from
                    the start of the data, and
                3) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
                4) 'now' ('nowish'), indicating the current time (rounded up)

        end : None | string | int | iterable of previous, optional
            Can be a single value or an iterable of values corresponding to
            `channels` giving the end of the channel's playback.
            If None or '', the end of the channel's available data is used.
            See `start` for a description of how this value is interpreted.

        repeat : bool, optional
            If True, loop the data continuously from the start after the end
            is reached. If False, stop after the data is read once.

        throttle : bool, optional
            If True, playback the samples at their recorded sample rate. If
            False, read samples as quickly as possible.

        gapless : bool, optional
            If True, output zeroed samples for any missing data between start
            and end. If False, skip missing samples and add an `rx_time` stream
            tag to indicate the gap.

        min_chunksize : None | int, optional
            Minimum number of samples to output at once. This value can be used
            to adjust the source's performance to reduce underruns and
            processing time. If None, a sensible default will be used.

        Notes
        -----
        A top-level directory must contain files in the format:
            [channel]/[YYYY-MM-DDTHH-MM-SS]/rf@[seconds].[%03i milliseconds].h5

        If more than one top level directory contains the same channel_name
        subdirectory, this is considered the same channel. An error is raised
        if their sample rates differ, or if their time periods overlap.

        Upon start, this block sends 'properties' messages on its output
        message port that contains a dictionaries with one key, the channel's
        name, and a value which is a dictionary of properties found in the
        channel's 'drf_properties.h5' file.

        This block emits the following stream tags at the appropriate sample
        for each of the channel's accompanying Digital Metadata samples:

            rx_time : (int secs, float frac) tuple
                Time since epoch of the sample.

            rx_rate : float
                Sample rate in Hz.

            rx_freq : float | 1-D array of floats
                Center frequency or frequencies of the subchannels based on
                the 'center_frequencies' metadata field.

            metadata : dict
                Any additional Digital Metadata fields are added to this
                dictionary tag of metadata.

        """
        options = locals()
        del options["self"]
        del options["top_level_dir"]
        del options["channels"]
        del options["start"]
        del options["end"]
        del options["throttle"]

        Reader = DigitalRFReader(top_level_dir)
        available_channel_names = Reader.get_channels()
        self._channel_names = self._get_channel_names(channels, available_channel_names)

        if start is None or isinstance(start, six.string_types):
            start = [start] * len(self._channel_names)
        try:
            s_iter = iter(start)
        except TypeError:
            s_iter = iter([start])
        if end is None or isinstance(end, six.string_types):
            end = [end] * len(self._channel_names)
        try:
            e_iter = iter(end)
        except TypeError:
            e_iter = iter([end])

        # make sources for each channel
        self._channels = []
        for ch, s, e in zip(self._channel_names, s_iter, e_iter):
            chsrc = digital_rf_channel_source(
                os.path.join(top_level_dir, ch), start=s, end=e, **options
            )
            self._channels.append(chsrc)

        out_sig_dtypes = [list(src.out_sig())[0] for src in self._channels]
        out_sig = gr.io_signaturev(
            len(out_sig_dtypes),
            len(out_sig_dtypes),
            [s.itemsize for s in out_sig_dtypes],
        )
        in_sig = gr.io_signature(0, 0, 0)

        gr.hier_block2.__init__(
            self,
            name="digital_rf_source",
            input_signature=in_sig,
            output_signature=out_sig,
        )

        msg_port_name = pmt.intern("properties")
        self.message_port_register_hier_out("properties")

        for k, src in enumerate(self._channels):
            if throttle:
                throt = gnuradio.blocks.throttle(
                    list(src.out_sig())[0].itemsize,
                    float(src._sample_rate),
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
            if ch is None or ch == "":
                # use first available channel (alphabetical)
                try:
                    ch_name = unselected_channels[0]
                except IndexError:
                    raise ValueError(
                        '"None" invalid for channel, all available '
                        "channels have been selected."
                    )
            else:
                # try ch as a list index into available channels
                try:
                    ch_name = available_channel_names[int(ch)]
                except (TypeError, ValueError):
                    # not an index, that's fine
                    ch_name = ch
                except IndexError:
                    raise IndexError("Channel index {0} does not exist.".format(ch))

            # now assume ch is a string, get from unselected channel list
            try:
                unselected_channels.remove(ch_name)
            except ValueError:
                errstr = "Channel {0} does not exist or has already been " "selected."
                raise ValueError(errstr.format(ch_name))
            channel_names.append(ch_name)
        return channel_names

    def get_gapless(self):
        return self._channels[0]._gapless

    def set_gapless(self, gapless):
        for ch in self._channels:
            ch.set_gapless(gapless)

    def get_repeat(self):
        return self._channels[0]._repeat

    def set_repeat(self, repeat):
        for ch in self._channels:
            ch.set_repeat(repeat)
