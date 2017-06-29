# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Module defining a Digital RF Source block."""
import os
import warnings
from collections import OrderedDict

import numpy as np
import pmt
from gnuradio import gr
import six

from digital_rf import (DigitalMetadataWriter, DigitalRFWriter,
                        _py_rf_write_hdf5, util)


def parse_time_pmt(val, samples_per_second):
    """Get (sec, frac, idx) from an rx_time pmt value."""
    tsec = np.uint64(pmt.to_uint64(pmt.tuple_ref(val, 0)))
    tfrac = pmt.to_double(pmt.tuple_ref(val, 1))
    # calculate sample index of time and floor to uint64
    tidx = np.uint64(tsec*samples_per_second + tfrac*samples_per_second)
    return long(tsec), tfrac, long(tidx)


def translate_rx_freq(tag):
    """Translate 'rx_freq' tag to 'center_frequencies' metadata sample."""
    offset = tag.offset
    key = 'center_frequencies'
    # put array in a list because we want the data to be a 1-D array and it
    # would be a single value if we didn't and the array has length 1
    val = [np.array(pmt.to_python(tag.value), ndmin=1)]
    yield offset, key, val


def translate_metadata(tag):
    """Translate 'metadata' dictionary tag to metadata samples."""
    offset = tag.offset
    md = pmt.to_python(tag.value)
    try:
        for key, val in md.items():
            yield offset, key, val
    except AttributeError:
        wrnstr = (
            "Received 'metadata' stream tag that isn't a dictionary. Ignoring."
        )
        warnings.warn(wrnstr)


def collect_tags_in_dict(tags, translator, tag_dict={}):
    """Add the stream tags to `tag_dict` by their offset."""
    for tag in tags:
        for offset, key, val in translator(tag):
            # add tag as its own dictionary to tag_dict[offset]
            tag_dict.setdefault(offset, {}).update(((key, val),))


def recursive_dict_update(d, u):
    """Update d with values from u, recursing into sub-dictionaries."""
    for k, v in u.items():
        if isinstance(v, dict):
            recursive_dict_update(d.setdefault(k, {}), v)
        else:
            d[k] = v


class digital_rf_channel_sink(gr.sync_block):
    """Sink block for writing a channel of Digital RF data."""
    def __init__(
        self, channel_dir, dtype, subdir_cadence_secs,
        file_cadence_millisecs, sample_rate_numerator, sample_rate_denominator,
        start=None, ignore_tags=False, is_complex=True, num_subchannels=1,
        uuid_str=None, center_frequencies=None, metadata={},
        is_continuous=True, compression_level=0,
        checksum=False, marching_periods=True, stop_on_skipped=True,
        debug=False,
    ):
        """Write a channel of data in Digital RF format.

        In addition to storing the input samples in Digital RF format, this
        block also populates the channel's accompanying Digital Metadata
        at the sample indices when the metadata changes or a data skip occurs.
        See the Notes section for details on what metadata is stored.

        Parameters
        ----------

        channel_dir : string
            The directory where this channel is to be written. It will be
            created if it does not exist. The basename (last component) of the
            path is considered the channel's name for reading purposes.

        dtype : numpy.dtype | object to be cast by numpy.dtype()
            Object that gives the numpy dtype of the data to be written. This
            value is passed into ``numpy.dtype`` to get the actual dtype
            (e.g. ``numpy.dtype('>i4')``). Scalar types, complex types, and
            structured complex types with 'r' and 'i' fields of scalar types
            are valid.

        subdir_cadence_secs : int
            The number of seconds of data to store in one subdirectory. The
            timestamp of any subdirectory will be an integer multiple of this
            value.

        file_cadence_millisecs : int
            The number of milliseconds of data to store in each file. Note that
            an integer number of files must exactly span a subdirectory,
            implying::

                (subdir_cadence_secs*1000 % file_cadence_millisecs) == 0

        sample_rate_numerator : long | int
            Numerator of sample rate in Hz.

        sample_rate_denominator : long | int
            Denominator of sample rate in Hz.


        Other Parameters
        ----------------

        start : None | int/long | float | string, optional
            A value giving the time/index of the channel's first sample. When
            `ignore_tags` is False, 'rx_time' tags will be used to identify
            data gaps and skip the sample index forward appropriately (tags
            that refer to an earlier time will be ignored).
            If None or '' and `ignore_tags` is False, drop data until an
            'rx_time' tag arrives and sets the start time (a ValueError is
            raised if `ignore_tags` is True).
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a timestamp (seconds since epoch).
            If a string, two forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

        ignore_tags : bool, optional
            If True, do not use 'rx_time' tags to set the sample index and do
            not write other tags as Digital Metadata.

        is_complex : bool, optional
            This parameter is only used when `dtype` is not complex.
            If True (the default), interpret supplied data as interleaved
            complex I/Q samples. If False, each sample has a single value.

        num_subchannels : int, optional
            Number of subchannels to write simultaneously. Default is 1.

        uuid_str : None | string, optional
            UUID string that will act as a unique identifier for the data and
            can be used to tie the data files to metadata. If None, a random
            UUID will be generated.

        center_frequencies : None | array_like of floats, optional
            List of subchannel center frequencies to include in initial
            metadata. If None, ``[0.0]*num_subchannels`` will be used.
            Subsequent center frequency metadata samples can be written using
            'rx_freq' stream tags.

        metadata : dict, optional
            Dictionary of additional metadata to include in initial Digital
            Metadata sample. Subsequent metadata samples can be written
            using 'metadata' stream tags, but all keys intended to be included
            should be set here first even if their values are empty.

        is_continuous : bool, optional
            If True, data will be written in continuous blocks. If False data
            will be written with gapped blocks. Fastest write/read speed is
            achieved with `is_continuous` True, `checksum` False, and
            `compression_level` 0 (all defaults).

        compression_level : int, optional
            0 for no compression (default), 1-9 for varying levels of gzip
            compression (1 == least compression, least CPU; 9 == most
            compression, most CPU).

        checksum : bool, optional
            If True, use HDF5 checksum capability. If False (default), no
            checksum.

        marching_periods : bool, optional
            If True, write a period to stdout for every subdirectory when
            writing.

        stop_on_skipped : bool, optional
            If True, stop writing when a sample would be skipped (such as from
            a dropped packet).

        debug : bool, optional
            If True, print debugging information.


        Notes
        -----

        By convention, this block sets the following Digital Metadata fields:

            uuid_str : string
                Value provided by the `uuid_str` argument.

            sample_rate_numerator : int
                Value provided by the `sample_rate_numerator` argument.

            sample_rate_denominator : int
                Value provided by the `sample_rate_denominator` argument.

            center_frequencies : list of floats with length `num_subchannels`
                Subchannel center frequencies as specified by
                `center_frequencies` argument and 'rx_freq' stream tags.

        Additional metadata fields can be set using the `metadata` argument and
        stream tags. Nested dictionaries are permitted and are helpful for
        grouping properties. For example, receiver-specific metadata is
        typically specified with a sub-dictionary using the 'receiver' field.


        This block acts on the following stream tags when `ignore_tags` is
        False:

            rx_time : (int secs, float frac) tuple
                Used to set the sample index from the given time since epoch.

            rx_freq : float
                Used to set the 'center_frequencies' value in the channel's
                Digital Metadata as described above.

            metadata : dict
                Used to populate additional (key, value) pairs in the channel's
                Digital Metadata. Any keys passed in 'metadata' tags should be
                included in the `metadata` argument at initialization to ensure
                that they always exist in the Digital Metadata.

        """
        dtype = np.dtype(dtype)
        # create structured dtype for interleaved samples if necessary
        if is_complex and (not np.issubdtype(dtype, np.complexfloating) and
                           not dtype.names):
            realdtype = dtype
            dtype = np.dtype([('r', realdtype), ('i', realdtype)])

        if num_subchannels == 1:
            in_sig = [dtype]
        else:
            in_sig = [(dtype, num_subchannels)]

        gr.sync_block.__init__(
            self,
            name="digital_rf_channel_sink",
            in_sig=in_sig,
            out_sig=None,
        )

        self._channel_dir = channel_dir
        self._dtype = dtype
        self._subdir_cadence_secs = subdir_cadence_secs
        self._file_cadence_millisecs = file_cadence_millisecs
        self._sample_rate_numerator = sample_rate_numerator
        self._sample_rate_denominator = sample_rate_denominator
        self._uuid_str = uuid_str
        self._ignore_tags = ignore_tags
        self._is_complex = is_complex
        self._num_subchannels = num_subchannels
        self._is_continuous = is_continuous
        self._compression_level = compression_level
        self._checksum = checksum
        self._marching_periods = marching_periods
        self._stop_on_skipped = stop_on_skipped
        self._debug = debug

        self._samples_per_second = (
            np.longdouble(np.uint64(sample_rate_numerator)) /
            np.longdouble(np.uint64(sample_rate_denominator))
        )
        # will be None if start is None or ''
        self._start_sample = util.parse_sample_identifier(
            start, self._samples_per_second, None,
        )

        # create metadata dictionary that will be updated and written whenever
        # new metadata is received in stream tags
        self._metadata = metadata.copy()
        if center_frequencies is None:
            center_frequencies = np.array([0.0]*self._num_subchannels)
        else:
            center_frequencies = np.ascontiguousarray(center_frequencies)
        self._metadata.update(
            # standard metadata by convention
            uuid_str='',
            sample_rate_numerator=self._sample_rate_numerator,
            sample_rate_denominator=self._sample_rate_denominator,
            # put in a list because we want the data to be a 1-D array and it
            # would be a single value if we didn't and the array has length 1
            center_frequencies=[center_frequencies],
        )

        # create directories for RF data channel and metadata
        self._metadata_dir = os.path.join(self._channel_dir, 'metadata')
        if not os.path.exists(self._metadata_dir):
            os.makedirs(self._metadata_dir)

        # dict of blocks to be written
        # keys: block index into data
        # values: relative sample index from start_sample
        self._blocks = OrderedDict()
        if self._start_sample is not None:
            # first sample in block ([0]) will be 0 relative to start_sample
            self._blocks[0] = 0

        if self._start_sample is None:
            if self._ignore_tags:
                raise ValueError('Must specify start if ignore_tags is True.')
            else:
                # need to wait to create DigitalRFWriter until we know the
                # start sample from an 'rx_time' tag
                self._Writer = None
        else:
            self._create_writer()

        # dict of metadata samples to be written
        # keys: block index into data (even though we don't need it for
        #   writing, used to not write metadata when there is no data)
        # values: (relative sample index, metadata dictionary) tuple
        self._mdsamples = OrderedDict()
        if self._start_sample is not None:
            # missing values filled from self._metadata when writing
            self._mdsamples[0] = (0, {})

    def _create_writer(self):
        # Digital RF writer
        self._Writer = DigitalRFWriter(
            self._channel_dir, self._dtype, self._subdir_cadence_secs,
            self._file_cadence_millisecs, self._start_sample,
            self._sample_rate_numerator, self._sample_rate_denominator,
            uuid_str=self._uuid_str, compression_level=self._compression_level,
            checksum=self._checksum, is_complex=self._is_complex,
            num_subchannels=self._num_subchannels,
            is_continuous=self._is_continuous,
            marching_periods=self._marching_periods,
        )
        # update UUID in metadata after parsing by DigitalRFWriter
        self._metadata.update(uuid_str=self._Writer.uuid)
        # Digital Metadata writer
        self._DMDWriter = DigitalMetadataWriter(
            metadata_dir=self._metadata_dir,
            subdir_cadence_secs=self._subdir_cadence_secs,
            file_cadence_secs=1,
            sample_rate_numerator=self._sample_rate_numerator,
            sample_rate_denominator=self._sample_rate_denominator,
            file_name='metadata',
        )

    def _add_blocks_from_time_tags(self, time_tags, in_data):
        """Add to self._blocks based on time tags, return updated `in_data`."""
        nread = self.nitems_read(0)
        for tag in time_tags:
            offset = tag.offset
            tsec, tfrac, tidx = parse_time_pmt(
                tag.value, self._samples_per_second,
            )
            if self._debug:
                tagstr = "Time tag @ sample {0} ({1}): {2}+{3}.".format(
                    offset, tidx, tsec, tfrac,
                )
                print(tagstr)

            # index into data block for this tag
            bidx = offset - nread

            if self._start_sample is None:
                # first time tag, set start_sample and create Writer
                self._start_sample = tidx
                self._create_writer()
                # drop beginning of data before first time tag so we can
                # have bidx equal 0 as required for rf_write_blocks
                in_data = in_data[bidx:]
                # change nread so any subsequent tags have the correct
                # block index for the truncated in_data
                nread = offset
                # add to _blocks and _mdsamples so we know to start writing now
                # (they are empty if we are here)
                self._blocks[0] = 0
                self._mdsamples[0] = (0, {})
            else:
                # get sample index relative to start
                sidx = tidx - self._start_sample

                # add new data block if valid and it indicates a gap
                prev_bidx, prev_sidx = (
                    reversed(self._blocks.items()).next()
                )
                next_continuous_sample = prev_sidx + (bidx - prev_bidx)
                if sidx < next_continuous_sample:
                    if self._debug:
                        errstr = (
                            "Time tag is invalid: time cannot go backwards"
                            " from index {0}. Skipping."
                        ).format(self._start_sample + next_continuous_sample)
                        print(errstr)
                    continue
                elif sidx == next_continuous_sample:
                    # don't create a new block because it's continuous
                    continue
                else:
                    # add new block to write based on time tag (possibly
                    # overriding assumed continuous write at bidx==0)
                    self._blocks[bidx] = sidx
        return in_data, nread

    def _add_metadata_from_tags(self, tags, nread):
        """Add to self._mdsamples based on tags collected by offset."""
        block_indices = np.asarray(self._blocks.keys())
        rel_sample_indices = self._blocks.values()
        for offset, tag_dict in sorted(tags.items()):
            block_index = offset - nread
            # find index of the block the metadata sample is located in
            idx = np.searchsorted(block_indices, block_index, side='right') - 1
            # get the relative sample index for the metadata sample
            rel_sample = (rel_sample_indices[idx] +
                          (block_index - block_indices[idx]))
            self._mdsamples[block_index] = (rel_sample, tag_dict)

    def work(self, input_items, output_items):
        in_data = input_items[0]
        nsamples = len(in_data)

        if not self._ignore_tags:
            # read time tags
            time_tags = self.get_tags_in_window(
                0, 0, nsamples, pmt.intern('rx_time'),
            )
            # separate data into blocks to be written
            in_data, nread = self._add_blocks_from_time_tags(
                time_tags, in_data,
            )

            tags_by_offset = {}
            # read frequency tags
            freq_tags = self.get_tags_in_window(
                0, 0, nsamples, pmt.intern('rx_freq'),
            )
            collect_tags_in_dict(freq_tags, translate_rx_freq, tags_by_offset)
            # read metadata tags
            meta_tags = self.get_tags_in_window(
                0, 0, nsamples, pmt.intern('metadata'),
            )
            collect_tags_in_dict(meta_tags, translate_metadata, tags_by_offset)

            # separate tags into metadata samples to be written
            self._add_metadata_from_tags(tags_by_offset, nread)

        # check if skip occurs and break if so after writing continuous data
        if self._stop_on_skipped and len(self._blocks) > 1:
            rel_sample_indices = self._blocks.values()[:2]
            block_indices = self._blocks.keys()[:2]
            _py_rf_write_hdf5.rf_block_write(
                self._Writer._channelObj,
                in_data,
                np.ascontiguousarray(rel_sample_indices[:1], dtype=np.uint64),
                np.ascontiguousarray(block_indices[:1], dtype=np.uint64),
            )
            last_rel_sample = (rel_sample_indices[0] +
                               (block_indices[1] - block_indices[0]))
            for bidx, (rel_sample, md) in self._mdsamples.items():
                if rel_sample <= last_rel_sample:
                    sample = rel_sample + self._start_sample
                    # update self._metadata with new values and then write that
                    recursive_dict_update(self._metadata, md)
                    self._DMDWriter.write(sample, self._metadata)
            print("Stopping at skipped sample as requested.")
            # return WORK_DONE
            return -1

        # write metadata
        for bidx, (rel_sample, md) in self._mdsamples.items():
            sample = rel_sample + self._start_sample
            # update self._metadata with new values and then write that
            recursive_dict_update(self._metadata, md)
            self._DMDWriter.write(sample, self._metadata)
        self._mdsamples.clear()

        # write data using block writer
        if self._blocks:
            next_continuous_sample = _py_rf_write_hdf5.rf_block_write(
                self._Writer._channelObj,
                in_data,
                np.ascontiguousarray(self._blocks.values(), dtype=np.uint64),
                np.ascontiguousarray(self._blocks.keys(), dtype=np.uint64),
            )
            self._blocks.clear()
            # set up next write call assuming it will be continuous unless
            # overridden by a time tag
            self._blocks[0] = next_continuous_sample

        return nsamples

    def stop(self):
        if self._Writer is not None:
            self._Writer.close()
        return super(digital_rf_channel_sink, self).stop()

    def get_debug(self):
        return self._debug

    def set_debug(self, debug):
        self._debug = debug

    def get_ignore_tags(self):
        return self._ignore_tags

    def set_ignore_tags(self, ignore_tags):
        self._ignore_tags = ignore_tags

    def get_stop_on_skipped(self):
        return self._stop_on_skipped

    def set_stop_on_skipped(self, stop_on_skipped):
        self._stop_on_skipped = stop_on_skipped


class digital_rf_sink(gr.hier_block2):
    """Sink block for writing Digital RF data."""
    def __init__(
        self, top_level_dir, channels, dtype, subdir_cadence_secs,
        file_cadence_millisecs, sample_rate_numerator, sample_rate_denominator,
        start=None, ignore_tags=False, is_complex=True, num_subchannels=1,
        uuid_str=None, center_frequencies=None, metadata={},
        is_continuous=True, compression_level=0,
        checksum=False, marching_periods=True, stop_on_skipped=True,
        debug=False,
    ):
        """Write data in Digital RF format.

        This block is useful for writing multiple channels of data that have
        the same parameters. If different parameters for each channel are
        needed, use multiple `digital_rf_channel_sink` blocks.

        In addition to storing the input samples in Digital RF format, this
        block also populates each channel's accompanying Digital Metadata
        at the sample indices when the metadata changes or a data skip occurs.
        See the Notes section for details on what metadata is stored.

        Parameters
        ----------

        top_level_dir : string
            The top-level directory in which Digital RF channel directories
            will be created. It will be created if it does not exist.

        channels : list of strings | string
            List of channel names with length matching the number of channels
            to be written (and the number of inputs). Each channel name will
            be used as the directory name for the channel inside
            `top_level_dir`. If a string, a single channel will be written with
            that name.

        dtype : numpy.dtype | object to be cast by numpy.dtype()
            Object that gives the numpy dtype of the data to be written. This
            value is passed into ``numpy.dtype`` to get the actual dtype
            (e.g. ``numpy.dtype('>i4')``). Scalar types, complex types, and
            structured complex types with 'r' and 'i' fields of scalar types
            are valid.

        subdir_cadence_secs : int
            The number of seconds of data to store in one subdirectory. The
            timestamp of any subdirectory will be an integer multiple of this
            value.

        file_cadence_millisecs : int
            The number of milliseconds of data to store in each file. Note that
            an integer number of files must exactly span a subdirectory,
            implying::

                (subdir_cadence_secs*1000 % file_cadence_millisecs) == 0

        sample_rate_numerator : long | int
            Numerator of sample rate in Hz.

        sample_rate_denominator : long | int
            Denominator of sample rate in Hz.


        Other Parameters
        ----------------

        start : None | int/long | float | string, optional
            A value giving the time/index of the channel's first sample. When
            `ignore_tags` is False, 'rx_time' tags will be used to identify
            data gaps and skip the sample index forward appropriately (tags
            that refer to an earlier time will be ignored).
            If None or '' and `ignore_tags` is False, drop data until an
            'rx_time' tag arrives and sets the start time (a ValueError is
            raised if `ignore_tags` is True).
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a timestamp (seconds since epoch).
            If a string, two forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'

        ignore_tags : bool, optional
            If True, do not use 'rx_time' tags to set the sample index and do
            not write other tags as Digital Metadata.

        is_complex : bool, optional
            This parameter is only used when `dtype` is not complex.
            If True (the default), interpret supplied data as interleaved
            complex I/Q samples. If False, each sample has a single value.

        num_subchannels : int, optional
            Number of subchannels to write simultaneously. Default is 1.

        uuid_str : None | string, optional
            UUID string that will act as a unique identifier for the data and
            can be used to tie the data files to metadata. If None, a random
            UUID will be generated.

        center_frequencies : None | array_like of floats, optional
            List of subchannel center frequencies to include in initial
            metadata. If None, ``[0.0]*num_subchannels`` will be used.
            Subsequent center frequency metadata samples can be written using
            'rx_freq' stream tags.

        metadata : dict, optional
            Dictionary of additional metadata to include in initial Digital
            Metadata sample. Subsequent metadata samples can be written
            using 'metadata' stream tags, but all keys intended to be included
            should be set here first even if their values are empty.

        is_continuous : bool, optional
            If True, data will be written in continuous blocks. If False data
            will be written with gapped blocks. Fastest write/read speed is
            achieved with `is_continuous` True, `checksum` False, and
            `compression_level` 0 (all defaults).

        compression_level : int, optional
            0 for no compression (default), 1-9 for varying levels of gzip
            compression (1 == least compression, least CPU; 9 == most
            compression, most CPU).

        checksum : bool, optional
            If True, use HDF5 checksum capability. If False (default), no
            checksum.

        marching_periods : bool, optional
            If True, write a period to stdout for every subdirectory when
            writing.

        stop_on_skipped : bool, optional
            If True, stop writing when a sample would be skipped (such as from
            a dropped packet).

        debug : bool, optional
            If True, print debugging information.


        Notes
        -----

        By convention, this block sets the following Digital Metadata fields:

            uuid_str : string
                Value provided by the `uuid_str` argument.

            sample_rate_numerator : int
                Value provided by the `sample_rate_numerator` argument.

            sample_rate_denominator : int
                Value provided by the `sample_rate_denominator` argument.

            center_frequencies : list of floats with length `num_subchannels`
                Subchannel center frequencies as specified by
                `center_frequencies` argument and 'rx_rate' stream tags.

        Additional metadata fields can be set using the `metadata` argument and
        stream tags. Nested dictionaries are permitted and are helpful for
        grouping properties. For example, receiver-specific metadata is
        typically specified with a sub-dictionary using the 'receiver' field.


        This block acts on the following stream tags when `ignore_tags` is
        False:

            rx_time : (int secs, float frac) tuple
                Used to set the sample index from the given time since epoch.

            rx_freq : float
                Used to set the 'center_frequencies' value in the channel's
                Digital Metadata as described above.

            metadata : dict
                Used to populate additional (key, value) pairs in the channel's
                Digital Metadata. Any keys passed in 'metadata' tags should be
                included in the `metadata` argument at initialization to ensure
                that they always exist in the Digital Metadata.

        """
        options = locals()
        del options['self']
        del options['top_level_dir']
        del options['channels']

        self._top_level_dir = os.path.abspath(top_level_dir)
        if isinstance(channels, six.string_types):
            channels = [channels]
        self._channel_names = channels

        # make sinks for every channel
        self._channels = []
        for ch in self._channel_names:
            channel_dir = os.path.join(self._top_level_dir, ch)
            chsink = digital_rf_channel_sink(channel_dir, **options)
            self._channels.append(chsink)

        in_sig_dtypes = [sink.in_sig()[0] for sink in self._channels]
        in_sig = gr.io_signaturev(
            len(in_sig_dtypes), len(in_sig_dtypes),
            [s.itemsize for s in in_sig_dtypes],
        )
        out_sig = gr.io_signature(0, 0, 0)

        gr.hier_block2.__init__(
            self,
            name="digital_rf_sink",
            input_signature=in_sig,
            output_signature=out_sig,
        )

        for k, sink in enumerate(self._channels):
            self.connect((self, k), sink)

    def get_debug(self):
        return self._channels[0].get_debug()

    def set_debug(self, debug):
        for ch in self._channels:
            ch.set_debug(debug)

    def get_ignore_tags(self):
        return self._channels[0].ignore_tags()

    def set_ignore_tags(self, ignore_tags):
        for ch in self._channels:
            ch.set_ignore_tags(ignore_tags)

    def get_stop_on_skipped(self):
        return self._channels[0].get_stop_on_skipped()

    def set_stop_on_skipped(self, stop_on_skipped):
        for ch in self._channels():
            ch.set_stop_on_skipped(stop_on_skipped)
