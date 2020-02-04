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
import sys
import traceback
import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from itertools import chain, tee

import numpy as np
import pmt
import six
from gnuradio import gr
from six.moves import zip

from digital_rf import DigitalMetadataWriter, DigitalRFWriter, _py_rf_write_hdf5, util


def parse_time_pmt(val, samples_per_second):
    """Get (sec, frac, idx) from an rx_time pmt value."""
    tsec = np.uint64(pmt.to_uint64(pmt.tuple_ref(val, 0)))
    tfrac = pmt.to_double(pmt.tuple_ref(val, 1))
    # calculate sample index of time and floor to uint64
    tidx = np.uint64(tsec * samples_per_second + tfrac * samples_per_second)
    return int(tsec), tfrac, int(tidx)


def translate_rx_freq(tag):
    """Translate 'rx_freq' tag to 'center_frequencies' metadata sample."""
    offset = tag.offset
    key = "center_frequencies"
    val = np.array(pmt.to_python(tag.value), ndmin=1)
    yield offset, key, val


def translate_metadata(tag):
    """Translate 'metadata' dictionary tag to metadata samples."""
    offset = tag.offset
    md = pmt.to_python(tag.value)
    try:
        for key, val in md.items():
            yield offset, key, val
    except AttributeError:
        wrnstr = "Received 'metadata' stream tag that isn't a dictionary. Ignoring."
        warnings.warn(wrnstr)


def collect_tags_in_dict(tags, translator, tag_dict=None):
    """Add the stream tags to `tag_dict` by their offset."""
    if tag_dict is None:
        tag_dict = {}
    for tag in tags:
        for offset, key, val in translator(tag):
            # add tag as its own dictionary to tag_dict[offset]
            tag_dict.setdefault(offset, {}).update(((key, val),))


def recursive_dict_update(d, u):
    """Update d with values from u, recursing into sub-dictionaries."""
    for k, v in u.items():
        if isinstance(v, dict):
            try:
                # copy because we don't want to modify the sub-dictionary
                # just use its values to create an updated sub-dictionary
                subdict = d[k].copy()
            except KeyError:
                subdict = {}
            d[k] = recursive_dict_update(subdict, v)
        else:
            d[k] = v
    return d


def pairwise(iterable):
    """Return iterable elements in pairs, e.g. range(3) -> (0, 1), (1, 2)."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class digital_rf_channel_sink(gr.sync_block):
    """Sink block for writing a channel of Digital RF data."""

    def __init__(
        self,
        channel_dir,
        dtype,
        subdir_cadence_secs,
        file_cadence_millisecs,
        sample_rate_numerator,
        sample_rate_denominator,
        start=None,
        ignore_tags=False,
        is_complex=True,
        num_subchannels=1,
        uuid_str=None,
        center_frequencies=None,
        metadata=None,
        is_continuous=True,
        compression_level=0,
        checksum=False,
        marching_periods=True,
        stop_on_skipped=False,
        stop_on_time_tag=False,
        debug=False,
        min_chunksize=None,
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

        dtype : np.dtype | object to be cast by np.dtype()
            Object that gives the numpy dtype of the data to be written. This
            value is passed into ``np.dtype`` to get the actual dtype
            (e.g. ``np.dtype('>i4')``). Scalar types, complex types, and
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

        sample_rate_numerator : int
            Numerator of sample rate in Hz.

        sample_rate_denominator : int
            Denominator of sample rate in Hz.


        Other Parameters
        ----------------
        start : None | int | float | string, optional
            A value giving the time/index of the channel's first sample. When
            `ignore_tags` is False, 'rx_time' tags will be used to identify
            data gaps and skip the sample index forward appropriately (tags
            that refer to an earlier time will be ignored).
            If None or '' and `ignore_tags` is False, drop data until an
            'rx_time' tag arrives and sets the start time (a ValueError is
            raised if `ignore_tags` is True).
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a UTC timestamp (seconds since
            epoch).
            If a string, three forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
                3) 'now' ('nowish'), indicating the current time (rounded up)

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

        stop_on_time_tag : bool, optional
            If True, stop writing when any but an initial 'rx_time' tag is received.

        debug : bool, optional
            If True, print debugging information.

        min_chunksize : None | int, optional
            Minimum number of samples to consume at once. This value can be
            used to adjust the sink's performance to reduce processing time.
            If None, a sensible default will be used.


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
        if is_complex and (
            not np.issubdtype(dtype, np.complexfloating) and not dtype.names
        ):
            realdtype = dtype
            dtype = np.dtype([("r", realdtype), ("i", realdtype)])

        if num_subchannels == 1:
            in_sig = [dtype]
        else:
            in_sig = [(dtype, num_subchannels)]

        gr.sync_block.__init__(
            self, name="digital_rf_channel_sink", in_sig=in_sig, out_sig=None
        )

        self._channel_dir = channel_dir
        self._channel_name = os.path.basename(channel_dir)
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
        self._stop_on_time_tag = stop_on_time_tag
        self._debug = debug

        self._work_done = False

        self._samples_per_second = np.longdouble(
            np.uint64(sample_rate_numerator)
        ) / np.longdouble(np.uint64(sample_rate_denominator))

        if min_chunksize is None:
            self._min_chunksize = max(int(self._samples_per_second // 1000), 1)
        else:
            self._min_chunksize = min_chunksize

        # reduce CPU usage by setting a minimum number of samples to handle
        # at once
        # (really want to set_min_noutput_items, but no way to do that from
        #  Python)
        try:
            self.set_output_multiple(self._min_chunksize)
        except RuntimeError:
            traceback.print_exc()
            errstr = "Failed to set sink block min_chunksize to {min_chunksize}."
            if min_chunksize is None:
                errstr += (
                    " This value was calculated automatically based on the sample rate."
                    " You may have to specify min_chunksize manually."
                )
            raise ValueError(errstr.format(min_chunksize=self._min_chunksize))

        # will be None if start is None or ''
        self._start_sample = util.parse_identifier_to_sample(
            start, self._samples_per_second, None
        )
        if self._start_sample is None:
            if self._ignore_tags:
                raise ValueError("Must specify start if ignore_tags is True.")
            # data without a time tag will be written starting at global index
            # of 0, i.e. the Unix epoch
            # we don't want to guess the start time because the user would
            # know better and it could obscure bugs by setting approximately
            # the correct time (samples in 1970 are immediately obvious)
            self._start_sample = 0
        self._next_rel_sample = 0

        # stream tags to read (in addition to rx_time, handled specially)
        if LooseVersion(gr.version()) >= LooseVersion("3.7.12"):
            self._stream_tag_translators = {
                # disable rx_freq until we figure out what to do with polyphase
                # pmt.intern('rx_freq'): translate_rx_freq,
                pmt.intern("metadata"): translate_metadata
            }
        else:
            # USRP source in gnuradio < 3.7.12 has bad rx_freq tags, so avoid
            # trouble by ignoring rx_freq tags for those gnuradio versions
            self._stream_tag_translators = {pmt.intern("metadata"): translate_metadata}

        # create metadata dictionary that will be updated and written whenever
        # new metadata is received in stream tags
        if metadata is None:
            metadata = {}
        self._metadata = metadata.copy()
        if center_frequencies is None:
            center_frequencies = np.array([0.0] * self._num_subchannels)
        else:
            center_frequencies = np.ascontiguousarray(center_frequencies)
        self._metadata.update(
            # standard metadata by convention
            uuid_str="",
            sample_rate_numerator=self._sample_rate_numerator,
            sample_rate_denominator=self._sample_rate_denominator,
            center_frequencies=center_frequencies,
        )

        # create directories for RF data channel and metadata
        self._metadata_dir = os.path.join(self._channel_dir, "metadata")
        if not os.path.exists(self._metadata_dir):
            os.makedirs(self._metadata_dir)

        # sets self._Writer, self._DMDWriter, and adds to self._metadata
        self._create_writer()

        # dict of metadata samples to be written, add for first sample
        # keys: absolute sample index for metadata
        # values: metadata dictionary to update self._metadata and then write
        self._md_queue = defaultdict(dict)
        self._md_queue[self._start_sample] = {}

    def _create_writer(self):
        # Digital RF writer
        self._Writer = DigitalRFWriter(
            self._channel_dir,
            self._dtype,
            self._subdir_cadence_secs,
            self._file_cadence_millisecs,
            self._start_sample,
            self._sample_rate_numerator,
            self._sample_rate_denominator,
            uuid_str=self._uuid_str,
            compression_level=self._compression_level,
            checksum=self._checksum,
            is_complex=self._is_complex,
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
            file_name="metadata",
        )

    def _read_tags(self, nsamples):
        """Read stream tags and set data blocks and metadata to write.

        Metadata from tags is added to the queue at ``self._md_queue``.

        """
        nread = self.nitems_read(0)
        md_queue = self._md_queue

        # continue writing at next continuous sample with start of block
        # unless overridden by a time tag
        data_blk_idxs = [0]
        data_rel_samples = [self._next_rel_sample]

        # read time tags
        time_tags = self.get_tags_in_window(0, 0, nsamples, pmt.intern("rx_time"))
        if time_tags and self._stop_on_time_tag and self._next_rel_sample != 0:
            self._work_done = True
        # separate data into blocks to be written
        for tag in time_tags:
            offset = tag.offset
            tsec, tfrac, tidx = parse_time_pmt(tag.value, self._samples_per_second)

            # index into data block for this tag
            bidx = offset - nread

            # get sample index relative to start
            sidx = tidx - self._start_sample

            # add new data block if valid and it indicates a gap
            prev_bidx = data_blk_idxs[-1]
            prev_sidx = data_rel_samples[-1]
            next_continuous_sample = prev_sidx + (bidx - prev_bidx)
            if sidx < next_continuous_sample:
                if self._debug:
                    errstr = (
                        "\n|{0}|rx_time tag @ sample {1}: {2}+{3} ({4})"
                        "\n INVALID: time cannot go backwards from index {5}."
                        " Skipping."
                    ).format(
                        self._channel_name,
                        offset,
                        tsec,
                        tfrac,
                        tidx,
                        self._start_sample + next_continuous_sample,
                    )
                    sys.stdout.write(errstr)
                    sys.stdout.flush()
                continue
            elif sidx == next_continuous_sample:
                # don't create a new block because it's continuous
                if self._debug:
                    tagstr = ("\n|{0}|rx_time tag @ sample {1}: {2}+{3} ({4})").format(
                        self._channel_name, offset, tsec, tfrac, tidx
                    )
                    sys.stdout.write(tagstr)
                    sys.stdout.flush()
                continue
            else:
                # add new block to write based on time tag
                if self._debug:
                    tagstr = (
                        "\n|{0}|rx_time tag @ sample {1}: {2}+{3} ({4})"
                        "\n {5} dropped samples."
                    ).format(
                        self._channel_name,
                        offset,
                        tsec,
                        tfrac,
                        tidx,
                        sidx - next_continuous_sample,
                    )
                    sys.stdout.write(tagstr)
                    sys.stdout.flush()
                # set flag to stop work when stop_on_skipped is set
                if self._stop_on_skipped and self._next_rel_sample != 0:
                    self._work_done = True
                if bidx == 0:
                    # override assumed continuous write
                    # data_blk_idxs[0] is already 0
                    data_rel_samples[0] = sidx
                else:
                    data_blk_idxs.append(bidx)
                    data_rel_samples.append(sidx)
                # reset metadata queue with only valid values
                for md_idx in list(md_queue.keys()):
                    md_sidx = md_idx - self._start_sample
                    if next_continuous_sample <= md_sidx and md_sidx < sidx:
                        del md_queue[md_idx]
                # new metadata sample to help flag data skip
                md_queue.setdefault(sidx + self._start_sample, {})

        # read other tags by data block (so we know the sample index)
        for (bidx, bend), sidx in zip(
            pairwise(chain(data_blk_idxs, (nsamples,))), data_rel_samples
        ):
            tags_by_offset = {}
            # read tags, translate to metadata dict, add to tag dict
            for tag_name, translator in self._stream_tag_translators.items():
                tags = self.get_tags_in_window(0, bidx, bend, tag_name)
                collect_tags_in_dict(tags, translator, tags_by_offset)
            # add tags to metadata sample dictionary
            for offset, tag_dict in tags_by_offset.items():
                mbidx = offset - nread
                # get the absolute sample index for the metadata sample
                msidx = (sidx + (mbidx - bidx)) + self._start_sample
                md_queue[msidx].update(tag_dict)

        return data_blk_idxs, data_rel_samples

    def work(self, input_items, output_items):
        in_data = input_items[0]
        nsamples = len(in_data)

        if not self._ignore_tags:
            # break data into blocks from time tags
            # get metadata from other tags and add to self._md_queue
            data_blk_idxs, data_rel_samples = self._read_tags(nsamples)
        else:
            # continue writing at next continuous sample with start of block
            data_blk_idxs = [0]
            data_rel_samples = [self._next_rel_sample]

        # make index lists into uint64 arrays
        data_rel_samples = np.array(data_rel_samples, dtype=np.uint64, ndmin=1)
        data_blk_idxs = np.array(data_blk_idxs, dtype=np.uint64, ndmin=1)

        # get any metadata samples to be written from queue
        if self._md_queue:
            md_samples, md_dict_updates = zip(
                *sorted(self._md_queue.items(), key=lambda x: x[0])
            )
            md_samples = np.array(md_samples, dtype=np.uint64, ndmin=1)
            # fill out metadata to be written using stored metadata
            md_dicts = [
                # function updates self._metadata in-place, want copy for list
                # to preserve the updated values at that particular state
                recursive_dict_update(self._metadata, md_update).copy()
                for md_update in md_dict_updates
            ]
        else:
            md_samples = []
            md_dicts = []
        self._md_queue.clear()

        # check if work_done has been flagged (stop on skipped or time tag)
        if self._work_done:
            if (
                data_rel_samples[0] == self._next_rel_sample
                or self._next_rel_sample == 0
            ):
                # write continuous data from this chunk first
                last_rel_sample = _py_rf_write_hdf5.rf_block_write(
                    self._Writer._channelObj,
                    in_data,
                    data_rel_samples[:1],
                    data_blk_idxs[:1],
                )
                last_sample = last_rel_sample + self._start_sample
                idx = np.searchsorted(md_samples, last_sample, "right")
                for md_sample, md_dict in zip(md_samples[:idx], md_dicts[:idx]):
                    self._DMDWriter.write(md_sample, md_dict)
            print("Stopping as requested.")
            # return WORK_DONE
            return -1

        try:
            # write metadata
            if md_dicts:
                self._DMDWriter.write(md_samples, md_dicts)

            # write data using block writer
            self._next_rel_sample = _py_rf_write_hdf5.rf_block_write(
                self._Writer._channelObj, in_data, data_rel_samples, data_blk_idxs
            )
        except (IOError, RuntimeError):
            # just print the exception so we can return WORK_DONE to notify
            # other blocks to shut down cleanly
            traceback.print_exc()
            return -1

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

    def get_stop_on_time_tag(self):
        return self._stop_on_time_tag

    def set_stop_on_time_tag(self, stop_on_time_tag):
        self._stop_on_time_tag = stop_on_time_tag


class digital_rf_sink(gr.hier_block2):
    """Sink block for writing Digital RF data."""

    def __init__(
        self,
        top_level_dir,
        channels,
        dtype,
        subdir_cadence_secs,
        file_cadence_millisecs,
        sample_rate_numerator,
        sample_rate_denominator,
        start=None,
        ignore_tags=False,
        is_complex=True,
        num_subchannels=1,
        uuid_str=None,
        center_frequencies=None,
        metadata=None,
        is_continuous=True,
        compression_level=0,
        checksum=False,
        marching_periods=True,
        stop_on_skipped=False,
        stop_on_time_tag=False,
        debug=False,
        min_chunksize=None,
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

        dtype : np.dtype | object to be cast by np.dtype()
            Object that gives the numpy dtype of the data to be written. This
            value is passed into ``np.dtype`` to get the actual dtype
            (e.g. ``np.dtype('>i4')``). Scalar types, complex types, and
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

        sample_rate_numerator : int
            Numerator of sample rate in Hz.

        sample_rate_denominator : int
            Denominator of sample rate in Hz.


        Other Parameters
        ----------------
        start : None | int | float | string, optional
            A value giving the time/index of the channel's first sample. When
            `ignore_tags` is False, 'rx_time' tags will be used to identify
            data gaps and skip the sample index forward appropriately (tags
            that refer to an earlier time will be ignored).
            If None or '' and `ignore_tags` is False, drop data until an
            'rx_time' tag arrives and sets the start time (a ValueError is
            raised if `ignore_tags` is True).
            If an integer, it is interpreted as a sample index given in the
            number of samples since the epoch (time_since_epoch*sample_rate).
            If a float, it is interpreted as a UTC timestamp (seconds since
            epoch).
            If a string, three forms are permitted:
                1) a string which can be evaluated to an integer/float and
                    interpreted as above,
                2) a time in ISO8601 format, e.g. '2016-01-01T16:24:00Z'
                3) 'now' ('nowish'), indicating the current time (rounded up)

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

        stop_on_time_tag : bool, optional
            If True, stop writing when any but an initial 'rx_time' tag is received.

        debug : bool, optional
            If True, print debugging information.

        min_chunksize : None | int, optional
            Minimum number of samples to consume at once. This value can be
            used to adjust the sink's performance to reduce processing time.
            If None, a sensible default will be used.


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
        del options["self"]
        del options["top_level_dir"]
        del options["channels"]

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

        in_sig_dtypes = [list(sink.in_sig())[0] for sink in self._channels]
        in_sig = gr.io_signaturev(
            len(in_sig_dtypes), len(in_sig_dtypes), [s.itemsize for s in in_sig_dtypes]
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
        for ch in self._channels:
            ch.set_stop_on_skipped(stop_on_skipped)

    def get_stop_on_time_tag(self):
        return self._channels[0].get_stop_on_time_tag()

    def set_stop_on_time_tag(self, stop_on_time_tag):
        for ch in self._channels:
            ch.set_stop_on_time_tag(stop_on_time_tag)
