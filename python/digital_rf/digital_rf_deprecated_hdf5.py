# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""digital_rf_deprecated_hdf5.py is a module that allows python to read deprecated digital rf version 1 data.

It uses h5py to read old format.  Maintain to support upconversion to digital rf 2 only. No ability to write Drf 1.


$Id$
"""
from __future__ import absolute_import, division, print_function

import datetime
import functools
import glob
import os
import os.path
import time
import warnings

import h5py
import numpy

import six


class read_hdf5(object):
    """The class read_hdf5 is an object used to read digital rf v1 Hdf5 files as specified
    in the http://www.haystack.mit.edu/pipermail/rapid-dev/2014-February/000273.html email thread.

    This class allows random access to the rf data.

    """

    def __init__(self, top_level_directory_arg, load_all_metadata=False):
        """__init__ will verify the data in top_level_directory_arg is as expected.  It will analyze metadata
        to the degree specified in the load_all_metadata flag so that other methods can return more quickly

        Inputs:
            top_level_directory_arg - either a single top level, directory, or a list.  A directory can be a file system path or a url,
                where the url points to a top level directory.
            load_all_metadata - if True, loads all possible metadata at init.  If False, only loads high
                level metadata for faster __init__ speed.   A basic rule of thumb:
                    **** use load_all_metadata=False to make __init__ faster   ****
                    **** use load_all_metadata=True to make read_vector faster, at the cost of slower __init__ ****

        A top level directory must contain <channel_name>/<YYYY-MM-DDTHH-MM-SS/rf@<unix_seconds>.<%03i milliseconds>.h5

        If more than one top level directory contains the same channel_name subdirectory, this is considered the same channel.  An error
        is raised if their sample rates differ, or if their time periods overlap.

        This method will create the following attributes:

        self._top_level_dir_dict - a dictionary with keys = top_level_directory string, value = access mode (eg, 'local', 'file', or 'http')
            This attribute is static, that is, it is not updated when self.reload() called

        self._channel_dict - a dictionary with keys = channel_name, and value is a _channel_metadata object.

        self._load_all_metadata - True if full metadata search required by default, False if minimal metadata.

        self._last_update_has_full_metadata - True if last update got full metadata, False is last update got minimal
            metadata.  At init will equal self._load_all_metadata, but will be set to the load_all_metadata in reload
            when that method is called later.
        """

        # first, make top_level_directory_arg a list if a string
        if type(top_level_directory_arg) in six.string_types:
            top_level_arg = [top_level_directory_arg]
        else:
            top_level_arg = top_level_directory_arg

        # create static attribute self._top_level_dir_dict
        self._top_level_dir_dict = {}
        for top_level_directory in top_level_arg:
            if top_level_directory[0:7] == "file://":
                self._top_level_dir_dict[top_level_directory] = "file"
            elif top_level_directory[0:7] == "http://":
                self._top_level_dir_dict[top_level_directory] = "http"
            else:
                # make sure absolute path used
                if top_level_directory[0] != "/":
                    this_top_level_dir = os.path.join(os.getcwd(), top_level_directory)
                else:
                    this_top_level_dir = top_level_directory
                self._top_level_dir_dict[this_top_level_dir] = "local"

        self._channel_dict = {}

        self._load_all_metadata = load_all_metadata

        self.reload()

    def reload(self, load_all_metadata=None):
        """reload updates the attribute self._channel_dict.

            Inputs:
                load_all_metadata - If load_all_metadata is True, then get complete metadata.  If
                    load_all_metadata is False, only get high level metadata. Default is None, in which
                    case load_all_metadata=self._load_all_metadata, as set in init.
        """
        if load_all_metadata is None:
            load_all_metadata = self._load_all_metadata

        self._last_update_has_full_metadata = load_all_metadata

        # first update the channel list
        channel_dict = (
            {}
        )  # a temporary dict with key = channels, value = list of top level directories where found
        for top_level_dir in self._top_level_dir_dict.keys():
            channels_found = self._get_channels_in_dir(top_level_dir)
            for channel in channels_found:
                channel_name = os.path.basename(channel)
                if channel_name in channel_dict:
                    channel_dict[channel_name].append(top_level_dir)
                else:
                    channel_dict[channel_name] = [top_level_dir]

        # next throw away any metadata where the entire channel not longer exists
        remove_keys = []
        for channel_name in self._channel_dict:
            if channel_name not in channel_dict:
                # channel no longer exists
                remove_keys.append(channel_name)
        if len(remove_keys):
            for remove_key in remove_keys:
                del self._channel_dict[remove_key]

        # update all channels
        for channel_name in channel_dict.keys():
            if channel_name not in self._channel_dict:
                # a new channel is found - create it
                top_level_dir_metadata_list = []
                for top_level_dir in channel_dict[channel_name]:
                    new_top_level_metaddata = _top_level_dir_metadata(
                        top_level_dir,
                        channel_name,
                        self._top_level_dir_dict[top_level_dir],
                    )
                    top_level_dir_metadata_list.append(new_top_level_metaddata)
                top_level_dir_metadata_list.sort()
                new_channel_metadata = _channel_metadata(
                    channel_name, top_level_dir_meta_list=top_level_dir_metadata_list
                )
                new_channel_metadata.update(complete_update=load_all_metadata)
                self._channel_dict[channel_name] = new_channel_metadata

            else:
                # handle any changes to the top_level_dir list
                chan_obj = self._channel_dict[
                    channel_name
                ]  # just to shorten the following code
                found_dirs = (
                    []
                )  # a list of directories in _channel_metadata.top_level_dir_meta_list found
                for top_level_dir in channel_dict[channel_name]:
                    found = False
                    for chan_top_dir in chan_obj.top_level_dir_meta_list:
                        if top_level_dir == chan_top_dir.top_level_dir:
                            found_dirs.append(chan_top_dir.top_level_dir)
                            found = True
                            break
                    if not found:
                        # this is a new top level
                        new_top_level_meta = _top_level_dir_metadata(
                            top_level_dir,
                            channel_name,
                            self._top_level_dir_dict[top_level_dir],
                        )
                        chan_obj.add_top_level(new_top_level_meta)
                        found_dirs.append(top_level_dir)

                # make sure all top level dirs in chan metadata still exist
                for chan_top_dir in chan_obj.top_level_dir_meta_list:
                    if chan_top_dir.top_level_dir not in found_dirs:
                        # this top level dir no longer has data
                        chan_obj.remove_top_level_metadata(chan_top_dir.top_level_dir)

                chan_obj.update(complete_update=load_all_metadata)

    def get_channels(self):
        """get_channels returns a alphabetically sorted list of channels in this read_hdf5 object
        """
        channels = list(self._channel_dict.keys())
        channels.sort()
        return channels

    def get_bounds(self, channel_name):
        """get_bounds returns a tuple of (first_unix_sample, last_unix_sample) for a given channel name
        """
        channel_metadata = self._channel_dict[channel_name]
        return (
            int(channel_metadata.unix_start_sample),
            int(channel_metadata.unix_start_sample + channel_metadata.sample_extent),
        )

    def get_rf_file_metadata(self, channel_name):
        """get_rf_file_metadata returns a dictionary of metadata found as attributes in the Hdf5 file /rf_data
        dataset for the given channel name.
        """
        return self._channel_dict[channel_name].metadata_dict

    def get_metadata(self, channel_name, timestamp=None):
        """get_metadata returns a h5py.File object pointing to the metadata*.h5 file at the top level of the
        channel directory.  The user is responsible for closing that file when done.  If timestamp is None,
        the latest metadata*.h5 will be returned.  Otherwise it will open the earliest metadata file with timestamp
        greater than or equal to timestamp

        Use of this old metadata scheme is now strongly discouraged. Use get_rf_file_metadata for metadata within the
        Digital RF standard, or Digital Metdata for other metadata.
        """
        warnings.warn(
            "get_metadata is now deprecated.  Use get_rf_file_metadata for metadata within the Digital RF standard, or Digital Metdata for other metadata.",
            DeprecationWarning,
        )
        # first, get a sorted list of all metadata*.hf files
        metadata_file_list = []
        metadata_basename_list = []  # to make sure there are no repeated basenames
        channel = self._channel_dict[channel_name]
        for top_level_dir_obj in channel.top_level_dir_meta_list:
            metadata_files = glob.glob(
                os.path.join(
                    top_level_dir_obj.top_level_dir,
                    top_level_dir_obj.channel_name,
                    "metadata@*.h5",
                )
            )
            metadata_files.sort()
            for metadata_file in metadata_files:
                basename = os.path.basename(metadata_file)
                if basename in metadata_basename_list:
                    raise IOError(
                        "found repeated metadata file names in channel %s"
                        % (channel_name)
                    )
                # verify its a good file
                try:
                    f = h5py.File(metadata_file, "r")
                    f.close()
                except:
                    continue
                metadata_basename_list.append(basename)
                metadata_file_list.append(metadata_file)

        if len(metadata_file_list) == 0:
            raise IOError("No metadata files found in channel %s" % (channel_name))

        # open right metadata file
        if timestamp is None:
            return h5py.File(metadata_file_list[-1], "r")
        else:
            rightFile = None
            for metadata_file in metadata_file_list:
                basename = os.path.basename(metadata_file)
                this_timestamp = int(basename[len("metadata@") : basename.find(".")])
                if this_timestamp <= timestamp:
                    rightFile = metadata_file
                else:
                    break
            if rightFile is None:
                raise IOError(
                    "All metadata files found in channel %s after timestamp"
                    % (channel_name, timestamp)
                )
            return h5py.File(rightFile, "r")

    def get_continuous_blocks(self, start_unix_sample, stop_unix_sample, channel_name):
        """get_continuous_blocks returns a numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block.  Only samples between (start_unix_sample, stop_unix_sample)
        inclusive will be returned.

        Calls the private method _get_continuous_blocks.  If that raises a _MissingMetadata exception, calls
        reload with load_all_metadata == True to get missing metadata, and then retries _get_continuous_blocks.

        Returns IOError if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                inclusive will be returned.  Value of both are samples since 1970-01-01

            channel_name - channel to examine
        """
        try:
            return self._get_continuous_blocks(
                start_unix_sample, stop_unix_sample, channel_name
            )
        except _MissingMetadata:
            self.reload(True)
            # try again
            return self._get_continuous_blocks(
                start_unix_sample, stop_unix_sample, channel_name
            )

    def read_vector(self, unix_sample, vector_length, channel_name):
        """read_vector returns a numpy vector of complex8 type, no matter the dtype of the Hdf5 file
        or the number of channels. Shape is (vector_length, num_subchannels). Single value (real) files will
        have the imaginary part set to zero.

        Calls read_vector_raw, then converts result.

        Inputs:
            unix_sample - the number of samples since 1970-01-01 at start of data

            vector_length - the number of continuous samples to include

            channel_name - the channel name to use

        This method will raise an IOError error if the returned vector would include any missing data.
        It will also raise an IOError is any of the files needed to read the data have been deleted.
        This is possible because metadata on which this call is based might be out of date.
        """
        z = self.read_vector_raw(unix_sample, vector_length, channel_name)

        if z.dtype == numpy.complex64:
            return z
        elif z.dtype in (numpy.complex128, numpy.complex256):
            return numpy.array(z, dtype=numpy.complex64)

        if not hasattr(z.dtype, "names"):
            return numpy.array(z, dtype=numpy.complex64)
        elif z.dtype.names is None:
            return numpy.array(z, dtype=numpy.complex64)
        z = numpy.array(z["r"] + z["i"] * 1.0j, dtype=numpy.complex64)
        return z

    def read_vector_raw(self, unix_sample, vector_length, channel_name):
        """read_vector_raw returns a numpy array of dim(up to num_samples, num_subchannels) of the dtype in the Hdf5 files.

        If complex data, real and imag data will have names 'r' and 'i' if underlying data are integers
        or be numpy complex data type if underlying data floats.

        Inputs:
            unix_sample - the number of samples since 1970-01-01 at start of data

            vector_length - the number of continuous samples to include

            channel_name - the channel name to use

        This method will raise an IOError error if the returned vector would include any missing data.
        It will also raise an IOError is any of the files needed to read the data have been deleted.
        This is possible because metadata on which this call is based might be out of date.
        """
        if vector_length < 1:
            raise IOError(
                "Number of samples requested must be greater than 0, not %i"
                % (vector_length)
            )

        # make sure everything is a long
        unix_sample = int(unix_sample)
        vector_length = int(vector_length)

        channel_metadata = self._channel_dict[channel_name]

        ret_array = None
        first_unix_sample = None

        if self._last_update_has_full_metadata:
            # make sure we don't request beyond the metadata
            last_top_level_metadata = channel_metadata.top_level_dir_meta_list[-1]
            if (
                last_top_level_metadata.unix_start_sample
                + last_top_level_metadata.sample_extent
                < unix_sample + vector_length
            ):
                raise IOError(
                    "request in _read_vector beyond existing Metadata using full metadata"
                )

        for top_level_dir in channel_metadata.top_level_dir_meta_list:
            # note - even with partial metadata, the edges of top_level_dir are absolutely correct
            if (
                top_level_dir.unix_start_sample + top_level_dir.sample_extent
                < unix_sample
            ):
                # this top level dir is too early
                continue
            if unix_sample + vector_length < top_level_dir.unix_start_sample:
                # this top level dir is too late
                continue
            # run the faster version if self._last_update_has_full_metadata == True, version with searching if False
            this_array, this_unix_sample = top_level_dir.get_continuous_vector(
                max(unix_sample, top_level_dir.unix_start_sample),
                min(
                    unix_sample + vector_length,
                    top_level_dir.unix_start_sample + top_level_dir.sample_extent,
                ),
                self._last_update_has_full_metadata,
            )
            ret_array = self._combine_continuous_vectors(
                ret_array, this_array, first_unix_sample, this_unix_sample
            )
            if first_unix_sample is None:
                first_unix_sample = unix_sample

        if ret_array is None:
            raise IOError(
                "No data found for channel %s between %i and %i"
                % (channel_name, unix_sample, unix_sample + vector_length)
            )

        if len(ret_array) != vector_length:
            raise IOError(
                "Requested %i samples, but only found %i"
                % (vector_length, len(ret_array))
            )

        return ret_array

    def read_vector_c81d(self, unix_sample, vector_length, channel_name, subchannel=0):
        """read_vector_c81d returns a numpy vector of complex8 type, no matter the dtype of the Hdf5 file
        or the number of channels. Error thrown if subchannel doesn't exist.

        Inputs:
            unix_sample - the number of samples since 1970-01-01 at start of data

            vector_length - the number of continuous samples to include

            channel_name - the channel name to use

            subchannel - which subchannel to use.  Default is 0 (first)

        This method will raise an IOError error if the returned vector would include any missing data.
        It will also raise an IOError is any of the files needed to read the data have been deleted.
        This is possible because metadata on which this call is based might be out of date.
        """
        z = self.read_vector_raw(unix_sample, vector_length, channel_name)

        if z.shape[1] < subchannel + 1:
            raise ValueError(
                "Returned data has only %i subchannels, does not have subchannel %i"
                % (z.shape[1], subchannel)
            )

        if z.dtype == numpy.complex64:
            return z[:, subchannel]
        elif z.dtype in (numpy.complex128, numpy.complex256):
            return numpy.array(z[:, subchannel], dtype=numpy.complex64)

        slice = z[:, subchannel]
        if not hasattr(slice.dtype, "names"):
            raise ValueError("Single valued channels cannot be cast to complex")
        elif slice.dtype.names is None:
            raise ValueError("Single valued channels cannot be cast to complex")
        slice = numpy.array(slice["r"] + slice["i"] * 1.0j, dtype=numpy.complex64)
        return slice

    def _get_continuous_blocks(self, start_unix_sample, stop_unix_sample, channel_name):
        """_get_continuous_blocks is a private method that returns a numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block.  Only samples between (start_unix_sample, stop_unix_sample)
        inclusive will be returned.

        Raises _MissingMetadata exception if reload needs to be called.

        Returns IOError if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                inclusive will be returned.  Value of both are samples since 1970-01-01

            channel_name - channel to examine
        """
        channel_metadata = self._channel_dict[channel_name]

        ret_array = numpy.array([], dtype=numpy.uint64)
        for top_level_dir in channel_metadata.top_level_dir_meta_list:
            if (
                top_level_dir.unix_start_sample + top_level_dir.sample_extent
                < start_unix_sample
            ):
                # this top level dir is too early
                continue
            if stop_unix_sample < top_level_dir.unix_start_sample:
                # this top level dir is too late
                continue
            this_array = top_level_dir.get_continuous_blocks(
                max(start_unix_sample, top_level_dir.unix_start_sample),
                min(
                    stop_unix_sample,
                    top_level_dir.unix_start_sample + top_level_dir.sample_extent,
                ),
            )
            ret_array = self._combine_blocks(
                ret_array, this_array, top_level_dir.samples_per_file
            )

        if len(ret_array) == 0:
            raise IOError(
                "No data found for channel %s between %i and %i"
                % (channel_name, start_unix_sample, stop_unix_sample)
            )

        return ret_array

    def _combine_continuous_vectors(
        self, first_array, second_array, first_start_sample, second_start_sample
    ):
        """_combine_continuous_vectors returns the concatenation of first_array and second_array,  Raises error
        if two vectors are not continuous.

        Inputs:
            first_array - first array to combine.  If None, just return second_array
            second_array - second_array to merge at end of first
            first_start_sample - unix_sample of first sample in first_array.  None if first_array is None
            second_start_sample - unix_sample of first sample in second_array
        """

        if first_array is None:
            return second_array

        if len(first_array) != second_start_sample - first_start_sample:
            raise IOError(
                "_combine_continuous_vectors trying to combine two non-continuous vectors"
            )

        return numpy.concatenate((first_array, second_array))

    def _combine_blocks(self, first_array, second_array, samples_per_file):
        """_combine_blocks combines two numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block. The first row of the second array may or may not be contiguous
        with the last row of the first array.  If it is contiguous, that row will not be included, and the
        number of samples in that first row will instead be added to the last row of first_array. If not contiguous,
        the two arrays are simply concatenated
        """
        if len(first_array) == 0:
            return second_array
        is_contiguous = False
        if first_array[-1][0] + first_array[-1][1] > second_array[0][0]:
            raise IOError(
                "overlapping data found in top level directories %i %i"
                % (first_array[-1][0] + first_array[-1][1], second_array[0][0])
            )
        if first_array[-1][0] + first_array[-1][1] == second_array[0][0]:
            is_contiguous = True
        if is_contiguous:
            first_array[-1][1] += second_array[0][1]
            if len(second_array) == 1:
                return first_array
            else:
                return numpy.concatenate([first_array, second_array[1:]])
        else:
            return numpy.concatenate([first_array, second_array])

    def _get_channels_in_dir(self, top_level_dir):
        """_get_channels_in_dir returns a list of channel names found in top_level_dir

        Inputs:
            top_level_dir - string indicating top_level_dir
        """
        # define glob string for sub_directories in form YYYY-MM-DDTHH-MM-SS
        sub_directory_glob = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"

        retList = []
        access_mode = self._top_level_dir_dict[top_level_dir]
        # for now only local access
        if access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        if access_mode == "local":
            potential_channels = glob.glob(
                os.path.join(top_level_dir, "*", sub_directory_glob)
            )
            for potential_channel in potential_channels:
                channel_name = os.path.dirname(potential_channel)
                if channel_name not in retList:
                    retList.append(channel_name)

        return retList


class _channel_metadata(object):
    """The _channel_metadata is a private class to hold and access metadata about a particular digital_rf channel.
    A channel can extend over one of more top level directories.
    """

    def __init__(
        self,
        channel_name,
        unix_start_sample=0,
        sample_extent=0,
        top_level_dir_meta_list=[],
    ):
        """__init__ creates a new _channel_metadata object

        Inputs:
            channel_name - channel name (name of subdirectory defining this channel)
            unix_start_sample - unix start sample - first sample time in unix timeseconds * sample rate. If default
                0, then unknown
            sample_extent - number of samples between first and last in data
            top_level_dir_meta_list - a time ordered list of _top_level_dir_metadata objects.  Default is empty list

        Affects:
            create four input arguments as attributes.  Also creates self.metadata_dict, which is a dictionary of atributes
            found in the Hdf5 files (eg, sample_rate)

        """
        self.channel_name = channel_name
        self.unix_start_sample = int(unix_start_sample)
        self.sample_extent = int(sample_extent)
        self.top_level_dir_meta_list = top_level_dir_meta_list
        self.top_level_dir_meta_list.sort()
        self.metadata_dict = {}  # stores all metadata for this _channel_metadata

    def update(self, complete_update=False):
        """update will cause this _channel_metadata object to update itself.

        If complete_update == False, then only get high level metadata.  If complete_update,
        update all possible metadata.

        Inputs:
            complete_update - if True, then update all metadata.  If False,
                    the default, only update high level metadata.
        """
        for top_level_meta in self.top_level_dir_meta_list:
            top_level_meta.update(complete_update)
            if "uuid_str" not in self.metadata_dict:
                for key in top_level_meta.metadata_dict.keys():
                    self.metadata_dict[key] = top_level_meta.metadata_dict[key]
        self.reset_indices()

    def add_top_level(self, top_level_dir_meta):
        """add_top_level will add a new _top_level_dir_metadata object to self.top_level_dir_meta_list

        Inputs:
            top_level_dir_meta - new _top_level_dir_metadata object to add
        """
        self.top_level_dir_meta_list.append(top_level_dir_meta)
        self.top_level_dir_meta_list.sort()
        self.update()

    def remove_top_level_metadata(self, top_level_dir):
        """remove_top_level_metadata removes all metadata associated with this channel and one top_level_directory.  Raise
        ValueError is top_level_dir not found
        """
        remove_list = []
        for i, top_level_dir_meta in enumerate(self.top_level_dir_meta_list):
            if top_level_dir == top_level_dir_meta.top_level_dir:
                remove_list.append(i)

        if len(remove_list) == 0:
            raise ValueError("No directory %s found in this channel" % (top_level_dir))
        elif len(remove_list) > 1:
            raise ValueError(
                "More than one directory %s found in this channel" % (top_level_dir)
            )

        self.top_level_dir_meta_list.pop(remove_list[0])

        self.reset_indicies()

    def reset_indices(self):
        """reset_indices recalculates self.unix_start_sample and self.sample_extent based on self.top_level_dir_meta_list
        """
        if len(self.top_level_dir_meta_list) == 0:
            # set to unknown
            self.unix_start_sample = int(0)
            self.sample_extent = int(0)
            return

        self._verify_non_overlapping_data()
        self.unix_start_sample = int(self.top_level_dir_meta_list[0].unix_start_sample)
        self.sample_extent = int(
            self.top_level_dir_meta_list[-1].unix_start_sample
            + self.top_level_dir_meta_list[-1].sample_extent
            - self.unix_start_sample
        )

    def _verify_non_overlapping_data(self):
        """_verify_non_overlapping_data raises an error if any overlapping top level directories found
        """
        for i, record in enumerate(self.top_level_dir_meta_list):
            if i == 0:
                last_unix_start_sample = record.unix_start_sample
                last_sample_extent = record.sample_extent
            else:
                this_unix_start_sample = record.unix_start_sample
                this_sample_extent = record.sample_extent
                if last_unix_start_sample + last_sample_extent > this_unix_start_sample:
                    raise IOError(
                        "Overlapping samples found in top level dir %s"
                        % (record.top_level_dir)
                    )
                last_unix_start_sample = this_unix_start_sample
                last_sample_extent = this_sample_extent


@functools.total_ordering
class _top_level_dir_metadata(object):
    """The _top_level_dir_metadata is a private class to hold and access metadata about a particular digital_rf channel in
    a particular top level directory.
    """

    def __init__(
        self,
        top_level_dir,
        channel_name,
        access_mode,
        unix_start_sample=0,
        sample_extent=0,
        samples_per_file=0,
        sub_directory_recarray=None,
        sub_directory_dict=None,
    ):
        """__init__ creates a new _top_level_dir_metadata

        Inputs:
            top_level_dir - full path the top level directory that contains the parent channel_name
            channel_name - the channel_name subdirectory name
            access_mode - string giving access mode (eg, 'local', 'file', or 'http')
            unix_start_sample - unix start sample - first sample time in unix timeseconds * sample rate. If default
                0, then unknown
            sample_extent - number of samples between first and last in data. If default 0, then unknown
            samples_per_file - number of samples per file. If default 0, then unknown
            sub_directory_recarray - a ordered numpy recarray with one row describing summary information about a single
                sub_directory in that channel/top_level_dir named YYYY-MM-DDTHH-MM-SS with the following columns:
                1) 'subdirectory' - in form YYYY-MM-DDTHH-MM-SS,
                2) 'unix_start_sample' (for that subdirectory).  May be zero in no detailed metadata for this directory yet
                3) 'sample_extent' eg, number of samples to last sample in that subdirectory.  May be zero if no detailed
                    metadata yet for this subdirectory.
                4) 'file_count' number of Hdf5 data files in directory.  Will be zero if no detailed
                    metadata yet for this subdirectory.
                5) 'last_timestamp' UTC time stamp of latest data file in directory. .  Will be zero if no detailed
                    metadata yet for this subdirectory
                Order is by subdirectory and/or unix_start_sample
            sub_directory_dict - a dictionary with key = sub_directory, value = _sub_directory_metadata object

        Affects: creates an attribute for each input argument

        Also creates cached attributes to speed reads with sparse metadata:
            self._last_file - open H5py file last read
            self._last_start_sample - sample start index of cached file
            This file must be gap free, or it is never cached

        """
        self.top_level_dir = top_level_dir
        self.channel_name = channel_name
        self.access_mode = access_mode
        self.unix_start_sample = int(unix_start_sample)
        self.sample_extent = int(sample_extent)
        self.samples_per_file = int(samples_per_file)
        self.sub_directory_recarray = sub_directory_recarray
        self.sub_directory_dict = sub_directory_dict
        self.metadata_dict = {}  # to be populated by rf file metadata

        # data type of sub_directory_array
        self.data_t = numpy.dtype(
            [
                ("subdirectory", numpy.str_, 512),
                ("unix_start_sample", numpy.uint64, 1),
                ("sample_extent", numpy.uint64, 1),
                ("file_count", numpy.uint64, 1),
                ("last_timestamp", numpy.double, 1),
            ]
        )

        if self.sub_directory_recarray is None:
            # create empty array
            self.sub_directory_recarray = numpy.array([], dtype=self.data_t)

        # define glob strings for sub_directories in form YYYY-MM-DDTHH-MM-SS and rf files in form rf@*.*.h5
        self._sub_directory_glob = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        self._rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"

        # attributes to allow caching
        self._last_file = None
        self._last_start_sample = None

    def update(self, complete_update=False):
        """update will cause this _top_level_dir_metadata object to update itself.

        If complete_update == False, then only get high level metadata.
        If complete_update, update all possible metadata.

        Inputs:
            complete_update - if True, then update all metadata, not matter the other arguments.  If False,
                    the default, limited update according to the other arguments.
        """
        if complete_update:
            self._full_update()
        else:
            self._high_level_reload()

    def get_continuous_blocks(self, start_unix_sample, stop_unix_sample):
        """get_continuous_blocks returns a numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block.  Only samples between (start_unix_sample, stop_unix_sample)
        inclusive will be returned.

        Returns IOError if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                inclusive will be returned.
        """
        # to improve speed, do searchsorted to get first index to look into
        first_index = numpy.searchsorted(
            self.sub_directory_recarray["unix_start_sample"],
            numpy.array([start_unix_sample]),
        )
        first_index = first_index[0]
        if first_index > 0:
            first_index -= 1
        ret_array = numpy.array([], dtype=numpy.uint64)
        for i in range(first_index, len(self.sub_directory_recarray)):
            this_start_sample = int(self.sub_directory_recarray["unix_start_sample"][i])
            if this_start_sample > stop_unix_sample:
                break
            if stop_unix_sample < this_start_sample:
                continue
            this_extent = int(self.sub_directory_recarray["sample_extent"][i])
            if this_extent == 0:
                raise _MissingMetadata("this_extent == 0")

            # now check that subdirectories with metadata are still up to date
            base_subdirectory = self.sub_directory_recarray["subdirectory"][i]
            file_count, last_timestamp = self._get_subdirectory_file_info(
                base_subdirectory
            )
            self.sub_directory_dict[base_subdirectory].update_if_needed(
                file_count, last_timestamp
            )

            sub_dir_metadata = self.sub_directory_dict[
                self.sub_directory_recarray["subdirectory"][i]
            ]
            this_array = sub_dir_metadata.get_continuous_blocks(
                max(start_unix_sample, this_start_sample),
                min(stop_unix_sample, (this_start_sample + this_extent) - 1),
            )
            ret_array = self._combine_blocks(
                ret_array, this_array, self.samples_per_file
            )

        return ret_array

    def get_continuous_vector(
        self, start_unix_sample, stop_unix_sample, last_update_has_full_metadata=False
    ):
        """get_continuous_vector returns a tuple of (numpy array of data, first unix_sample in returned data)
        Only samples between (start_unix_sample, stop_unix_sample) (excludes stop_unix_sample) will be returned.

        Returns (None, None) if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                (excludes stop_unix_sample) will be returned.

            last_update_has_full_metadata - if True, use standard metadata to access data.  If False (the default),
                use glob to search for right file without using detailed metadata (slower performance, but less metadata
                discovery time).
        """
        if last_update_has_full_metadata:
            # to improve speed, do searchsorted to get first index to look into
            first_index = numpy.searchsorted(
                self.sub_directory_recarray["unix_start_sample"],
                numpy.array([start_unix_sample]),
            )
            first_index = first_index[0]
            if first_index > 0:
                first_index -= 1
            ret_array = None
            first_unix_sample = None
            for i in range(first_index, len(self.sub_directory_recarray)):
                this_start_sample = int(
                    self.sub_directory_recarray["unix_start_sample"][i]
                )
                if this_start_sample >= stop_unix_sample:
                    break
                if stop_unix_sample < this_start_sample:
                    continue
                this_extent = int(self.sub_directory_recarray["sample_extent"][i])
                if this_extent == 0:
                    raise IOError("Metadata not found in recarray")
                if this_start_sample + this_extent <= start_unix_sample:
                    continue

                sub_dir_metadata = self.sub_directory_dict[
                    self.sub_directory_recarray["subdirectory"][i]
                ]
                this_array, unix_sample = sub_dir_metadata.get_continuous_vector(
                    max(start_unix_sample, this_start_sample),
                    min(stop_unix_sample, this_start_sample + this_extent),
                )

                ret_array = self._combine_continuous_vectors(
                    ret_array, this_array, first_unix_sample, unix_sample
                )
                if first_unix_sample is None:
                    first_unix_sample = unix_sample

            return (ret_array, first_unix_sample)

        else:
            # check if we can use cached file
            if self._last_start_sample:
                if (
                    start_unix_sample > self._last_start_sample
                    and stop_unix_sample
                    <= self._last_start_sample + self.samples_per_file
                ):
                    return self._get_data_from_cache(
                        start_unix_sample, stop_unix_sample
                    )
                else:
                    # cache has expired
                    try:
                        self._last_file.close()
                    except:
                        pass
                    self._last_file = None
                    self._last_start_sample = None

            # only partial metadata
            files_to_search = self._get_files_to_search(
                start_unix_sample, stop_unix_sample
            )
            ret_array = None
            for file_to_search in files_to_search:
                arr = self._read_data_from_file(
                    file_to_search, start_unix_sample, stop_unix_sample, ret_array
                )
                if (not arr is None) and ret_array is None:
                    ret_array = arr
                elif arr is None:
                    continue
                else:
                    ret_array = numpy.concatenate((ret_array, arr))
                if not ret_array is None:
                    if len(ret_array) == stop_unix_sample - start_unix_sample:
                        break

            # verify success
            if ret_array is None:
                raise IOError("No valid data found")
            if len(ret_array) != stop_unix_sample - start_unix_sample:
                raise IOError(
                    "Wanted len %i, but got len %i"
                    % (stop_unix_sample - start_unix_sample, len(ret_array))
                )

            return (ret_array, start_unix_sample)

    def _high_level_reload(self):
        """_high_level_reload updates only high level metadata.  Basically this is only the first and last sample
        """
        base_subdirectory_list = self._get_subdirectories()
        dt1970 = datetime.datetime(1970, 1, 1)

        # first pass is to remove any subdirectories that have disappeared
        rows_to_delete_arr = []
        for i, subdirectory in enumerate(self.sub_directory_recarray["subdirectory"]):
            if subdirectory not in base_subdirectory_list:
                rows_to_delete_arr.append(i)
                del self.sub_directory_dict[subdirectory]
        if len(rows_to_delete_arr) > 0:
            rows_to_delete_arr = numpy.array(rows_to_delete_arr, numpy.int64)
            self.sub_directory_recarray = numpy.delete(
                self.sub_directory_recarray, rows_to_delete_arr
            )

        # next pass creates any new rows in _sub_directory_metadata
        for base_subdirectory in base_subdirectory_list:
            try:
                file_count, last_timestamp = self._get_subdirectory_file_info(
                    base_subdirectory
                )
            except IOError:
                new_sub_dir_meta = _sub_directory_metadata(
                    self.top_level_dir,
                    self.channel_name,
                    self.access_mode,
                    base_subdirectory,
                )
                if not self.sub_directory_dict is None:
                    self.sub_directory_dict[base_subdirectory] = new_sub_dir_meta
                else:
                    self.sub_directory_dict = {base_subdirectory: new_sub_dir_meta}
                # extend self.sub_directory_recarray by one
                self.sub_directory_recarray.resize(len(self.sub_directory_recarray) + 1)

                # get estimate of first sample without IO
                subDirDT = datetime.datetime.strptime(
                    os.path.basename(base_subdirectory), "%Y-%m-%dT%H-%M-%S"
                )
                total_secs = (subDirDT - dt1970).total_seconds()
                if len(list(self.metadata_dict.keys())) == 0:
                    # force it to be created
                    new_sub_dir_meta.get_last_sample()
                    self.metadata_dict = new_sub_dir_meta.metadata_dict
                    self.samples_per_file = self.metadata_dict["samples_per_file"]
                sample_index = total_secs * self.metadata_dict["sample_rate"]
                self.sub_directory_recarray[-1] = (
                    base_subdirectory,
                    sample_index,
                    0,
                    0,
                    0,
                )  # use default values

        # update first and last only
        self.sub_directory_dict[base_subdirectory_list[0]].update()
        if len(base_subdirectory_list) > 1:
            self.sub_directory_dict[base_subdirectory_list[-1]].update()

        self.unix_start_sample = int(
            self.sub_directory_dict[base_subdirectory_list[0]].get_first_sample()
        )
        last_sample = int(
            self.sub_directory_dict[base_subdirectory_list[-1]].get_last_sample()
        )
        self.sample_extent = int(last_sample - self.unix_start_sample)

    def _full_update(self):
        """_full_update will cause this _top_level_dir_metadata object to update all possible metadata
        """
        update_needed = False  # will be set to True if any subdirectory updated
        base_subdirectory_list = self._get_subdirectories(verify_files=True)

        # first pass is to remove any subdirectories that have disappeared
        rows_to_delete_arr = []
        for i, subdirectory in enumerate(self.sub_directory_recarray["subdirectory"]):
            if subdirectory not in base_subdirectory_list:
                rows_to_delete_arr.append(i)
                del self.sub_directory_dict[subdirectory]
        if len(rows_to_delete_arr) > 0:
            rows_to_delete_arr = numpy.array(rows_to_delete_arr, numpy.int64)
            self.sub_directory_recarray = numpy.delete(
                self.sub_directory_recarray, rows_to_delete_arr
            )

        # next pass
        for i, base_subdirectory in enumerate(base_subdirectory_list):
            try:
                file_count, last_timestamp = self._get_subdirectory_file_info(
                    base_subdirectory
                )
                # call update_if_needed to make faster
                if self.sub_directory_dict[base_subdirectory].update_if_needed(
                    file_count, last_timestamp
                ):
                    (
                        first_unix_sample,
                        sample_extent,
                        file_count,
                        samples_per_file,
                        last_timestamp,
                    ) = self.sub_directory_dict[
                        base_subdirectory
                    ].get_summary_metadata()
                    self.sub_directory_recarray[i] = (
                        base_subdirectory,
                        first_unix_sample,
                        sample_extent,
                        file_count,
                        last_timestamp,
                    )
                    update_needed = True
            except IOError:
                update_needed = True
                new_sub_dir_meta = _sub_directory_metadata(
                    self.top_level_dir,
                    self.channel_name,
                    self.access_mode,
                    base_subdirectory,
                )
                new_sub_dir_meta.update()
                if len(list(self.metadata_dict.keys())) == 0:
                    self.metadata_dict = new_sub_dir_meta.metadata_dict
                if not self.sub_directory_dict is None:
                    self.sub_directory_dict[base_subdirectory] = new_sub_dir_meta
                else:
                    self.sub_directory_dict = {base_subdirectory: new_sub_dir_meta}
                (
                    first_unix_sample,
                    sample_extent,
                    file_count,
                    samples_per_file,
                    last_timestamp,
                ) = new_sub_dir_meta.get_summary_metadata()
                # extend self.sub_directory_recarray by one
                self.sub_directory_recarray.resize(len(self.sub_directory_recarray) + 1)
                self.sub_directory_recarray[-1] = (
                    base_subdirectory,
                    first_unix_sample,
                    sample_extent,
                    file_count,
                    last_timestamp,
                )

                # handle samples_per_file
                if self.samples_per_file == 0:
                    self.samples_per_file = int(samples_per_file)
                elif self.samples_per_file != int(samples_per_file):
                    raise IOError(
                        "Samples per file changed from %i to %i with subdirectory %s"
                        % (self.samples_per_file, samples_per_file, base_subdirectory)
                    )
                continue

        if update_needed:
            self._verify_non_overlapping_data()
            # update summary metadata
            self.unix_start_sample = int(
                self.sub_directory_recarray["unix_start_sample"][0]
            )
            last_sample = int(
                self.sub_directory_recarray["unix_start_sample"][-1]
            ) + int(self.sub_directory_recarray["sample_extent"][-1])
            self.sample_extent = int(last_sample - self.unix_start_sample)

    def _verify_non_overlapping_data(self):
        """_verify_non_overlapping_data raises an error if any overlapping subdirectories found
        """
        for i, record in enumerate(self.sub_directory_recarray):
            if i == 0:
                last_unix_start_sample = record["unix_start_sample"]
                last_sample_extent = record["sample_extent"]
            else:
                this_unix_start_sample = record["unix_start_sample"]
                this_sample_extent = record["sample_extent"]
                if last_unix_start_sample + last_sample_extent > this_unix_start_sample:
                    if this_sample_extent != 0:  # otherwise not trustworthy
                        raise IOError(
                            "Overlapping samples found in subdirectory %s - last_unix_start_sample=%i, last_sample_extent=%i, this_unix_start_sample=%i"
                            % (
                                record["subdirectory"],
                                last_unix_start_sample,
                                last_sample_extent,
                                this_unix_start_sample,
                            )
                        )
                last_unix_start_sample = this_unix_start_sample
                last_sample_extent = this_sample_extent

    def _get_subdirectory_file_info(self, subdirectory):
        """_get_subdirectory_file_info returns a tuple ot (num_files, last_timestamp) for a given
        subdirectory using the self.sub_directory_recarray recarray.  Raises IOError if subdirectory
        not found in recarray.
        """
        result = numpy.argwhere(
            self.sub_directory_recarray["subdirectory"] == subdirectory
        )
        if len(result) == 0:
            raise IOError("subdirectory %s not found" % (subdirectory))
        if len(result) > 1:
            raise ValueError("got unexpected result %s" % (str(result)))
        return (
            self.sub_directory_recarray["file_count"][result[0][0]],
            self.sub_directory_recarray["last_timestamp"][result[0][0]],
        )

    def _combine_continuous_vectors(
        self, first_array, second_array, first_start_sample, second_start_sample
    ):
        """_combine_continuous_vectors returns the concatenation of first_array and second_array,  Raises error
        if two vectors are not continuous.

        Inputs:
            first_array - first array to combine.  If None, just return second_array
            second_array - second_array to merge at end of first
            first_start_sample - unix_sample of first sample in first_array.  None if first_array is None
            second_start_sample - unix_sample of first sample in second_array
        """

        if first_array is None:
            return second_array

        if len(first_array) != second_start_sample - first_start_sample:
            raise IOError(
                "_combine_continuous_vectors trying to combine two non-continuous vectors"
            )

        return numpy.concatenate((first_array, second_array))

    def _get_subdirectories(self, verify_files=False):
        """_get_subdirectories returns a sorted list of base subdirectory names

        Inputs:
            verify_files - If True, only return subdirectories with h5 files.  If False (the default),
            return any subdirectory that matches the format, independent of whether it has files
        """
        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))
        subdirectory_list = glob.glob(
            os.path.join(
                self.top_level_dir, self.channel_name, self._sub_directory_glob
            )
        )
        subdirectory_list.sort()
        if not verify_files:
            return subdirectory_list
        retList = []  # only return those with files
        for subdirectory in subdirectory_list:
            if len(glob.glob(os.path.join(subdirectory, "*.h5"))) > 0:
                retList.append(subdirectory)
        return retList

    def _combine_blocks(self, first_array, second_array, samples_per_file):
        """_combine_blocks combines two numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block. The first row of the second array may or may not be contiguous
        with the last row of the first array.  If it is contiguous, that row will not be included, and the
        number of samples in that first row will instead be added to the last row of first_array. If not contiguous,
        the two arrays are simply concatenated
        """
        if len(first_array) == 0:
            return second_array
        is_contiguous = False
        if first_array[-1][0] + first_array[-1][1] > second_array[0][0]:
            raise IOError(
                "overlapping data found in top level directories %i %i"
                % (first_array[-1][0] + first_array[-1][1], second_array[0][0])
            )
        if first_array[-1][0] + first_array[-1][1] == second_array[0][0]:
            is_contiguous = True
        if is_contiguous:
            first_array[-1][1] += second_array[0][1]
            if len(second_array) == 1:
                return first_array
            else:
                return numpy.concatenate([first_array, second_array[1:]])
        else:
            return numpy.concatenate([first_array, second_array])

    def _get_files_to_search(self, start_unix_sample, stop_unix_sample):
        """_get_files_to_search is a private method designed to return a subset of data files that might contain
        data to return.  Used only when complete metadata not available.

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                (excludes stop_unix_sample) will be returned, so only return files that might contain that range
        """
        seconds_per_file = 1 + int(
            self.metadata_dict["samples_per_file"][0]
            // self.metadata_dict["sample_rate"][0]
        )
        start_integer_sec = int(
            start_unix_sample // self.metadata_dict["sample_rate"][0]
        )
        stop_integer_sec = int(stop_unix_sample // self.metadata_dict["sample_rate"][0])
        seconds_list = range(start_integer_sec - seconds_per_file, stop_integer_sec + 1)

        # now glob for these files in all directories until no more found
        files_to_search = []
        first_index = numpy.searchsorted(
            self.sub_directory_recarray["unix_start_sample"],
            numpy.array([start_unix_sample]),
        )
        first_index = first_index[0]
        if first_index == len(self.sub_directory_recarray["unix_start_sample"]):
            first_index -= 1
        first_index_sec = int(
            self.sub_directory_recarray["unix_start_sample"][first_index]
            // self.metadata_dict["sample_rate"][0]
        )
        if first_index > 0 and start_integer_sec - seconds_per_file < first_index_sec:
            first_index -= 1
        for i in range(first_index, len(self.sub_directory_recarray)):
            files_this_subdir = 0  # one method to determine when to break
            this_dir_start_sec = int(
                self.sub_directory_recarray["unix_start_sample"][i]
                // self.metadata_dict["sample_rate"][0]
            )
            if this_dir_start_sec > stop_integer_sec + seconds_per_file:
                # another break test
                break
            for second in seconds_list:
                glob_str = os.path.join(
                    self.sub_directory_recarray["subdirectory"][i],
                    "rf@%i.???.h5" % (second),
                )
                new_files = glob.glob(glob_str)
                files_this_subdir += len(new_files)
                files_to_search += new_files
            # break if none found after second subdirectory
            if i > first_index and files_this_subdir == 0:
                break

        files_to_search.sort()
        return files_to_search

    def _read_data_from_file(
        self, file_to_search, start_unix_sample, stop_unix_sample, ret_array
    ):
        """_read_data_from_file reads data (if any) from file.  Used with minimal metadata
        """
        # make sure cache is clear if this called
        if self._last_start_sample:
            try:
                self._last_file.close()
            except:
                pass
            self._last_file = None
            self._last_start_sample = None

        samples_per_file = int(self.metadata_dict["samples_per_file"][0])
        f = h5py.File(file_to_search, "r")
        rf_data_index = f["/rf_data_index"]

        if ret_array is None:
            # see if this is the first file with data
            file_start_index = None
            # walk through rf_data_index until right index found (if any)
            for i in range(len(rf_data_index)):
                this_file_index = int(rf_data_index[i, 1])
                this_sample_index = int(rf_data_index[i, 0])
                if i < len(rf_data_index) - 1:
                    samples_left = int(rf_data_index[i + 1, 1]) - this_file_index
                else:
                    samples_left = int(samples_per_file - this_file_index)
                if this_sample_index + samples_left > start_unix_sample:
                    # the starting read point for the first file was found
                    file_start_index = int(
                        this_file_index + (start_unix_sample - this_sample_index)
                    )
                    # make sure there are no gaps in this file to be read
                    if i < len(rf_data_index) - 1:
                        samples_left_to_read = min(
                            int(rf_data_index[i + 1, 1]) - file_start_index,
                            stop_unix_sample - start_unix_sample,
                        )
                        if samples_left_to_read < stop_unix_sample - start_unix_sample:
                            f.close()
                            raise IOError(
                                "Gap found in first file %s read" % (file_to_search)
                            )
                    else:
                        samples_left_to_read = min(
                            samples_per_file - file_start_index,
                            stop_unix_sample - start_unix_sample,
                        )
                    rf_data = f["/rf_data"][
                        file_start_index : file_start_index + samples_left_to_read
                    ]
                    # see if we can cache this file
                    if len(rf_data_index) == 1:
                        self._last_start_sample = this_sample_index
                        if not self._last_file is None:
                            try:
                                self._last_file.close()
                            except:
                                pass
                        self._last_file = f
                    else:
                        f.close()
                    return rf_data

            # no data found
            try:
                f.close()
            except:
                pass
            return None

        else:
            # make sure cache is clear if muliple file read
            if self._last_start_sample:
                try:
                    self._last_file.close()
                except:
                    pass
                self._last_file = None
                self._last_start_sample = None

            # append all needed data from this file
            # first, verify this file begins where we expect
            first_file_sample = rf_data_index[0, 0]
            if first_file_sample != start_unix_sample + len(ret_array):
                f.close()
                raise IOError(
                    "gap found at file %s -expected index %i, got %i"
                    % (
                        file_to_search,
                        start_unix_sample + len(ret_array),
                        first_file_sample,
                    )
                )
            # verify no gaps over this read
            if len(rf_data_index) > 1:
                samples_in_this_file = rf_data_index[1, 1] - rf_data_index[0, 1]
                if samples_in_this_file < (stop_unix_sample - start_unix_sample) - len(
                    ret_array
                ):
                    f.close()
                    raise IOError(
                        "not enough samples in file %s before data gap"
                        % (file_to_search)
                    )

            samples_to_read = min(
                samples_per_file,
                (stop_unix_sample - start_unix_sample) - len(ret_array),
            )
            rf_data = f["/rf_data"][0:samples_to_read]
            f.close()
            return rf_data

    def _get_data_from_cache(self, start_unix_sample, stop_unix_sample):
        """_get_data_from_cache simple returns the desired data from the cached Hdf5 file

        Inputs: start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                (excludes stop_unix_sample) will be returned.

        Calling method tested that this read is possible entirely within this file
        """
        start_index = start_unix_sample - self._last_start_sample
        samples_to_read = stop_unix_sample - start_unix_sample
        return (
            self._last_file["/rf_data"][start_index : start_index + samples_to_read],
            start_unix_sample,
        )

    def __eq__(self, other):
        """__eq__ compares two _top_level_dir_metadata objects for equality
        """
        # only the same channel can be compared
        if self.channel_name != other.channel_name:
            raise ValueError(
                "Cannot compare mismatched channel names %s and %s"
                % (self.channel_name, other.channel_name)
            )

        if self.unix_start_sample != 0 and other.unix_start_sample != 0:
            return self.unix_start_sample == other.unix_start_sample

        # use subdirectory names instead
        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        first_subdirectory_list = glob.glob(
            os.path.join(
                self.top_level_dir, self.channel_name, self._sub_directory_glob
            )
        )
        first_subdirectory_list.sort()
        if len(first_subdirectory_list) == 0:
            raise ValueError(
                "Cannot compare top level directory because it has no data"
                % (self.top_level_dir)
            )
        first_subdirectory = os.path.basename(first_subdirectory_list[0])

        second_subdirectory_list = glob.glob(
            os.path.join(
                other.top_level_dir, other.channel_name, self._sub_directory_glob
            )
        )
        second_subdirectory_list.sort()
        if len(second_subdirectory_list) == 0:
            raise ValueError(
                "Cannot compare top level directory because it has no data"
                % (other.top_level_dir)
            )
        second_subdirectory = os.path.basename(second_subdirectory_list[0])

        return first_subdirectory == second_subdirectory

    def __ne__(self, other):
        """__ne__ compares two _top_level_dir_metadata objects for inequality
        """
        return not (self == other)

    def __lt__(self, other):
        """__lt__ compares two _top_level_dir_metadata objects
        """
        # only the same channel can be compared
        if self.channel_name != other.channel_name:
            raise ValueError(
                "Cannot compare mismatched channel names %s and %s"
                % (self.channel_name, other.channel_name)
            )

        if self.unix_start_sample != 0 and other.unix_start_sample != 0:
            return self.unix_start_sample < other.unix_start_sample

        # use subdirectory names instead
        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        first_subdirectory_list = glob.glob(
            os.path.join(
                self.top_level_dir, self.channel_name, self._sub_directory_glob
            )
        )
        first_subdirectory_list.sort()
        if len(first_subdirectory_list) == 0:
            raise ValueError(
                "Cannot compare top level directory because it has no data"
                % (self.top_level_dir)
            )
        first_subdirectory = os.path.basename(first_subdirectory_list[0])

        second_subdirectory_list = glob.glob(
            os.path.join(
                other.top_level_dir, other.channel_name, self._sub_directory_glob
            )
        )
        second_subdirectory_list.sort()
        if len(second_subdirectory_list) == 0:
            raise ValueError(
                "Cannot compare top level directory because it has no data"
                % (other.top_level_dir)
            )
        second_subdirectory = os.path.basename(second_subdirectory_list[0])

        return first_subdirectory < second_subdirectory

    def __del__(self):
        """__del__makes sure self._last_file is closed.  Does not happen automatically.
        """
        if not self._last_file is None:
            try:
                self._last_file.close()
            except:
                pass


class _sub_directory_metadata(object):
    """The _sub_directory_metadata is a private class to hold and access metadata about a particular digital_rf channel in
    a particular subdirectory.
    """

    def __init__(self, top_level_dir, channel_name, access_mode, subdirectory):
        """__init__ creates a new _sub_directory_metadata object

        Inputs:
            top_level_dir - full path the top level directory that contains the parent channel_name
            channel_name - the channel_name subdirectory name
            access_mode - string giving access mode (eg, 'local', 'file', or 'http')
            subdirectory - subdirectory name in form YYYY-MM-DDTHH-MM-SS

        Affects:
            Sets self.metadata to None.  When update called, self.metadata will be set to a numpy.recarray
            with columns:
                1. unix_sample_index - number of samples since 1970-01-01 to the start of a contiguous data block (uint64_t)
                2. file_index - where in the file this contiguous block of data begins
                3. rf_basename (25 char string)

            Also sets self.cont_metadata to None.  When update called, self.cont_metadata will be set
            to a numpy.recarray about block of contiguous data with columns:
                1. unix_sample_index - number of samples since 1970-01-01 to the start of a contiguous data block (uint64_t)
                2. sample_extent - number of continuous samples

            Also sets self.samples_per_file and self.file_count and self.last_timestamp to None.  Will be set with
            first call to update

            Also creates cached attributes to speed reads with detailed metadata:
            self._last_file - open H5py file last read
            self._last_start_sample - sample start index of cached file
            This file must be gap free, or it is never cached

        """
        self.top_level_dir = top_level_dir
        self.channel_name = channel_name
        self.access_mode = access_mode
        self.subdirectory = subdirectory
        self.metadata = None
        self.cont_metadata = None
        self.samples_per_file = None
        self.file_count = None
        self.last_timestamp = None  # timestamp of last file in UTC
        self.metadata_dict = {}  # to be populated by rf file metadata

        self._rf_file_glob = "rf@[0-9]*.[0-9][0-9][0-9].h5"

        # data type of sub_directory_array
        self.data_t = numpy.dtype(
            [
                ("unix_sample_index", numpy.uint64, 1),
                ("file_index", numpy.uint64, 1),
                ("rf_basename", numpy.str_, 25),
            ]
        )
        self.cont_data_t = numpy.dtype(
            [("unix_sample_index", numpy.uint64, 1), ("sample_extent", numpy.uint64, 1)]
        )

        # set to an empty recarray
        if self.metadata is None:
            self.metadata = numpy.array([], dtype=self.data_t)
        if self.cont_metadata is None:
            self.cont_metadata = numpy.array([], dtype=self.cont_data_t)

        # attributes to allow caching
        self._last_file = None
        self._last_start_sample = None

    def get_summary_metadata(self):
        """get_summary_metadata returns a tuple of (first_unix_sample, sample_extent, file_count, samples_per_file,
        last_timestamp) for this _sub_directory_metadata object.

        Raises IOError if no self.metadata
        """
        if len(self.metadata) == 0:
            raise IOError(
                "Must call update before calling get_summary_metadata, or subdirectory %s empty"
                % (self.subdirectory)
            )

        first_unix_sample = int(self.metadata["unix_sample_index"][0])
        last_unix_sample = int(self.metadata["unix_sample_index"][-1]) + (
            (self.samples_per_file - int(self.metadata["file_index"][-1])) - 1
        )
        return (
            first_unix_sample,
            1 + (last_unix_sample - first_unix_sample),
            self.file_count,
            self.samples_per_file,
            self.last_timestamp,
        )

    def update_if_needed(self, file_count, last_timestamp):
        """update_if_needed calls update only if input file_count or last_timestamp indicate an update
        is needed. Returns True if update actually called, False otherwise

        Inputs:
            file_count - number of file in subdirectory when last checked
            last_timestamp - UTC timestamp of last file in subdirectory when last checked
        """

        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        rf_file_list = glob.glob(
            os.path.join(
                self.top_level_dir,
                self.channel_name,
                self.subdirectory,
                self._rf_file_glob,
            )
        )
        if len(rf_file_list) == 0:
            raise IOError("subdirectory %s empty" % (self.subdirectory))

        rf_file_list.sort()
        if len(rf_file_list) != file_count:
            self.update()
            return True
        elif abs(self._get_utc_timestamp(rf_file_list[-1]) - last_timestamp) > 2.0:
            # leave margin for error
            self.update()
            return True

        return False

    def update(self):
        """update updates self.metadata.  If it was a file name, it reads that data into memory, and then updates it
        """
        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        rf_file_list = glob.glob(
            os.path.join(
                self.top_level_dir,
                self.channel_name,
                self.subdirectory,
                self._rf_file_glob,
            )
        )
        rf_file_list.sort()
        rf_file_basename_list = [os.path.basename(rf_file) for rf_file in rf_file_list]

        # first check to see if we can update things quickly if the data is continuous
        if self._update_continuous_data(rf_file_basename_list, rf_file_list):
            return

        unique_rf_basenames = numpy.unique(self.metadata["rf_basename"])

        # first step is to delete all lines where the rf file has been deleted
        rows_to_delete_arr = numpy.array([], dtype=numpy.int64)
        for rf_basename in unique_rf_basenames:
            if rf_basename not in rf_file_basename_list:
                # get a list of all rows with that file
                result = numpy.argwhere(self.metadata["rf_basename"] == rf_basename)
                result = result.flatten()
                rows_to_delete_arr = numpy.concatenate((rows_to_delete_arr, result))
        if len(rows_to_delete_arr) > 0:
            self.metadata = numpy.delete(self.metadata, rows_to_delete_arr)
            unique_rf_basenames = numpy.unique(self.metadata["rf_basename"])

        # the next step is to add rows from each file where it does not yet exist in self.metadata
        if len(self.metadata["rf_basename"]) > 0:
            first_file_index = (
                rf_file_basename_list.index(self.metadata["rf_basename"][-1]) + 1
            )
        else:
            first_file_index = 0
        # we are only looping over files not already in self.metadata, and all data will be appended
        for i, rf_file_basename in enumerate(rf_file_basename_list[first_file_index:]):
            # verify the last file is not still being written
            if rf_file_basename == rf_file_basename_list[-1]:
                if self._file_is_open(rf_file_list[-1]):
                    self.file_count = (
                        len(rf_file_basename_list) - 1
                    )  # last file not counted
                    if self.file_count > 0:
                        self.last_timestamp = self._get_utc_timestamp(rf_file_list[-2])
                    continue
                else:
                    self.file_count = len(rf_file_basename_list)
                    self.last_timestamp = self._get_utc_timestamp(rf_file_list[-1])
            added_rows = self._get_new_rows(rf_file_basename)
            if added_rows is None:
                continue
            if len(list(self.metadata_dict.keys())) == 0:
                self.metadata_dict = self._get_rf_metadata(rf_file_basename)
            self.metadata = numpy.concatenate((self.metadata, added_rows))

        self._update_cont_metadata()

    def get_continuous_blocks(self, start_unix_sample, stop_unix_sample):
        """get_continuous_blocks returns a numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block.  Only samples between (start_unix_sample, stop_unix_sample)
        inclusive will be returned.

        Returns IOError if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                inclusive will be returned.
        """
        # to improve speed, do searchsorted to get first index to look into
        first_index = numpy.searchsorted(
            self.cont_metadata["unix_sample_index"], numpy.array([start_unix_sample])
        )
        first_index = first_index[0]
        if first_index > 0:
            first_index -= 1

        # for now, deal with two edges later
        bool_arr = self.cont_metadata["unix_sample_index"] >= start_unix_sample
        bool_arr1 = self.cont_metadata["unix_sample_index"] <= stop_unix_sample
        bool_arr = numpy.logical_and(bool_arr, bool_arr1)
        metadata_slice = self.cont_metadata[bool_arr]

        # handle front edge
        ones = numpy.ones((len(bool_arr),))
        zeros = numpy.zeros((len(bool_arr),))
        indices = numpy.nonzero(numpy.where(bool_arr, ones, zeros))
        if len(indices[0]):
            first_index = indices[0][0]
            if first_index != 0:
                # in this case, we may need to add one line before, but modify it
                previous_sample_index = self.cont_metadata[first_index - 1][
                    "unix_sample_index"
                ]
                previous_sample_extent = self.cont_metadata[first_index - 1][
                    "sample_extent"
                ]
                if previous_sample_index + previous_sample_extent >= start_unix_sample:
                    metadata_slice = numpy.concatenate(
                        (
                            self.cont_metadata[first_index - 1 : first_index],
                            metadata_slice,
                        )
                    )
                    metadata_slice[0] = (
                        start_unix_sample,
                        previous_sample_extent
                        - (start_unix_sample - previous_sample_index),
                    )
            # else there is no need to fix front edge
        else:
            previous_sample_index = self.cont_metadata[0]["unix_sample_index"]
            previous_sample_extent = self.cont_metadata[0]["sample_extent"]
            if previous_sample_index + previous_sample_extent >= start_unix_sample:
                metadata_slice = numpy.zeros((1,), dtype=self.cont_data_t)
                metadata_slice[0] = (
                    start_unix_sample,
                    previous_sample_extent
                    - (start_unix_sample - previous_sample_index),
                )

        # fix end if need
        last_sample_index = metadata_slice[-1]["unix_sample_index"]
        last_sample_extent = metadata_slice[-1]["sample_extent"]
        real_last_sample_extent = 1 + (stop_unix_sample - last_sample_index)
        if real_last_sample_extent < last_sample_extent:
            metadata_slice[-1]["sample_extent"] = real_last_sample_extent

        ret_arr = numpy.zeros((len(metadata_slice), 2), dtype=numpy.uint64)
        ret_arr[:, 0] = metadata_slice["unix_sample_index"]
        ret_arr[:, 1] = metadata_slice["sample_extent"]

        return ret_arr

    def get_continuous_vector(self, start_unix_sample, stop_unix_sample):
        """get_continuous_vector returns a tuple of (numpy array of data, first unix_sample in returned data)
        Only samples between (start_unix_sample, stop_unix_sample) (excludes stop_unix_sample) will be returned.

        Returns IOError if no blocks found

        Inputs:
            start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                (excludes stop_unix_sample) will be returned.
        """
        # check if we can use cached file
        if self._last_start_sample:
            if (
                start_unix_sample > self._last_start_sample
                and stop_unix_sample <= self._last_start_sample + self.samples_per_file
            ):
                return self._get_data_from_cache(start_unix_sample, stop_unix_sample)
            else:
                # cache has expired
                try:
                    self._last_file.close()
                except:
                    pass
                self._last_file = None
                self._last_start_sample = None

        # first verify no gaps in this range
        self._verify_no_gaps(start_unix_sample, stop_unix_sample)

        # to improve speed, do searchsorted to get first index to look into
        first_index = numpy.searchsorted(
            self.metadata["unix_sample_index"], numpy.array([start_unix_sample])
        )
        first_index = int(first_index[0])
        if first_index == len(self.metadata):
            first_index -= 1
        elif self.metadata["unix_sample_index"][first_index] > start_unix_sample:
            first_index -= 1
        ret_array = None
        first_unix_sample = None
        this_hdf5_file = None
        samples_read = 0
        samples_to_read = stop_unix_sample - start_unix_sample
        for i in range(first_index, len(self.metadata)):
            if self.metadata["unix_sample_index"][i] >= stop_unix_sample:
                raise IOError("Did not get expected read - debug")

            this_hdf5_file = self.metadata["rf_basename"][i]
            full_hdf5_file = os.path.join(
                self.top_level_dir, self.channel_name, self.subdirectory, this_hdf5_file
            )

            # get max possible length of this read as block_len
            if i == len(self.metadata) - 1:
                # last index
                block_len = self.samples_per_file - int(self.metadata["file_index"][i])
            elif self.metadata["rf_basename"][i + 1] == this_hdf5_file:
                block_len = int(self.metadata["file_index"][i + 1]) - int(
                    self.metadata["file_index"][i]
                )
            else:
                block_len = self.samples_per_file - int(self.metadata["file_index"][i])

            # next get file start index
            if i == first_index:
                offset = int(start_unix_sample) - int(
                    self.metadata["unix_sample_index"][first_index]
                )
                block_len -= offset  # we will not get the full read predicted above
                start_file_index = int(self.metadata["file_index"][i]) + offset
            else:
                start_file_index = int(self.metadata["file_index"][i])

            # next get read len
            if samples_read + block_len > samples_to_read:
                read_len = samples_to_read - samples_read
            else:
                read_len = block_len

            # finally - read it!!!
            f = h5py.File(full_hdf5_file, "r")
            rf_data = f["/rf_data"][start_file_index : start_file_index + read_len]

            if ret_array is None:
                ret_array = rf_data
            else:
                ret_array = numpy.concatenate((ret_array, rf_data))
            samples_read += read_len
            if samples_read == samples_to_read:
                # check whether we can cache it
                if i == first_index and len(f["/rf_data_index"]) == 1:
                    self._last_start_sample = int(self.metadata["unix_sample_index"][i])
                    if not self._last_file is None:
                        try:
                            self._last_file.close()
                        except:
                            pass
                    self._last_file = f
                else:
                    f.close()
                break
            else:
                f.close()

        return (ret_array, start_unix_sample)

    def get_first_sample(self):
        """get_first_sample returns the first sample index in this subdirectory.  May be exact (if self.metadata
        not is None) or an estimate based one subdirectory naming convention.
        """
        if len(self.metadata) > 0:
            return self.metadata["unix_sample_index"][0]

        rf_file_list = glob.glob(
            os.path.join(
                self.top_level_dir,
                self.channel_name,
                self.subdirectory,
                self._rf_file_glob,
            )
        )

        if len(rf_file_list) == 0:
            raise IOError(
                "No valid rf files found in subdirectory %s"
                % (
                    os.path.join(
                        self.top_level_dir, self.channel_name, self.subdirectory
                    )
                )
            )

        rf_file_list.sort()

        new_rows = self._get_new_rows(os.path.basename(rf_file_list[0]))
        return new_rows["unix_sample_index"][0]

    def get_last_sample(self):
        """get_last_sample returns the last sample in a subdirectory.
        """
        if len(self.metadata) > 0 and len(list(self.metadata_dict.keys())):
            return (
                self.metadata["unix_sample_index"][-1]
                + self.metadata_dict["samples_per_file"]
                - self.metadata["file_index"][-1]
            )

        rf_file_list = glob.glob(
            os.path.join(
                self.top_level_dir,
                self.channel_name,
                self.subdirectory,
                self._rf_file_glob,
            )
        )

        if len(rf_file_list) == 0:
            raise IOError(
                "No valid rf files found in subdirectory %s"
                % (
                    os.path.join(
                        self.top_level_dir, self.channel_name, self.subdirectory
                    )
                )
            )

        rf_file_list.sort()

        if len(list(self.metadata_dict.keys())) == 0:
            self.metadata_dict = self._get_rf_metadata(
                os.path.basename(rf_file_list[0])
            )

        index = -1
        if self._file_is_open(rf_file_list[-1]) and len(rf_file_list) > 1:
            index = -2

        new_rows = self._get_new_rows(os.path.basename(rf_file_list[index]))
        if (new_rows is None and index == -2) or (
            new_rows is None and len(rf_file_list) == 1
        ):
            raise IOError(
                "Unable to read rf file %s - one possible error is an empty file, which can be removed via <find . -size 0 -exec rm {} \;>"
                % (rf_file_list[index])
            )
        elif new_rows is None and index == -1:
            index = -2
            new_rows = self._get_new_rows(os.path.basename(rf_file_list[index]))
            if new_rows is None:
                raise IOError(
                    "Unable to read rf file %s - one possible error is an empty file, which can be removed via <find . -size 0 -exec rm {} \;>"
                    % (rf_file_list[index])
                )
        try:
            return (
                new_rows["unix_sample_index"][-1]
                + self.metadata_dict["samples_per_file"]
                - new_rows["file_index"][-1]
            )
        except:
            if index == -1 and len(rf_file_list) > 1:
                # try earlier file
                index = -2
                new_rows = self._get_new_rows(os.path.basename(rf_file_list[index]))
                if new_rows is None:
                    raise IOError(
                        "Unable to read rf file %s - one possible error is an empty file, which can be removed via <find . -size 0 -exec rm {} \;>"
                        % (rf_file_list[index])
                    )
                return (
                    new_rows["unix_sample_index"][-1]
                    + self.metadata_dict["samples_per_file"]
                    - new_rows["file_index"][-1]
                )
            else:
                raise

    def _update_continuous_data(self, rf_file_basename_list, rf_file_list):
        """_update_continuous_data updates all metadata if data in subdirectory is continuous, then return True.  Does nothing
        and returns False if not continuous data.

        Determines if continuous by looking only at first and last file in rf_file_basename_list.

        Inputs:
            rf_file_basename_list - sorted list of basenames in subdirectory
            rf_file_list - sorted list of full names  in subdirectory
        """
        first_index = None
        last_index = None

        # get info from first file
        for i, rf_file_basename in enumerate(rf_file_basename_list):
            if first_index is None:
                first_row = self._get_new_rows(rf_file_basename)
                if not first_row is None:
                    if len(first_row) > 1:
                        return False
                    first_index = i
                    first_sample = first_row["unix_sample_index"][0]
                    break
                else:
                    continue

        if first_index is None:
            return False

        # get info from last file
        if not self._file_is_open(rf_file_list[-1]):
            last_row = self._get_new_rows(rf_file_basename_list[-1])
            last_index = len(rf_file_basename_list)
            self.last_timestamp = self._get_utc_timestamp(rf_file_list[-1])
        else:
            last_row = self._get_new_rows(rf_file_basename_list[-2])
            last_index = len(rf_file_basename_list) - 1
            self.last_timestamp = self._get_utc_timestamp(rf_file_list[-2])
        if last_row is None:
            return False
        if len(last_row) > 1:
            return False
        last_sample = last_row["unix_sample_index"]
        self.file_count = last_index - first_index
        if (self.file_count - 1) * self.samples_per_file < last_sample - first_sample:
            # data gaps detected
            return False

        # create all metadata
        self.metadata = numpy.recarray((self.file_count,), dtype=self.data_t)
        sample_data = numpy.arange(
            0,
            self.file_count * self.samples_per_file,
            self.samples_per_file,
            dtype=numpy.int64,
        )
        sample_data += first_sample
        self.metadata["unix_sample_index"] = sample_data
        self.metadata["file_index"][:] = 0
        self.metadata["rf_basename"] = rf_file_basename_list[first_index:last_index]

        self._update_cont_metadata()
        return True

    def _verify_no_gaps(self, start_unix_sample, stop_unix_sample):
        """_verify_no_gaps raises an IOError if there is a gap between start_unix_sample, stop_unix_sample
        """
        # to improve speed, do searchsorted to get first index to look into
        first_index = numpy.searchsorted(
            self.cont_metadata["unix_sample_index"], numpy.array([start_unix_sample])
        )
        first_index = first_index[0]
        if first_index == len(self.cont_metadata):
            first_index -= 1
        elif self.cont_metadata["unix_sample_index"][first_index] > start_unix_sample:
            first_index -= 1
        offset = start_unix_sample - int(
            self.cont_metadata["unix_sample_index"][first_index]
        )
        if (
            self.cont_metadata["sample_extent"][first_index] - offset
            < stop_unix_sample - start_unix_sample
        ):
            raise IOError(
                "gap found between samples %i and %i"
                % (start_unix_sample, stop_unix_sample)
            )

    def _get_new_rows(self, rf_file_basename):
        """_get_new_rows is a private method that returns all needed rows for self.metadata in the correct recarray
        format for rf_file_basename, or None if that file has disappeared

        Inputs:
            rf_file_basename - rf file to examine

        Throws IOError if global indices overlap with previous metadata
        """
        # read data from /rf_data_index
        fullname = os.path.join(
            self.top_level_dir, self.channel_name, self.subdirectory, rf_file_basename
        )
        try:
            f = h5py.File(fullname, "r")
        except IOError:
            # presumably file deleted
            return None
        rf_data_index = f["/rf_data_index"]
        samples_per_file = f["rf_data"].attrs["samples_per_file"][0]
        if self.samples_per_file is None:
            self.samples_per_file = int(samples_per_file)
        elif self.samples_per_file != int(samples_per_file):
            raise IOError(
                "Illegal change in samples_per_file from %i to %i in file %s"
                % (self.samples_per_file, int(samples_per_file), fullname)
            )

        # create recarray
        new_rows = numpy.zeros((len(rf_data_index),), dtype=self.data_t)
        new_rows["unix_sample_index"] = rf_data_index[:, 0]
        new_rows["file_index"] = rf_data_index[:, 1]
        new_rows["rf_basename"] = rf_file_basename

        f.close()

        return new_rows

    def _get_rf_metadata(self, rf_file_basename):
        """_get_rf_metadata is a private method that returns a dictionary of all metadata stored in each rf file,
        or empty dict if that file has disappeared

        Inputs:
            rf_file_basename - rf file to examine

        Returns dictionary with string keys:
            sample_rate
            samples_per_file
            uuid_str
        """
        ret_dict = {}
        fullname = os.path.join(
            self.top_level_dir, self.channel_name, self.subdirectory, rf_file_basename
        )
        try:
            f = h5py.File(fullname, "r")
        except IOError:
            return {}
        dataset = f["/rf_data"]
        for attr in dataset.attrs:
            ret_dict[str(attr)] = dataset.attrs[attr]

        f.close()
        return ret_dict

    def _combine_blocks(self, first_array, second_array, samples_per_file):
        """_combine_blocks combines two numpy array of dtype u64 and shape (N,2) where the first
        column represents the unix_sample of a continuous block of data, and the second column represents the
        number of samples in that continuous block. The first row of the second array may or may not be contiguous
        with the last row of the first array.  If it is contiguous, that row will not be included, and the
        number of samples in that first row will instead be added to the last row of first_array. If not contiguous,
        the two arrays are simply concatenated
        """
        if len(first_array) == 0:
            return second_array
        is_contiguous = False
        if first_array[-1][0] + first_array[-1][1] > second_array[0][0]:
            raise IOError(
                "overlapping data found in top level directories %i, %i"
                % (first_array[-1][0] + first_array[-1][1], second_array[0][0])
            )
        if first_array[-1][0] + first_array[-1][1] == second_array[0][0]:
            is_contiguous = True
        if is_contiguous:
            first_array[-1][1] += second_array[0][1]
            if len(second_array) == 1:
                return first_array
            else:
                return numpy.concatenate([first_array, second_array[1:]])
        else:
            return numpy.concatenate([first_array, second_array])

    def _update_cont_metadata(self):
        """_update_cont_metadata completely rebuilds self.cont_metadata
        """
        cont_meta = []
        # handle empty dir case
        if len(self.metadata) == 0:
            self.cont_metadata = numpy.zeros((len(cont_meta),), dtype=self.cont_data_t)
            return
        for i in range(len(self.metadata)):
            if i == 0:
                cont_meta.append([self.metadata["unix_sample_index"][0], 0])
                last_sample = self.metadata["unix_sample_index"][0]
                last_index = 0
                continue
            this_sample = self.metadata["unix_sample_index"][i]
            this_index = self.metadata["file_index"][i]
            if this_index == 0:
                num_samples = self.samples_per_file - last_index
            else:
                num_samples = this_index - last_index
                if num_samples < 1:
                    raise ValueError("bug in self.metadata")
            cont_meta[-1][1] += num_samples
            if this_sample - last_sample == num_samples:
                if this_index != 0:
                    raise ValueError("bug 2 in self.metadata")
            else:
                cont_meta.append([this_sample, 0])
            last_sample = this_sample
            last_index = this_index

        # handle end of last file
        edge_samples = self.samples_per_file - last_index

        cont_meta[-1][1] += edge_samples

        cont_meta = numpy.array(cont_meta)

        # create self.cont_metadata
        self.cont_metadata = numpy.zeros((len(cont_meta),), dtype=self.cont_data_t)
        self.cont_metadata["unix_sample_index"] = cont_meta[:, 0]
        self.cont_metadata["sample_extent"] = cont_meta[:, 1]

    def _file_is_open(self, rf_file):
        """_file_is_open returns True if rf_file might be open (or corrupt), False otherwise
        """
        if time.time() - os.path.getmtime(rf_file) < 3:
            return True
        else:
            try:
                f = h5py.File(rf_file)
                f["/rf_data"].attrs["digital_rf_version"]
                f.close()
                return False
            except:
                try:
                    f.close()
                except:
                    pass
                return True

    def _get_utc_timestamp(self, fullfile):
        """_get_utc_timestamp returns the last modification timestamp of fullfile in UTC
        """
        # for now only local access
        if self.access_mode not in ("local"):
            raise ValueError("access_mode %s not yet implemented" % (self.access_mode))

        return os.path.getmtime(fullfile) - time.timezone

    def _get_data_from_cache(self, start_unix_sample, stop_unix_sample):
        """_get_data_from_cache simple returns the desired data from the cached Hdf5 file

        Inputs: start_unix_sample, stop_unix_sample - only samples between (start_unix_sample, stop_unix_sample)
                (excludes stop_unix_sample) will be returned.

        Calling method tested that this read is possible entirely within this file
        """
        start_index = start_unix_sample - self._last_start_sample
        samples_to_read = stop_unix_sample - start_unix_sample
        return (
            self._last_file["/rf_data"][start_index : start_index + samples_to_read],
            start_unix_sample,
        )


class _MissingMetadata(Exception):
    """_MissingMetadata is a Exception that will be raised when metadata needs to be updated
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
