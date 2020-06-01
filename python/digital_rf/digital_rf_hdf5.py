# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Read and write data in Digital RF HDF5 format.

It uses h5py to read, and exposes the capabilities of the C libdigital_rf
library to write.

Reading/writing functionality is available from two classes: DigitalRFReader
and DigitalRFWriter.

"""
from __future__ import absolute_import, division, print_function

import collections
import datetime
import fractions
import glob
import os
import re
import sys
import uuid
import warnings

import h5py
import numpy as np
import packaging.version
import six

# local imports
from . import _py_rf_write_hdf5, digital_metadata, list_drf
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = (
    "get_unix_time",
    "recreate_properties_file",
    "DigitalRFReader",
    "DigitalRFWriter",
)


libdigital_rf_version = _py_rf_write_hdf5.get_version()


def recreate_properties_file(channel_dir):
    """Re-create top-level drf_properties.h5 in channel dir.

    This function re-creates a missing drf_properties.h5 file in a Digital RF
    channel directory using the duplicate attributes stored in one of the data
    files.


    Parameters
    ----------
    channel_dir : string
        Channel directory containing Digital RF subdirectories in the form
        YYYY-MM-DDTHH-MM-SS, but missing a drf_properties.h5 file.


    Notes
    -----
    The following properties are read from a data file and stored as attributes
    in a new drf_properties.h5 file:

        H5Tget_class : int
            Result of H5Tget_class(hdf5_data_object->hdf5_data_object)
        H5Tget_offset : int
            Result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
        H5Tget_order : int
            Result of H5Tget_order(hdf5_data_object->hdf5_data_object)
        H5Tget_precision : int
            Result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
        H5Tget_size : int
            Result of H5Tget_size(hdf5_data_object->hdf5_data_object)
        digital_rf_time_description : string
            Text description of Digital RF time conventions.
        digital_rf_version : string
            Version string of Digital RF writer.
        epoch : string
            Start time at sample 0 (always 1970-01-01 UT midnight)
        file_cadence_millisecs : int
        is_complex : int
        is_continuous : int
        num_subchannels : int
        sample_rate_numerator : int
        sample_rate_denominator : int
        subdir_cadence_secs : int

    """
    properties_file = os.path.join(channel_dir, "drf_properties.h5")
    if os.access(properties_file, os.R_OK):
        raise IOError("drf_properties.h5 already exists in %s" % (channel_dir))

    subdirs = glob.glob(os.path.join(channel_dir, list_drf.GLOB_SUBDIR))
    if len(subdirs) == 0:
        errstr = "No subdirectories in form YYYY-MM-DDTHH-MM-SS found in %s"
        raise IOError(errstr % str(channel_dir))

    this_subdir = subdirs[len(subdirs) // 2]

    rf_file_glob = list_drf.GLOB_DRFFILE.replace("*", "rf", 1)
    rf_files = glob.glob(os.path.join(this_subdir, rf_file_glob))
    if len(rf_files) == 0:
        errstr = "No rf files in form rf@<ts>.h5 found in %s"
        raise IOError(errstr % this_subdir)

    with h5py.File(rf_files[0], "r") as fi:
        with h5py.File(properties_file, "w") as fo:
            md = fi["rf_data"].attrs
            fo.attrs["H5Tget_class"] = md["H5Tget_class"]
            fo.attrs["H5Tget_size"] = md["H5Tget_size"]
            fo.attrs["H5Tget_order"] = md["H5Tget_order"]
            fo.attrs["H5Tget_precision"] = md["H5Tget_precision"]
            fo.attrs["H5Tget_offset"] = md["H5Tget_offset"]
            fo.attrs["subdir_cadence_secs"] = md["subdir_cadence_secs"]
            fo.attrs["file_cadence_millisecs"] = md["file_cadence_millisecs"]
            fo.attrs["sample_rate_numerator"] = md["sample_rate_numerator"]
            fo.attrs["sample_rate_denominator"] = md["sample_rate_denominator"]
            fo.attrs["is_complex"] = md["is_complex"]
            fo.attrs["num_subchannels"] = md["num_subchannels"]
            fo.attrs["is_continuous"] = md["is_continuous"]
            fo.attrs["epoch"] = md["epoch"]
            fo.attrs["digital_rf_time_description"] = md["digital_rf_time_description"]
            fo.attrs["digital_rf_version"] = md["digital_rf_version"]


def get_unix_time(unix_sample_index, sample_rate_numerator, sample_rate_denominator):
    """Get unix time corresponding to a sample index for a given sample rate.

    Returns a tuple (dt, ps) containing a datetime and picosecond value. The
    returned datetime will contain microsecond precision. Picoseconds are also
    returned for users wanting greater precision than is available in the
    datetime object.


    Parameters
    ----------
    unix_sample_index : int
        Number of samples at given sample rate since UT midnight 1970-01-01

    sample_rate_numerator : int
        Numerator of sample rate in Hz.

    sample_rate_denominator : int
        Denominator of sample rate in Hz.


    Returns
    -------
    dt : datetime.datetime
        Time corresponding to the sample, with microsecond precision. This will
        give a Unix second of ``(unix_sample_index // sample_rate)``.

    picosecond : int
        Number of picoseconds since the last second in the returned datetime
        for the time corresponding to the sample.

    """
    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        picosecond,
    ) = _py_rf_write_hdf5.get_unix_time(
        unix_sample_index, sample_rate_numerator, sample_rate_denominator
    )
    dt = datetime.datetime(
        year, month, day, hour, minute, second, microsecond=int(picosecond / 1e6)
    )
    return (dt, picosecond)


class DigitalRFWriter(object):
    """Write a channel of data in Digital RF HDF5 format."""

    _writer_version = libdigital_rf_version

    def __init__(
        self,
        directory,
        dtype,
        subdir_cadence_secs,
        file_cadence_millisecs,
        start_global_index,
        sample_rate_numerator,
        sample_rate_denominator,
        uuid_str=None,
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    ):
        """Initialize writer to channel directory with given parameters.

        Parameters
        ----------
        directory : string
            The directory where this channel is to be written. It must already
            exist and be writable.

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

        start_global_index : int
            The index of the first sample given in number of samples since the
            epoch. For a given ``start_time`` in seconds since the epoch, this
            can be calculated as::

                floor(start_time * (np.longdouble(sample_rate_numerator) /
                                    np.longdouble(sample_rate_denominator)))

        sample_rate_numerator : int
            Numerator of sample rate in Hz.

        sample_rate_denominator : int
            Denominator of sample rate in Hz.


        Other Parameters
        ----------------
        uuid_str : None | string, optional
            UUID string that will act as a unique identifier for the data and
            can be used to tie the data files to metadata. If None, a random
            UUID will be generated.

        compression_level : int, optional
            0 for no compression (default), 1-9 for varying levels of gzip
            compression (1 == least compression, least CPU; 9 == most
            compression, most CPU).

        checksum : bool, optional
            If True, use Hdf5 checksum capability. If False (default), no
            checksum.

        is_complex : bool, optional
            This parameter is only used when `dtype` is not complex.
            If True (the default), interpret supplied data as interleaved
            complex I/Q samples. If False, each sample has a single value.

        num_subchannels : int, optional
            Number of subchannels to write simultaneously. Default is 1.

        is_continuous : bool, optional
            If True, data will be written in continuous blocks. If False data
            will be written with gapped blocks. Fastest write/read speed is
            achieved with `is_continuous` True, `checksum` False, and
            `compression_level` 0 (all defaults).

        marching_periods : bool, optional
            If True, write a period to stdout for every file when
            writing.

        """
        if not os.access(directory, os.W_OK):
            errstr = "Directory %s does not exist or is not writable"
            raise IOError(errstr % directory)
        self.directory = directory

        # use numpy to get all needed info about this datatype
        # set self.realdtype and self.is_complex
        dtype = np.dtype(dtype)
        self._dtype_is_complexfloating = False
        if np.issubdtype(dtype, np.complexfloating):
            self.is_complex = True
            self.realdtype = np.dtype("f{0}".format(dtype.itemsize // 2))
        elif dtype.names == ("r", "i"):
            self.is_complex = True
            self.realdtype = dtype["r"]
        elif not dtype.names:
            self.is_complex = bool(is_complex)
            self.realdtype = dtype
        else:
            raise ValueError("Structured dtype must have fields ('r', 'i').")
        # set self.dtype and self.structdtype
        if self.is_complex:
            self.structdtype = np.dtype([("r", self.realdtype), ("i", self.realdtype)])
            if np.issubdtype(self.realdtype, np.floating):
                # if floats, try to get equivalent complex type
                try:
                    self.dtype = np.dtype("c{0}".format(self.realdtype.itemsize * 2))
                except TypeError:
                    self.dtype = self.structdtype
                else:
                    self._dtype_is_complexfloating = True
            else:
                self.dtype = self.structdtype
        else:
            self.dtype = self.realdtype
            self.structdtype = None
        # set byteorder
        self.byteorder = self.realdtype.byteorder
        if self.byteorder == "=":
            # simplify C code by conversion here
            if sys.byteorder == "big":
                self.byteorder = ">"
            else:
                self.byteorder = "<"

        if subdir_cadence_secs != int(subdir_cadence_secs) or subdir_cadence_secs < 1:
            errstr = "subdir_cadence_secs must be positive integer, not %s"
            raise ValueError(errstr % str(subdir_cadence_secs))
        self.subdir_cadence_secs = int(subdir_cadence_secs)

        if (
            file_cadence_millisecs != int(file_cadence_millisecs)
            or file_cadence_millisecs < 1
        ):
            errstr = "file_cadence_millisecs must be positive integer, not %s"
            raise ValueError(errstr % str(file_cadence_millisecs))
        self.file_cadence_millisecs = int(file_cadence_millisecs)

        if self.subdir_cadence_secs * 1000 % self.file_cadence_millisecs != 0:
            raise ValueError(
                "(subdir_cadence_secs*1000 % file_cadence_millisecs)" " must equal 0"
            )

        if start_global_index < 0:
            errstr = "start_global_index cannot be negative (%s)"
            raise ValueError(errstr % str(start_global_index))
        self.start_global_index = int(start_global_index)

        if (
            sample_rate_numerator != int(sample_rate_numerator)
            or sample_rate_numerator < 1
        ):
            errstr = "sample_rate_numerator must be positive integer, not %s"
            raise ValueError(errstr % str(sample_rate_numerator))
        self.sample_rate_numerator = int(sample_rate_numerator)

        if (
            sample_rate_denominator != int(sample_rate_denominator)
            or sample_rate_denominator < 1
        ):
            errstr = "sample_rate_denominator must be positive integer, not %s"
            raise ValueError(errstr % str(sample_rate_denominator))
        self.sample_rate_denominator = int(sample_rate_denominator)

        if uuid_str is None:
            # generate random UUID
            uuid_str = uuid.uuid4().hex
        elif not isinstance(uuid_str, six.string_types):
            errstr = "uuid_str must be a string, not %s type"
            raise ValueError(errstr % str(type(uuid_str)))
        self.uuid = str(uuid_str)

        if compression_level not in range(10):
            errstr = "compression_level must be 0-9, not %s"
            raise ValueError(errstr % str(compression_level))
        self.compression_level = compression_level

        self.checksum = bool(checksum)

        if num_subchannels < 1:
            errstr = "Number of subchannels must be at least one, not %i"
            raise ValueError(errstr % num_subchannels)
        self.num_subchannels = int(num_subchannels)

        self.is_continuous = bool(is_continuous)

        if marching_periods:
            use_marching_periods = 1
        else:
            use_marching_periods = 0

        # call the underlying C extension, which will call the C init method
        self._channelObj = _py_rf_write_hdf5.init(
            directory,
            self.byteorder,
            self.realdtype.kind,
            self.realdtype.itemsize,
            self.subdir_cadence_secs,
            self.file_cadence_millisecs,
            self.start_global_index,
            self.sample_rate_numerator,
            self.sample_rate_denominator,
            uuid_str,
            compression_level,
            int(self.checksum),
            int(self.is_complex),
            self.num_subchannels,
            self.is_continuous,
            use_marching_periods,
        )

        if not self._channelObj:
            raise ValueError("Failed to create DigitalRFWriter")

        self._last_file_written = None
        self._last_dir_written = None
        self._last_utc_timestamp = None

        # set the next available sample to write at
        self._next_avail_sample = int(0)
        self._total_samples_written = int(0)
        self._total_gap_samples = int(0)

    def __enter__(self):
        """Enter method to enable context manager `with` statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method to enable context manager `with` statement."""
        self.close()

    @classmethod
    def get_version(cls):
        """Return the version string of the Digital RF writer."""
        return cls._writer_version

    def rf_write(self, arr, next_sample=None):
        """Write the next in-sequence samples from a given array.

        Parameters
        ----------
        arr : array_like
            Array of data to write. The array must have the same number of
            subchannels and type as declared when initializing the writer
            object, or an error will be raised. For single valued data, number
            of columns == number of subchannels. For complex data, there are
            two sizes of input arrays that are allowed:
                1. For a complex array or a structured array with column names
                    'r' and 'i' (as stored in the HDF5 file), the shape must
                    be (N, num_subchannels).
                2. For a non-structured, non-complex array, the shape must be
                    (N, 2*num_subchannels). I/Q are assumed to be interleaved.

        next_sample : long, optional
            Index of next sample to write relative to `start_global_index` of
            the first sample. If None (default), the array will be written to
            the next available sample after previous writes,
            `self._next_avail_sample`. A ValueError is raised if `next_sample`
            is less than `self._next_avail_sample`.


        Returns
        -------
        next_avail_sample : int
            Index of the next available sample after the array has been
            written.


        See Also
        --------
        rf_write_blocks


        Notes
        -----
        Here's an example of one way to create a structured numpy array with
        complex data with dtype int16::

            arr_data = np.ones(
                (num_rows, num_subchannels),
                dtype=[('r', np.int16), ('i', np.int16)],
            )
            for i in range(num_subchannels):
                for j in range(num_rows):
                    arr_data[j,i]['r'] = 2
                    arr_data[j,i]['i'] = 3

        The same data could be also be passed as an interleaved array::

            arr_data = np.ones(
                (num_rows, num_subchannels*2),
                dtype=np.int16,
            )

        """
        # verify input arr argument
        arr = self._cast_input_array(arr)

        if next_sample is None:
            next_sample = self._next_avail_sample
        else:
            next_sample = int(next_sample)
        if next_sample < self._next_avail_sample:
            errstr = "Trying to write at sample %i, but next available sample is %i"
            raise ValueError(errstr % (next_sample, self._next_avail_sample))

        try:
            next_avail_sample = _py_rf_write_hdf5.rf_write(
                self._channelObj, arr, next_sample
            )
        except AttributeError:
            # self._channelObj doesn't exist because writer has been closed
            raise IOError("Writer has been closed, cannot write.")

        # update index attributes
        nwritten = arr.shape[0]
        self._total_samples_written += nwritten
        gap_size = next_sample - self._next_avail_sample
        self._total_gap_samples += gap_size
        self._next_avail_sample = next_avail_sample

        return next_avail_sample

    def rf_write_blocks(self, arr, global_sample_arr, block_sample_arr):
        """Write blocks of data with interleaved gaps.

        Parameters
        ----------
        arr : array_like
            Array of data to write. See `rf_write` for a complete description
            of allowed forms.

        global_sample_arr : array_like of shape (N,) and type uint64
            An array that sets the global sample index (relative to
            `start_global_index` of the first sample) for each continuous
            block of data in arr. The values must be increasing, and the first
            value must be >= self._next_avail_sample or a ValueError raised.

        block_sample_arr : array_like of shape (N,) and type uint64
            An array that gives the index into arr for the start of each
            continuous block. The first value must be 0, and all values
            must be < len(arr). Increments between values must be > 0 and less
            than the corresponding increment in `global_sample_arr`.


        Returns
        -------
        next_avail_sample : int
            Index of the next available sample after the array has been
            written.


        See Also
        --------
        rf_write

        """
        # verify input arr argument
        arr = self._cast_input_array(arr)

        # cast global_sample_arr and block_sample_arr
        global_sample_arr = self._cast_sample_array(global_sample_arr)
        block_sample_arr = self._cast_sample_array(block_sample_arr)

        # check global_sample_arr and block_sample_arr values
        if global_sample_arr[0] < self._next_avail_sample:
            errstr = ("global_sample_arr[0] must be at least {0}, not {1}").format(
                self._next_avail_sample, global_sample_arr[0]
            )
            raise ValueError(errstr)
        if block_sample_arr[0] != 0:
            errstr = ("block_sample_arr[0] must be 0, not {0}.").format(
                block_sample_arr[0]
            )
            raise ValueError(errstr)
        if len(global_sample_arr) != len(block_sample_arr):
            errstr = (
                "Must have the same lengths: global_sample_arr ({0}) and"
                " block_sample_arr ({1})."
            ).format(len(global_sample_arr), len(block_sample_arr))
            raise ValueError(errstr)
        # view uint64 result as int64 as a hack to get negative results
        # when it makes sense
        block_steps = np.diff(block_sample_arr).view(dtype=np.int64)
        if np.any(block_steps < 1):
            errstr = ("block_sample_arr ({0}) must have increasing values").format(
                block_sample_arr
            )
            raise ValueError(errstr)
        # view uint64 result as int64 as a hack to get negative results
        # when it makes sense
        global_steps = np.diff(global_sample_arr).view(dtype=np.int64)
        if np.any(global_steps < 1):
            errstr = ("global_sample_arr ({0}) must have increasing values").format(
                global_sample_arr
            )
            raise ValueError(errstr)
        if block_sample_arr[-1] >= arr.shape[0]:
            errstr = (
                "block_sample_arr ({0}) has indices that reference past the"
                " end of the supplied data (with length {1})"
            ).format(block_sample_arr, arr.shape[0])
            raise ValueError(errstr)
        if np.any(block_steps > global_steps):
            errstr = (
                "Sample indices in global_sample_arr ({0}) would require"
                " overwriting data given the size of the corresponding data"
                " blocks in block_sample_arr ({1})"
            ).format(global_sample_arr, block_sample_arr)
            raise ValueError(errstr)

        # data passed initial tests, try to write
        try:
            next_avail_sample = _py_rf_write_hdf5.rf_block_write(
                self._channelObj, arr, global_sample_arr, block_sample_arr
            )
        except AttributeError:
            # self._channelObj doesn't exist because writer has been closed
            raise IOError("Writer has been closed, cannot write.")

        # update index attributes
        nwritten = arr.shape[0]
        self._total_samples_written += nwritten
        gap_size = (next_avail_sample - self._next_avail_sample) - nwritten
        self._total_gap_samples += gap_size
        self._next_avail_sample = next_avail_sample

        return next_avail_sample

    def get_total_samples_written(self):
        """Return the total number of samples written in per channel.

        This does not include gaps.

        """
        return self._total_samples_written

    def get_next_available_sample(self):
        """Return the index of the next sample available for writing.

        This is equal to (total_samples_written + total_gap_samples).

        """
        return self._next_avail_sample

    def get_total_gap_samples(self):
        """Return the total number of samples contained in data gaps."""
        return self._total_gap_samples

    def get_last_file_written(self):
        """Return the full path to the last file written."""
        try:
            return _py_rf_write_hdf5.get_last_file_written(self._channelObj)
        except AttributeError:
            return self._last_file_written

    def get_last_dir_written(self):
        """Return the full path to the last directory written."""
        try:
            return _py_rf_write_hdf5.get_last_dir_written(self._channelObj)
        except AttributeError:
            return self._last_dir_written

    def get_last_utc_timestamp(self):
        """Return UTC timestamp of the time of the last data written."""
        try:
            return _py_rf_write_hdf5.get_last_utc_timestamp(self._channelObj)
        except AttributeError:
            return self._last_utc_timestamp

    def close(self):
        """Free memory of the underlying C object and close the last HDF5 file.

        No more data can be written using this writer instance after close has
        been called.

        """
        if hasattr(self, "_channelObj"):
            # store last written properties so we can use them after close
            self._last_file_written = self.get_last_file_written()
            self._last_dir_written = self.get_last_dir_written()
            self._last_utc_timestamp = self.get_last_utc_timestamp()
            # now free the channel object
            del self._channelObj

    def _cast_input_array(self, arr):
        """Cast input array to correct type and check for the correct shape.

        Parameters
        ----------
        arr : array_like
            See `rf_write` method for a complete description of allowed values.


        Returns
        -------
        arr : ndarray of type self.dtype or self.structdtype


        Raises
        ------
        TypeError
            If the array type cannot be cast to the writer type.

        ValueError
            If the array shape does not match the specified number of
            subchannels.

        """
        # make sure arr is a contiguous array (as required by libidigital_rf)
        arr = np.ascontiguousarray(arr)
        # cast array to the correct type (if possible)
        if (
            self._dtype_is_complexfloating
            and np.issubdtype(arr.dtype, np.complexfloating)
        ) or not self.is_complex:  # cplx  # real input (failing above)->real output
            # input dtype and storage format are the same
            # one of real->real, complex->complex
            arr = arr.astype(self.dtype, casting="safe", copy=False)
        elif arr.dtype.names is not None and arr.dtype.names == ("r", "i"):
            # input as structured complex, stored as structure complex
            arr = arr.astype(self.structdtype, casting="safe", copy=False)
        else:
            # input as real, stored as structured complex
            # first make sure the real type is compatible
            arr = arr.astype(self.realdtype, casting="safe", copy=False)
            # then get a complex view
            arr = arr.view(dtype=self.structdtype)
        # check array shape
        if arr.ndim == 1:
            if self.num_subchannels > 1:
                errstr = "1 subchannel provided, {0} required.".format(
                    self.num_subchannels
                )
                raise ValueError(errstr)
            else:
                # make arr 2-D
                arr = arr.reshape((-1, 1))
        elif arr.ndim == 2:
            if arr.shape[1] != self.num_subchannels:
                errstr = "{0} subchannels provided, {1} required.".format(
                    arr.shape[1], self.num_subchannels
                )
                raise ValueError(errstr)
        else:
            errstr = "Illegal shape, must be (N, {0}) not {1}.".format(
                self.num_subchannels, arr.shape
            )
            raise ValueError(errstr)
        return arr

    def _cast_sample_array(self, sample_arr):
        """Cast sample array to equivalent values of uint64.

        Parameters
        ----------
        sample_arr : array_like
            Array of (global, block) sample indices.


        Returns
        -------
        sample_arr : ndarray of type uint64


        Raises
        ------
        TypeError
            If the array type cannot be cast to equivalent values of uint64.

        ValueError
            If the array is not 1-D.

        """
        # make sure arr is a contiguous array (as required by libidigital_rf)
        sample_arr = np.ascontiguousarray(sample_arr)
        # cast array to the correct type (if possible)
        sample_arr_uint64 = sample_arr.astype(np.uint64, casting="unsafe", copy=False)
        if not np.allclose(sample_arr_uint64, sample_arr):
            raise TypeError("Cannot cast sample_arr to uint64.")
        if sample_arr_uint64.ndim > 1:
            raise ValueError("sample_arr must be 1-D")
        return sample_arr_uint64


class DigitalRFReader(object):
    """Read data in Digital RF HDF5 format.

    This class allows random access to the rf data.

    """

    def __init__(self, top_level_directory_arg):
        """Initialize reader to directory containing Digital RF channels.

        Parameters
        ----------
        top_level_directory_arg : string
            Either a single top level directory containing Digital RF channel
            directories, or a list of such. A directory can be a file system
            path or a url, where the url points to a top level directory. Each
            must be a local path, or start with 'http://'', 'file://'', or
            'ftp://''.


        Notes
        -----
        A top level directory must contain files in the format:
            [channel]/[YYYY-MM-DDTHH-MM-SS]/rf@[seconds].[%03i milliseconds].h5

        If more than one top level directory contains the same channel_name
        subdirectory, this is considered the same channel. An error is raised
        if their sample rates differ, or if their time periods overlap.

        """
        # This method will create the following private attributes:
        # _top_level_dir_dict
        #   a dictionary with keys = top_level_directory string,
        #   value = access mode (eg, 'local', 'file', or 'http')
        # _channel_dict
        #   a dictionary with keys = channel_name,
        #   and value is a _channel_properties object.

        # first, make top_level_directory_arg a list if a string
        if isinstance(top_level_directory_arg, six.string_types):
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
            elif top_level_directory[0:7] == "ftp://":
                self._top_level_dir_dict[top_level_directory] = "ftp"
            else:
                # make sure absolute path used
                this_top_level_dir = os.path.abspath(top_level_directory)
                self._top_level_dir_dict[this_top_level_dir] = "local"

        self._channel_dict = {}
        # populate self._channel_dict
        # a temporary dict with key = channels, value = list of top level
        # directories where found
        channel_dict = {}
        for top_level_dir in self._top_level_dir_dict.keys():
            channels_found = self._get_channels_in_dir(top_level_dir)
            for channel in channels_found:
                channel_name = os.path.basename(channel)
                if channel_name in channel_dict:
                    channel_dict[channel_name].append(top_level_dir)
                else:
                    channel_dict[channel_name] = [top_level_dir]

        # update all channels
        for channel_name in channel_dict.keys():
            top_level_dir_properties_list = []
            for top_level_dir in channel_dict[channel_name]:
                new_top_level_metadata = _top_level_dir_properties(
                    top_level_dir, channel_name, self._top_level_dir_dict[top_level_dir]
                )
                top_level_dir_properties_list.append(new_top_level_metadata)
            new_channel_properties = _channel_properties(
                channel_name, top_level_dir_meta_list=top_level_dir_properties_list
            )
            self._channel_dict[channel_name] = new_channel_properties

        if not self._channel_dict:
            errstr = (
                "No channels found: top_level_directory_arg = {0}."
                " If path is correct, you may need to run"
                " recreate_properties_file to re-create missing"
                " drf_properties.h5 files."
            )
            raise ValueError(errstr.format(top_level_directory_arg))

        # dictionary to store cached Digital Metadata reader for each channel
        self._channel_metadata_reader = {}

    def __enter__(self):
        """Enter method to enable context manager `with` statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method to enable context manager `with` statement."""
        self.close()

    def close(self):
        """Close reader object and any open associated HDF5 files.

        This object cannot be used once it is closed.

        """
        self._channel_dict.clear()

    def get_channels(self):
        """Return an alphabetically sorted list of channels."""
        channels = sorted(self._channel_dict.keys())
        return channels

    def read(self, start_sample, end_sample, channel_name, sub_channel=None):
        """Read continuous blocks of data between start and end samples.

        This is the basic read method, upon which more specialized read methods
        are based. For general use, `read_vector` is recommended. This method
        returns data as it is stored in the HDF5 file: in blocks of continous
        samples and with HDF5-native types (e.g. complex integer-typed data has
        a stuctured dtype with 'r' and 'i' fields).


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will contain all subchannels of data and
            be 2-d. If an integer, the return array will be 1-d and contain the
            data of the subchannel given by that integer index.


        Returns
        -------
        OrderedDict
            The dictionary's keys are the start sample of each continuous block
            found between `start_sample` and `end_sample` (inclusive). Each
            value is the numpy array of continous data starting at the key's
            index. The returned array has the same type as the data stored in
            the HDF5 file's rf_data dataset.


        See Also
        --------
        get_continuous_blocks : Similar, except no data is read.
        read_vector : Read data into a vector of complex64 type.
        read_vector_c81d : Read data into a 1-d vector of complex64 type.
        read_vector_raw : Read data into a vector of HDF5-native type.

        """
        file_properties = self.get_properties(channel_name)
        if end_sample < start_sample:
            errstr = "start_sample %i greater than end sample %i"
            raise ValueError(errstr % (start_sample, end_sample))

        if sub_channel is not None:
            num_subchannels = file_properties["num_subchannels"]
            if sub_channel >= num_subchannels:
                errstr = "Data only has %i sub_channels, no sub_channel index %i"
                raise ValueError(errstr % (num_subchannels, sub_channel))

        # first get the names of all possible files with data
        subdir_cadence_secs = file_properties["subdir_cadence_secs"]
        file_cadence_millisecs = file_properties["file_cadence_millisecs"]
        samples_per_second = file_properties["samples_per_second"]
        filepaths = self._get_file_list(
            start_sample,
            end_sample,
            samples_per_second,
            subdir_cadence_secs,
            file_cadence_millisecs,
        )

        # key = start_sample, value = numpy array of contiguous data as in file
        cont_data_dict = {}
        for top_level_obj in self._channel_dict[channel_name].top_level_dir_meta_list:
            top_level_obj._read(
                start_sample,
                end_sample,
                filepaths,
                cont_data_dict,
                len_only=False,
                sub_channel=sub_channel,
            )

        # merge contiguous blocks
        return self._combine_blocks(cont_data_dict)

    def get_bounds(self, channel_name):
        """Get indices of first- and last-known sample for a given channel.

        Parameters
        ----------
        channel_name : string
            Name of channel, one of ``get_channels()``.


        Returns
        -------
        first_sample_index : int | None
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : int | None
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        """
        first_unix_sample = None
        last_unix_sample = None
        for top_level_obj in self._channel_dict[channel_name].top_level_dir_meta_list:
            this_first_sample, this_last_sample = top_level_obj._get_bounds()
            if first_unix_sample is not None:
                if this_first_sample is not None:
                    if this_first_sample < first_unix_sample:
                        first_unix_sample = this_first_sample
                    if this_last_sample > last_unix_sample:
                        last_unix_sample = this_last_sample
            else:
                if this_first_sample is not None:
                    first_unix_sample = this_first_sample
                    last_unix_sample = this_last_sample

        return (first_unix_sample, last_unix_sample)

    def get_properties(self, channel_name, sample=None):
        """Get dictionary of the properties particular to a Digital RF channel.

        Parameters
        ----------
        channel_name : string
            Name of channel, one of ``get_channels()``.

        sample : None | int
            If None, return the properties of the top-level drf_properties.h5
            file in the channel directory which applies to all samples. If a
            sample index is given, then return the properties particular to the
            file containing that sample index. This includes the top-level
            properties and additional attributes that can vary from file to
            file. If no data file is found associated with the input sample,
            then an IOError is raised.


        Returns
        -------
        dict
            Dictionary providing the properties.


        Notes
        -----
        The top-level properties, always returned, are:

            H5Tget_class : int
                Result of H5Tget_class(hdf5_data_object->hdf5_data_object)
            H5Tget_offset : int
                Result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
            H5Tget_order : int
                Result of H5Tget_order(hdf5_data_object->hdf5_data_object)
            H5Tget_precision : int
                Result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
            H5Tget_size : int
                Result of H5Tget_size(hdf5_data_object->hdf5_data_object)
            digital_rf_time_description : string
                Text description of Digital RF time conventions.
            digital_rf_version : string
                Version string of Digital RF writer.
            epoch : string
                Start time at sample 0 (always 1970-01-01 UT midnight)
            file_cadence_millisecs : int
            is_complex : int
            is_continuous : int
            num_subchannels : int
            sample_rate_numerator : int
            sample_rate_denominator : int
            samples_per_second : np.longdouble
            subdir_cadence_secs : int

        The additional properties particular to each file are:

            computer_time : int
                Unix time of initial file creation.
            init_utc_timestamp : int
                Changes at each restart of the recorder - needed if leap
                seconds correction applied.
            sequence_num : int
                Incremented for each file, starting at 0.
            uuid_str : string
                Set independently at each restart of the recorder.

        """
        global_properties = self._channel_dict[channel_name].properties
        if sample is None:
            return global_properties

        subdir_cadence_secs = global_properties["subdir_cadence_secs"]
        file_cadence_millisecs = global_properties["file_cadence_millisecs"]
        samples_per_second = global_properties["samples_per_second"]

        file_list = self._get_file_list(
            sample,
            sample,
            samples_per_second,
            subdir_cadence_secs,
            file_cadence_millisecs,
        )

        if len(file_list) != 1:
            raise ValueError("file_list is %s" % (str(file_list)))

        sample_properties = global_properties.copy()
        for top_level_obj in self._channel_dict[channel_name].top_level_dir_meta_list:
            fullfile = os.path.join(
                top_level_obj.top_level_dir, top_level_obj.channel_name, file_list[0]
            )
            if os.access(fullfile, os.R_OK):
                with h5py.File(fullfile, "r") as f:
                    md = {}
                    for key, val in f["rf_data"].attrs.items():
                        try:
                            # get python type if a numpy scalar or 1-D array
                            val = val.item()
                        except AttributeError:
                            pass
                        if isinstance(val, bytes):
                            # we know byte strings are ascii encoded,
                            # h5py (>=2.9) will decode all string attributes,
                            # decode here for consistency with on older h5py
                            val = val.decode("ascii")
                        md[key] = val
                    sample_properties.update(md)
                    return sample_properties

        errstr = "No data file found in channel %s associated with sample %i"
        raise IOError(errstr % (channel_name, sample))

    def get_digital_metadata(self, channel_name, top_level_dir=None):
        """Return `DigitalMetadataReader` object for <channel_name>/metadata.

        By convention, metadata in Digital Metadata format is stored in the
        'metadata' directory in a particular channel directory. This method
        returns a reader object for accessing that metadata. If no such
        directory exists, an IOError is raised.


        Parameters
        ----------
        channel_name : string
            Name of channel, one of ``get_channels()``.

        top_level_dir : None | string
            If None, use *first* metadata path starting from the top-level
            directory list of the current DigitalRFReader object, in case there
            is more than one match. Otherwise, use the given path as the
            top-level directory.


        Returns
        -------
        DigitalMetadataReader
            Metadata reader object for the given channel.

        """
        try:
            return self._channel_metadata_reader[channel_name]
        except KeyError:
            pass
        if top_level_dir is None:
            top_level_dirs = list(self._top_level_dir_dict.keys())
        else:
            top_level_dirs = [top_level_dir]
        for this_top_level_dir in top_level_dirs:
            metadata_dir = os.path.join(this_top_level_dir, channel_name, "metadata")
            if os.access(metadata_dir, os.R_OK):
                reader = digital_metadata.DigitalMetadataReader(metadata_dir)
                self._channel_metadata_reader[channel_name] = reader
                return reader

        # None found
        errstr = "Could not find valid digital_metadata in channel %s"
        raise IOError(errstr % channel_name)

    def read_metadata(self, start_sample, end_sample, channel_name, method="ffill"):
        """Read Digital Metadata accompanying a Digital RF channel.

        By convention, metadata in Digital Metadata format is stored in the
        'metadata' directory in a particular channel directory. This function
        reads that metadata for a specified sample range by getting a
        DigitalMetadataReader and calling its `read` function.


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : None | int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate). If None,
            use `end_sample` equal to `start_sample`.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        method : None | 'pad'/'ffill'
            If None, return only samples within the given range. If 'pad' or
            'ffill', the first sample no later than `start_sample` (if any)
            will also be included so that values are forward filled into the
            desired range.

        Returns
        -------
        OrderedDict
            The dictionary's keys are the sample index for each sample of
            metadata found between `start_sample` and `end_sample` (inclusive).
            Each value is a metadata sample given as a dictionary with column
            names as keys and numpy objects as leaf values.

        Notes
        -----
        For convenience, some pertinent metadata inherent to the Digital RF
        channel is added to the Digital Metadata, including:

            sample_rate_numerator : int
            sample_rate_denominator : int
            samples_per_second : np.longdouble

        """
        properties = self.get_properties(channel_name)
        added_metadata = {
            key: properties[key]
            for key in (
                "sample_rate_numerator",
                "sample_rate_denominator",
                "samples_per_second",
            )
        }
        try:
            reader = self.get_digital_metadata(channel_name)
        except IOError:
            ret_dict = collections.OrderedDict()
        else:
            ret_dict = reader.read(
                start_sample=start_sample,
                end_sample=end_sample,
                columns=None,
                method=method,
            )

        for d in ret_dict.values():
            d.update(added_metadata)
        if not ret_dict:
            # return inherent metadata even if Digital Metadata doesn't exist
            ret_dict[start_sample] = added_metadata
        return ret_dict

    def get_continuous_blocks(self, start_sample, end_sample, channel_name):
        """Find continuous blocks of data between start and end samples.

        This is similar to `read`, except it returns the length of the blocks
        of continous data instead of the data itself.


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.


        Returns
        -------
        OrderedDict
            The dictionary's keys are the start sample of each continuous block
            found between `start_sample` and `end_sample` (inclusive). Each
            value is the number of samples contained in that continous block
            of data.


        See Also
        --------
        read : Similar, except the data itself is returned.

        """
        # first get the names of all possible files with data
        file_properties = self.get_properties(channel_name)
        subdir_cadence_secs = file_properties["subdir_cadence_secs"]
        file_cadence_millisecs = file_properties["file_cadence_millisecs"]
        samples_per_second = file_properties["samples_per_second"]
        filepaths = self._get_file_list(
            start_sample,
            end_sample,
            samples_per_second,
            subdir_cadence_secs,
            file_cadence_millisecs,
        )

        # key = start_sample, value = len of contiguous data as in file
        cont_data_dict = {}
        for top_level_obj in self._channel_dict[channel_name].top_level_dir_meta_list:
            top_level_obj._read(
                start_sample, end_sample, filepaths, cont_data_dict, len_only=True
            )

        # merge contiguous blocks
        return self._combine_blocks(cont_data_dict, len_only=True)

    def get_last_write(self, channel_name):
        """Return tuple of time and path of the last file written to a channel.

        Parameters
        ----------
        channel_name : string
            Name of channel, one of ``get_channels()``.


        Returns
        -------
        timestamp : float | None
            Modification time of the last file written. None if there is no
            data.

        path : string | None
            Full path of the last file written. None if there is no data.

        """
        first_sample, last_sample = self.get_bounds(channel_name)
        if first_sample is None:
            return (None, None)
        file_properties = self.get_properties(channel_name)
        subdir_cadence_seconds = file_properties["subdir_cadence_secs"]
        file_cadence_millisecs = file_properties["file_cadence_millisecs"]
        samples_per_second = file_properties["samples_per_second"]
        file_list = self._get_file_list(
            last_sample - 1,
            last_sample,
            samples_per_second,
            subdir_cadence_seconds,
            file_cadence_millisecs,
        )
        file_list.reverse()
        for key in self._top_level_dir_dict.keys():
            for last_file in file_list:
                full_last_file = os.path.join(key, channel_name, last_file)
                if os.access(full_last_file, os.R_OK):
                    return (os.path.getmtime(full_last_file), full_last_file)

        # not found
        return (None, None)

    def read_vector(self, start_sample, vector_length, channel_name, sub_channel=None):
        """Read a complex vector of data beginning at the given sample index.

        This method returns the vector of the data beginning at `start_sample`
        with length `vector_length` for the given channel and sub_channel(s).
        The vector is always cast to a complex64 dtype no matter the original
        type of the data.

        This method calls `read` and converts the data appropriately. It will
        raise an IOError error if the returned vector would include any missing
        data.


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will contain all subchannels of data and
            be 2-d or 1-d depending on the number of subchannels. If an
            integer, the return array will be 1-d and contain the data of the
            subchannel given by that integer index.


        Returns
        -------
        array
            An array of dtype complex64 and shape (`vector_length`,) or
            (`vector_length`, N) where N is the number of subchannels.


        See Also
        --------
        read_vector_c81d : Read data into a 1-d vector of complex64 type.
        read_vector_raw : Read data into a vector of HDF5-native type.
        read : Read continuous blocks of data between start and end samples.

        """
        z = self.read_vector_raw(start_sample, vector_length, channel_name, sub_channel)

        if z.dtype.names is not None:
            y = np.empty(z.shape, dtype=np.complex64)
            y.real = z["r"]
            y.imag = z["i"]
            return y
        else:
            return np.array(z, dtype=np.complex64, copy=False)

    def read_vector_raw(
        self, start_sample, vector_length, channel_name, sub_channel=None
    ):
        """Read a vector of data beginning at the given sample index.

        This method returns the vector of the data beginning at `start_sample`
        with length `vector_length` for the given channel. The data is returned
        in its HDF5-native type (e.g. complex integer-typed data has a
        stuctured dtype with 'r' and 'i' fields).

        This method calls `read` and converts the data appropriately. It will
        raise an IOError error if the returned vector would include any missing
        data.


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will contain all subchannels of data and
            be 2-d or 1-d depending on the number of subchannels. If an
            integer, the return array will be 1-d and contain the data of the
            subchannel given by that integer index.


        Returns
        -------
        array
            An array of shape (`vector_length`,) or (`vector_length`, N) where
            N is the number of subchannels.


        See Also
        --------
        read_vector : Read data into a vector of complex64 type.
        read_vector_c81d : Read data into a 1-d vector of complex64 type.
        read : Read continuous blocks of data between start and end samples.

        """
        if vector_length < 1:
            estr = "Number of samples requested must be greater than 0, not %i"
            raise IOError(estr % vector_length)

        start_sample = int(start_sample)
        end_sample = start_sample + (int(vector_length) - 1)
        data_dict = self.read(start_sample, end_sample, channel_name, sub_channel)

        if len(data_dict) > 1:
            errstr = (
                "Data gaps found with start_sample %i and vector_length %i"
                " with channel %s"
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))
        elif len(data_dict) == 0:
            errstr = (
                "No data found with start_sample %i and vector_length %i"
                " with channel %s"
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))

        key, z = data_dict.popitem()
        # always return 1-D if possible
        z = z.squeeze()

        if len(z) != vector_length:
            errstr = "Requested %i samples, but got %i"
            raise IOError(errstr % (vector_length, len(z)))

        return z

    def read_vector_c81d(
        self, start_sample, vector_length, channel_name, sub_channel=0
    ):
        """Read a complex vector of data beginning at the given sample index.

        This method is identical to `read_vector`, except the default
        subchannel is 0 instead of None. As such, it always returns a 1-d
        vector of type complex64.


        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will contain all subchannels of data and
            be 2-d or 1-d depending on the number of subchannels. If an
            integer, the return array will be 1-d and contain the data of the
            subchannel given by that integer index.


        Returns
        -------
        array
            An array of dtype complex64 and shape (`vector_length`,).


        See Also
        --------
        read_vector : Read data into a vector of complex64 type.
        read_vector_raw : Read data into a vector of HDF5-native type.
        read : Read continuous blocks of data between start and end samples.

        """
        return self.read_vector(start_sample, vector_length, channel_name, sub_channel)

    @staticmethod
    def _get_file_list(
        sample0,
        sample1,
        samples_per_second,
        subdir_cadence_seconds,
        file_cadence_millisecs,
    ):
        """Get an ordered list of data file names that could contain data.

        This takes a first and last sample and generates the possible filenames
        spanning that time according to the subdirectory and file cadences.


        Parameters
        ----------
        sample0 : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        sample1 : int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        samples_per_second : np.longdouble
            Sample rate.

        subdir_cadence_secs : int
            Number of seconds of data found in one subdir. For example, 3600
            subdir_cadence_secs will be saved in each subdirectory.

        file_cadence_millisecs : int
            Number of milliseconds of data per file. Rule:
            (subdir_cadence_secs*1000 % file_cadence_millisecs) must equal 0.


        Returns
        -------
        list
            List of file paths that span the given time interval and conform
            to the subdirectory and file cadence naming scheme.

        """
        if (sample1 - sample0) > 1e12:
            warnstr = "Requested read size, %i samples, is very large"
            warnings.warn(warnstr % (sample1 - sample0), RuntimeWarning)
        sample0 = int(sample0)
        sample1 = int(sample1)
        # need to go through numpy uint64 to prevent conversion to float
        start_ts = int(np.uint64(sample0 / samples_per_second))
        end_ts = int(np.uint64(sample1 / samples_per_second)) + 1
        start_msts = int(np.uint64(sample0 / samples_per_second * 1000))
        end_msts = int(np.uint64(sample1 / samples_per_second * 1000))

        # get subdirectory start and end ts
        start_sub_ts = int(
            (start_ts // subdir_cadence_seconds) * subdir_cadence_seconds
        )
        end_sub_ts = int((end_ts // subdir_cadence_seconds) * subdir_cadence_seconds)

        ret_list = []  # ordered list of full file paths to return

        for sub_ts in range(
            start_sub_ts,
            int(end_sub_ts + subdir_cadence_seconds),
            subdir_cadence_seconds,
        ):
            sub_datetime = datetime.datetime.utcfromtimestamp(sub_ts)
            subdir = sub_datetime.strftime("%Y-%m-%dT%H-%M-%S")
            # create numpy array of all file TS in subdir
            file_msts_in_subdir = np.arange(
                sub_ts * 1000,
                int(sub_ts + subdir_cadence_seconds) * 1000,
                file_cadence_millisecs,
            )
            # file has valid samples if last time in file is after start time
            # and first time in file is before end time
            valid_in_subdir = np.logical_and(
                file_msts_in_subdir + file_cadence_millisecs - 1 >= start_msts,
                file_msts_in_subdir <= end_msts,
            )
            valid_file_ts_list = np.compress(valid_in_subdir, file_msts_in_subdir)
            for valid_file_ts in valid_file_ts_list:
                file_basename = "rf@%i.%03i.h5" % (
                    valid_file_ts // 1000,
                    valid_file_ts % 1000,
                )
                full_file = os.path.join(subdir, file_basename)
                ret_list.append(full_file)

        return ret_list

    def _combine_blocks(self, cont_data_dict, len_only=False):
        """Order and combine data given as dictionary into continuous blocks.

        Parameters
        ----------
        cont_data_dict : dict
            Dictionary where keys are the start sample of a block of data and
            values are arrays of the data as found in a file. These blocks do
            not cross file boundaries and so may need to be combined in this
            method.

        len_only : bool
            If True, returned dictionary values are lengths. If False,
            values are the continuous arrays themselves.


        Returns
        -------
        OrderedDict
            The dictionary's keys are the start sample of each continuous block
            in ascending order. Each value is the array or length of continous
            data starting at the key's index.

        """
        ret_dict = collections.OrderedDict()
        if len(cont_data_dict) == 0:
            # no data
            return ret_dict

        present_arr = None
        next_cont_sample = None
        for key, arr in sorted(cont_data_dict.items()):
            if present_arr is None:
                present_key = key
                present_arr = arr
            elif key == next_cont_sample:
                if len_only:
                    present_arr += arr
                else:
                    present_arr = np.concatenate((present_arr, arr))
            else:
                # non-continuous data found
                ret_dict[present_key] = present_arr
                present_key = key
                present_arr = arr

            if len_only:
                next_cont_sample = key + arr
            else:
                next_cont_sample = key + len(arr)

        # add last block
        ret_dict[present_key] = present_arr
        return ret_dict

    def _get_channels_in_dir(self, top_level_dir):
        """Return a list of channel paths found in a top-level directory.

        A channel is any subdirectory with a drf_properties.h5 file.


        Parameters
        ----------
        top_level_dir : string
            Path of the top-level directory.


        Returns
        -------
        list
            A list of strings giving the channel paths found.

        """
        retList = []
        access_mode = self._top_level_dir_dict[top_level_dir]

        if access_mode == "local":
            # detect if top_level_dir is a channel directory and raise
            # helpful error to let user know they need to specify parent
            properties_paths = [
                f
                for f in glob.glob(
                    os.path.join(top_level_dir, list_drf.GLOB_DRFPROPFILE)
                )
                if re.match(list_drf.RE_DRFPROP, f)
            ]
            if properties_paths:
                errstr = (
                    "'{0}' is a channel directory, but a top-level directory"
                    " containing channel directories is required. You probably"
                    " want to use '{1}' instead."
                ).format(top_level_dir, os.path.dirname(top_level_dir))
                raise ValueError(errstr)
            # list and match all channel dirs with properties files
            properties_paths = [
                f
                for f in glob.glob(
                    os.path.join(top_level_dir, "*", list_drf.GLOB_DRFPROPFILE)
                )
                if re.match(list_drf.RE_DRFPROP, f)
            ]
            for properties_path in properties_paths:
                channel_path = os.path.dirname(properties_path)
                if channel_path not in retList:
                    retList.append(channel_path)

        else:
            raise ValueError("access_mode %s not implemented" % (access_mode))

        return retList


class _channel_properties(object):
    """Properties for a Digital RF channel over one or more top-level dirs."""

    def __init__(self, channel_name, top_level_dir_meta_list=None):
        """Create a new _channel_properties object.

        This populates `self.properties`, which is a dictionary of
        attributes found in the HDF5 files (eg, samples_per_second). It also
        sets the attribute `max_samples_per_file`.


        Parameters
        ----------
        channel_name : string
            Name of subdirectory defining this channel.

        top_level_dir_meta_list : list
            A time ordered list of _top_level_dir_properties objects.

        """
        if top_level_dir_meta_list is None:
            top_level_dir_meta_list = []
        self.channel_name = channel_name
        self.top_level_dir_meta_list = top_level_dir_meta_list
        self.properties = self._read_properties()
        file_cadence_millisecs = self.properties["file_cadence_millisecs"]
        samples_per_second = self.properties["samples_per_second"]
        self.max_samples_per_file = int(
            np.uint64(np.ceil(file_cadence_millisecs * samples_per_second / 1000))
        )

    def _read_properties(self):
        """Get a dict of the properties stored in the drf_properties.h5 file.

        Returns
        -------
        dict
            A dictionary of the properties stored in the channel's
            drf_properties.h5 file.

        """
        ret_dict = {}

        for top_level_dir in self.top_level_dir_meta_list:
            if len(list(top_level_dir.properties.keys())) > 0:
                return top_level_dir.properties

        return ret_dict


class _top_level_dir_properties(object):
    """A Digital RF channel in a specific top-level directory."""

    _min_version = packaging.version.parse("2.0")
    _max_version = packaging.version.parse(
        packaging.version.parse(__version__).base_version
    )

    def __init__(self, top_level_dir, channel_name, access_mode):
        """Create a new _top_level_dir_properties object.

        Parameters
        ----------
        top_level_dir : string
            Full path the top-level directory that contains the parent
            `channel_name`.

        channel_name : string
            The subdirectory name for the channel.

        access_mode : string
            String giving the access mode ('local', 'file', or 'http').

        """
        self.top_level_dir = top_level_dir
        self.channel_name = channel_name
        self.access_mode = access_mode
        self._cachedFilename = None  # full name of last file opened
        self._cachedFile = None  # h5py.File object of last file opened
        # expect that _read_properties() will not raise error since we
        # already checked for existence of drf_properties.h5 before init
        self.properties = self._read_properties()
        try:
            version = self.properties["digital_rf_version"]
        except KeyError:
            # version is before 2.3 when key was added to metadata.h5/
            # drf_properties.h5 (versions before 2.0 will not have metadata.h5/
            # drf_properties.h5, so the directories will not register as
            # channels and the reader will not try to read them, so we can
            # assume at least 2.0)
            version = "2.0"
        version = packaging.version.parse(version)
        if version < self._min_version:
            errstr = (
                "The Digital RF files being read are version {0}, which is"
                " less than the required version ({1})."
            ).format(version.base_version, self._min_version.base_version)
            raise IOError(errstr)
        elif version > self._max_version:
            warnstr = (
                "The Digital RF files being read are version {0}, which is"
                " higher than the maximum supported version ({1}) for this"
                " digital_rf package. If you encounter errors, you will have"
                " upgrade to at least version {0} of digital_rf."
            ).format(version.base_version, self._max_version.base_version)
            warnings.warn(warnstr, RuntimeWarning)

    def _read_properties(self):
        """Get a dict of the properties stored in the drf_properties.h5 file.

        If no drf_properties.h5 file is found, an IOError is raised.


        Returns
        -------
        dict
            A dictionary of the properties stored in the channel's
            drf_properties.h5 file.

        """
        ret_dict = {}

        if self.access_mode == "local":
            # list and match first properties file
            properties_file = next(
                (
                    f
                    for f in glob.glob(
                        os.path.join(
                            self.top_level_dir,
                            self.channel_name,
                            list_drf.GLOB_DRFPROPFILE,
                        )
                    )
                    if re.match(list_drf.RE_DRFPROP, f)
                ),
                None,
            )
            if properties_file is None:
                raise IOError("drf_properties.h5 not found")
            f = h5py.File(properties_file, "r")
            for key, val in f.attrs.items():
                try:
                    # get scalar python type if a numpy scalar or 1-D array
                    val = val.item()
                except AttributeError:
                    pass
                if isinstance(val, bytes):
                    # we know byte strings are ascii encoded,
                    # h5py (>=2.9) will decode all string attributes but
                    # ascii arrays (even single-element) will always be bytes,
                    # DRF 2.0 writes string attributes as single-element
                    # arrays but >=2.1 writes them as scalars like h5py,
                    # decode here for consistency with h5py (>=2.9) scalars
                    val = val.decode("ascii")
                ret_dict[key] = val
            f.close()

        else:
            raise ValueError("mode %s not implemented" % (self.access_mode))

        # calculate samples_per_second as longdouble and add to properties
        # (so we only have to do this in one place)
        try:
            srn = ret_dict["sample_rate_numerator"]
            srd = ret_dict["sample_rate_denominator"]
        except KeyError:
            # if no sample_rate_numerator/sample_rate_denominator, then we must
            # have an older version with samples_per_second as uint64
            sps = ret_dict["samples_per_second"]
            spsfrac = fractions.Fraction(sps).limit_denominator()
            ret_dict["samples_per_second"] = np.longdouble(sps)
            ret_dict["sample_rate_numerator"] = spsfrac.numerator
            ret_dict["sample_rate_denominator"] = spsfrac.denominator
        else:
            sps = np.longdouble(np.uint64(srn)) / np.longdouble(np.uint64(srd))
            ret_dict["samples_per_second"] = sps

        # success
        return ret_dict

    def _read(
        self,
        start_sample,
        end_sample,
        filepaths,
        cont_data_dict,
        len_only=False,
        sub_channel=None,
    ):
        """Add continous data entries to `cont_data_dict`.

        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        filepaths : list
            A list of all valid subdir/filename that might contain data.

        cont_data_dict : dict
            Dictionary to add entries to. The keys are the start sample of
                each continuous block found between `start_sample` and
                `end_sample`, and the values are arrays or lengths of
                continuous data that start at the key.

        len_only : bool, optional
            If True, the values in `cont_data_dict` are lengths. If False,
            the values are the continuous arrays themselves.

        sub_channel : None | int, optional
            If None, include all subchannels. Otherwise, include only the
            subchannel given by that integer index.

        """
        if self.access_mode == "local":
            for fp in filepaths:
                fullfile = os.path.join(self.top_level_dir, self.channel_name, fp)
                if not os.access(fullfile, os.R_OK):
                    continue
                if fullfile != self._cachedFilename:
                    if self._cachedFile is not None:
                        try:
                            self._cachedFile.close()
                        except ValueError:
                            # already closed
                            pass
                    self._cachedFile = h5py.File(fullfile, "r")
                    self._cachedFilename = fullfile
                rf_data = self._cachedFile["rf_data"]
                rf_data_len = rf_data.shape[0]

                rf_index = self._cachedFile["rf_data_index"][...]
                rf_index_len = rf_index.shape[0]
                # loop through each row in rf_index
                for row in range(rf_index_len):
                    block_start_sample = int(rf_index[row, 0])
                    block_start_index = int(rf_index[row, 1])
                    if row + 1 == rf_index_len:
                        block_stop_index = rf_data_len
                    else:
                        block_stop_index = int(rf_index[row + 1, 1])
                    block_stop_sample = block_start_sample + (
                        block_stop_index - block_start_index
                    )

                    if start_sample <= block_start_sample:
                        read_start_index = block_start_index
                        read_start_sample = block_start_sample
                    elif start_sample < block_stop_sample:
                        read_start_index = block_start_index + (
                            start_sample - block_start_sample
                        )
                        read_start_sample = start_sample
                    else:
                        # no data in this block to read
                        continue

                    if end_sample + 1 >= block_stop_sample:
                        read_stop_index = block_stop_index
                    else:
                        read_stop_index = block_stop_index - (
                            block_stop_sample - (end_sample + 1)
                        )

                    # skip if no data found
                    if read_start_index >= read_stop_index:
                        continue
                    if not len_only:
                        if sub_channel is None:
                            data = rf_data[read_start_index:read_stop_index]
                        else:
                            data = rf_data[
                                read_start_index:read_stop_index, sub_channel
                            ]
                        cont_data_dict[read_start_sample] = data
                    else:
                        cont_data_dict[read_start_sample] = (
                            read_stop_index - read_start_index
                        )

        else:
            raise ValueError("mode %s not implemented" % (self.access_mode))

    def _get_bounds(self):
        """Get indices of first- and last-known sample for the channel.

        Returns
        -------
        first_sample_index : int | None
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : int | None
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        """
        first_unix_sample = None
        last_unix_sample = None
        if self.access_mode == "local":
            channel_dir = os.path.join(self.top_level_dir, self.channel_name)
            # loop through files in order to get first sample
            for path in list_drf.ilsdrf(
                channel_dir,
                recursive=False,
                reverse=False,
                include_drf=True,
                include_dmd=False,
                include_drf_properties=False,
            ):
                try:
                    first_unix_sample = self._get_first_sample(path)
                except IOError:
                    # can't open file (e.g. doesn't exist anymore)
                    continue
                except (AttributeError, IndexError, KeyError, ValueError):
                    errstr = (
                        "Warning: corrupt file %s found and ignored."
                        " Deleting it will speed up get_bounds()."
                    )
                    print(errstr % path)
                    continue
                else:
                    break

            # loop through files in reverse order to get last sample
            for path in list_drf.ilsdrf(
                channel_dir,
                recursive=False,
                reverse=True,
                include_drf=True,
                include_dmd=False,
                include_drf_properties=False,
            ):
                try:
                    last_unix_sample = self._get_last_sample(path)
                except IOError:
                    # can't open file (e.g. doesn't exist anymore)
                    continue
                except (AttributeError, IndexError, KeyError, ValueError):
                    errstr = (
                        "Warning: corrupt file %s found and ignored."
                        " Deleting it will speed up get_bounds()."
                    )
                    print(errstr % path)
                    continue
                else:
                    break
        else:
            raise ValueError("mode %s not implemented" % (self.access_mode))

        return (first_unix_sample, last_unix_sample)

    def _get_first_sample(self, fullname):
        """Return the first sample in a given rf file."""
        with h5py.File(fullname, "r") as f:
            return int(f["rf_data_index"][0][0])

    def _get_last_sample(self, fullname):
        """Return the last sample in a given rf file."""
        with h5py.File(fullname, "r") as f:
            total_samples = f["rf_data"].shape[0]
            rf_data_index = f["rf_data_index"]
            last_start_sample = rf_data_index[-1][0]
            last_index = rf_data_index[-1][1]
            return int(last_start_sample + (total_samples - (last_index + 1)))
