"""Read and write data in Digital RF HDF5 format.

It uses h5py to read, and exposes the capabilities of the C libdigital_rf
library to write.

Reading/writing functionality is available from two classes: DigitalRFReader
and DigitalRFWriter.

"""

import collections
import datetime
import distutils.version
import fractions
import glob
import os
import re
import sys
import types
import warnings

import h5py
import numpy

# local imports
from . import _py_rf_write_hdf5, digital_metadata, list_drf
from ._version import __version__

__all__ = (
    'get_unix_time', 'recreate_properties_file',
    'DigitalRFReader', 'DigitalRFWriter',
)

# constants
_min_version = '2.0'  # the version digital rf must be to be successfully read


def recreate_properties_file(channel_dir):
    """Helper function to re-create top-level drf_properties.h5 in channel dir.

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

        H5Tget_class : long
            Result of H5Tget_class(hdf5_data_object->hdf5_data_object)
        H5Tget_offset : long
            Result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
        H5Tget_order : long
            Result of H5Tget_order(hdf5_data_object->hdf5_data_object)
        H5Tget_precision : long
            Result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
        H5Tget_size : long
            Result of H5Tget_size(hdf5_data_object->hdf5_data_object)
        digital_rf_time_description : string
            Text description of Digital RF time conventions.
        digital_rf_version : string
            Version string of Digital RF writer.
        epoch : string
            Start time at sample 0 (always 1970-01-01 UT midnight)
        file_cadence_millisecs : long
        is_complex : int
        is_continuous : int
        num_subchannels : int
        sample_rate_numerator : long
        sample_rate_denominator : long
        subdir_cadence_secs : long

    """
    properties_file = os.path.join(channel_dir, 'drf_properties.h5')
    if os.access(properties_file, os.R_OK):
        raise IOError('drf_properties.h5 already exists in %s' % (channel_dir))

    subdirs = glob.glob(os.path.join(channel_dir, list_drf.GLOB_SUBDIR))
    if len(subdirs) == 0:
        errstr = 'No subdirectories in form YYYY-MM-DDTHH-MM-SS found in %s'
        raise IOError(errstr % str(channel_dir))

    this_subdir = subdirs[len(subdirs) / 2]

    rf_file_glob = list_drf.GLOB_DRFFILE.replace('*', 'rf', 1)
    rf_files = glob.glob(os.path.join(this_subdir, rf_file_glob))
    if len(rf_files) == 0:
        errstr = 'No rf files in form rf@<ts>.h5 found in %s'
        raise IOError(errstr % this_subdir)

    with h5py.File(rf_files[0], 'r') as fi:
        with h5py.File(properties_file, 'w') as fo:
            md = fi['rf_data'].attrs
            fo.attrs['H5Tget_class'] = md['H5Tget_class']
            fo.attrs['H5Tget_size'] = md['H5Tget_size']
            fo.attrs['H5Tget_order'] = md['H5Tget_order']
            fo.attrs['H5Tget_precision'] = md['H5Tget_precision']
            fo.attrs['H5Tget_offset'] = md['H5Tget_offset']
            fo.attrs['subdir_cadence_secs'] = md['subdir_cadence_secs']
            fo.attrs['file_cadence_millisecs'] = md['file_cadence_millisecs']
            fo.attrs['sample_rate_numerator'] = md['sample_rate_numerator']
            fo.attrs['sample_rate_denominator'] = md['sample_rate_denominator']
            fo.attrs['is_complex'] = md['is_complex']
            fo.attrs['num_subchannels'] = md['num_subchannels']
            fo.attrs['is_continuous'] = md['is_continuous']
            fo.attrs['epoch'] = md['epoch']
            fo.attrs['digital_rf_time_description'] = (
                md['digital_rf_time_description']
            )
            fo.attrs['digital_rf_version'] = md['digital_rf_version']


def get_unix_time(
    unix_sample_index, sample_rate_numerator, sample_rate_denominator,
):
    """Get unix time corresponding to a sample index for a given sample rate.

    Returns a tuple (dt, ps) containing a datetime and picosecond value. The
    returned datetime will contain microsecond precision. Picoseconds are also
    returned for users wanting greater precision than is available in the
    datetime object.


    Parameters
    ----------

    unix_sample_index : int | long
        Number of samples at given sample rate since UT midnight 1970-01-01

    sample_rate_numerator : long
        Numerator of sample rate in Hz.

    sample_rate_denominator : long
        Denominator of sample rate in Hz.


    Returns
    -------

    dt : datetime.datetime
        Time corresponding to the sample, with microsecond precision. This will
        give a Unix second of ``(unix_sample_index // sample_rate)``.

    picosecond : long
        Number of picoseconds since the last second in the returned datetime
        for the time corresponding to the sample.

    """
    year, month, day, hour, minute, second, picosecond = \
        _py_rf_write_hdf5.get_unix_time(
            unix_sample_index, sample_rate_numerator, sample_rate_denominator
        )
    dt = datetime.datetime(
        year, month, day, hour, minute, second,
        microsecond=long(picosecond / 1e6),
    )
    return((dt, picosecond))


class DigitalRFWriter:
    """Write a channel of data in Digital RF HDF5 format."""

    def __init__(
        self, directory, dtype, subdir_cadence_secs,
        file_cadence_millisecs, start_global_index, sample_rate_numerator,
        sample_rate_denominator, uuid_str, compression_level=0, checksum=False,
        is_complex=True, num_subchannels=1, is_continuous=True,
        marching_periods=True
    ):
        """Initialize writer to channel directory with given parameters.

        Parameters
        ----------

        directory : string
            The directory where this channel is to be written. It must already
            exist and be writable.

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

        start_global_index : long
            The index of the first sample given in number of samples since the
            epoch. For a given ``start_time`` in seconds since the epoch, this
            can be calculated as::

                floor(start_time * (numpy.longdouble(sample_rate_numerator) /
                                    numpy.longdouble(sample_rate_denominator)))

        sample_rate_numerator : long | int
            Numerator of sample rate in Hz.

        sample_rate_denominator : long | int
            Denominator of sample rate in Hz.

        uuid_str : string
            UUID string that will act as a unique identifier for the data and
            can be used to tie the data files to metadata.


        Other Parameters
        ----------------

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
            will be written with gapped blocks. Fastest read speed is achieved
            with is_continuous True, checksum False, and compression_level 0
            (all defaults).

        marching_periods : bool, optional
            If True, write a period to stdout for every subdirectory when
            writing.

        """
        if not os.access(directory, os.W_OK):
            errstr = 'Directory %s does not exist or is not writable'
            raise IOError(errstr % directory)
        self.directory = directory

        # use numpy to get all needed info about this datatype
        dtype = numpy.dtype(dtype)
        if numpy.issubdtype(dtype, numpy.complexfloating):
            self.is_complex = True
            self.itemdtype = numpy.dtype('f{0}'.format(dtype.itemsize/2))
        elif dtype.names == ('r', 'i') or dtype.names == ('i', 'r'):
            self.is_complex = True
            self.itemdtype = dtype['r']
        else:
            self.is_complex = bool(is_complex)
            self.itemdtype = dtype
        self.byteorder = self.itemdtype.byteorder
        if self.byteorder == '=':
            # simplify C code by conversion here
            if sys.byteorder == 'big':
                self.byteorder = '>'
            else:
                self.byteorder = '<'

        if subdir_cadence_secs < 1:
            errstr = 'subdir_cadence_secs must be positive, not %s'
            raise ValueError(errstr % str(subdir_cadence_secs))
        self.subdir_cadence_secs = long(subdir_cadence_secs)

        if file_cadence_millisecs < 1:
            errstr = 'file_cadence_millisecs must be positive, not %s'
            raise ValueError(errstr % str(file_cadence_millisecs))
        self.file_cadence_millisecs = long(file_cadence_millisecs)

        if self.subdir_cadence_secs * 1000 % self.file_cadence_millisecs != 0:
            raise ValueError(
                '(subdir_cadence_secs*1000 % file_cadence_millisecs)'
                ' must equal 0'
            )

        if start_global_index < 0:
            errstr = 'start_global_index cannot be negative (%s)'
            raise ValueError(errstr % str(start_global_index))
        self.start_global_index = long(start_global_index)

        intTypes = (types.IntType, types.LongType, numpy.integer)
        if not isinstance(sample_rate_numerator, intTypes):
            errstr = 'sample_rate_numerator illegal type %s'
            raise ValueError(errstr % str(type(sample_rate_numerator)))
        if not isinstance(sample_rate_denominator, intTypes):
            errstr = 'sample_rate_denominator illegal type %s'
            raise ValueError(errstr % str(type(sample_rate_denominator)))

        if sample_rate_numerator <= 0:
            errstr = 'sample_rate_numerator must be positive, not %s'
            raise ValueError(errstr % str(sample_rate_numerator))
        self.sample_rate_numerator = long(sample_rate_numerator)

        if sample_rate_denominator <= 0:
            errstr = 'sample_rate_denominator must be positive, not %s'
            raise ValueError(errstr % str(sample_rate_denominator))
        self.sample_rate_denominator = long(sample_rate_denominator)

        if not isinstance(uuid_str, types.StringTypes):
            errstr = 'uuid_str must be StringType, not %s'
            raise ValueError(errstr % str(type(uuid_str)))
        self.uuid = str(uuid_str)

        if compression_level not in range(10):
            errstr = 'compression_level must be 0-9, not %s'
            raise ValueError(errstr % str(compression_level))
        self.compression_level = compression_level

        self.checksum = bool(checksum)

        if num_subchannels < 1:
            errstr = 'Number of subchannels must be at least one, not %i'
            raise ValueError(errstr % num_subchannels)
        self.num_subchannels = int(num_subchannels)

        self.is_continuous = bool(is_continuous)

        if marching_periods:
            use_marching_periods = 1
        else:
            use_marching_periods = 0

        # call the underlying C extension, which will call the C init method
        self._channelObj = _py_rf_write_hdf5.init(
            directory, self.byteorder, self.itemdtype.char,
            self.itemdtype.itemsize,
            self.subdir_cadence_secs, self.file_cadence_millisecs,
            self.start_global_index, self.sample_rate_numerator,
            self.sample_rate_denominator, uuid_str, compression_level,
            int(self.checksum), int(self.is_complex), self.num_subchannels,
            self.is_continuous, use_marching_periods,
        )

        if not self._channelObj:
            raise ValueError('Failed to create DigitalRFWriter')

        # set the next available sample to write at
        self._next_avail_sample = long(0)
        self._total_samples_written = long(0)
        self._total_gap_samples = long(0)

    def rf_write(self, arr, next_sample=None):
        """Write the next in-sequence samples from a given array.

        Parameters
        ----------

        arr : array
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
            Global index of next sample to write to. If None (default), the
            array will be written to the next available sample after previous
            writes, self._next_avail_sample. An error is raised if next_sample
            is less than self._next_avail_sample.


        Returns
        -------

        next_avail_sample : long
            Index of the next available sample after the array has been
            written.


        See Also
        --------

        rf_write_blocks


        Notes
        -----

        Here's an example of one way to create a structured numpy array with
        complex data with dtype int16::

            arr_data = numpy.ones(
                (num_rows, num_subchannels),
                dtype=[('r', numpy.int16), ('i', numpy.int16)],
            )
            for i in range(num_subchannels):
                for j in range(num_rows):
                    arr_data[j,i]['r'] = 2
                    arr_data[j,i]['i'] = 3

        The same data could be also be passed as an interleaved array::

            arr_data = numpy.ones(
                (num_rows, num_subchannels*2),
                dtype=numpy.int16,
            )

        """
        arr = numpy.ascontiguousarray(arr)

        # verify input arr argument
        self._verify_input(arr)

        if next_sample is None:
            next_sample = self._next_avail_sample
        else:
            next_sample = long(next_sample)
        if next_sample < self._next_avail_sample:
            errstr = (
                'Trying to write at sample %i, but next available sample is %i'
            )
            raise ValueError(errstr % (next_sample, self._next_avail_sample))

        vector_length = int(arr.shape[0])

        _py_rf_write_hdf5.rf_write(self._channelObj, arr, next_sample)

        # update index attributes
        self._total_gap_samples += next_sample - self._next_avail_sample
        self._total_samples_written += vector_length
        self._next_avail_sample += (next_sample -
                                    self._next_avail_sample) + vector_length

    def rf_write_blocks(self, arr, global_sample_arr, block_sample_arr):
        """Write blocks of data with interleaved gaps.

        If is_continuous set in init, then the length of `global_sample_arr`
        and `block_sample_arr` must be 1 or an error is raised.


        Parameters
        ----------

        arr : array
            Array of data to write. See `rf_write` for a complete description
            of allowed forms.

        global_sample_arr : array of shape (N,) and type uint64
            An array that sets the global sample index for each continuous
            block of data in arr. The values must be increasing, and the first
            value must be >= self._next_avail_sample or a ValueError raised.

        block_sample_arr : array of shape (N,) and type uint64
            An array that gives the index into arr for the start of each
            continuous block. The first value must be zero, and all values
            must be < len(arr). Increments between values must be > 0 and less
            than the corresponding increment in `global_sample_arr`.


        Returns
        -------

        next_avail_sample : long
            Index of the next available sample after the array has been
            written.


        See Also
        --------

        rf_write

        """
        arr = numpy.ascontiguousarray(arr)

        # verify input arr argument
        self._verify_input(arr)

        if global_sample_arr[0] < self._next_avail_sample:
            errstr = (
                'first value in global_sample_arr must be at least %i, not %i'
            )
            raise ValueError(
                errstr % (self._next_avail_sample, global_sample_arr[0])
            )

        if block_sample_arr.dtype != numpy.uint64:
            errstr = (
                'block_sample_arr has dtype %s, but needs to have numpy.uint64'
            )
            raise ValueError(errstr % str(block_sample_arr.dtype))

        if block_sample_arr[0] != 0:
            errstr = 'first value in block_sample_arr must be 0, not %i'
            raise ValueError(errstr % (block_sample_arr[0]))

        if len(global_sample_arr) != len(block_sample_arr):
            errstr = (
                'len of global_sample_arr (%i) must equal len of'
                ' block_sample_arr (%i)'
            )
            raise ValueError(
                errstr % (len(global_sample_arr), len(block_sample_arr))
            )

        if self.is_continuous and len(global_sample_arr) > 1:
            raise IOError(
                'Cannot write gapped data after setting is_continuous True.'
            )

        # data passed initial tests, try to write
        _py_rf_write_hdf5.rf_block_write(
            self._channelObj, arr, global_sample_arr, block_sample_arr
        )

        # update index attributes
        # potential gap between writes
        self._total_gap_samples += (global_sample_arr[0] -
                                    self._next_avail_sample)
        self._total_gap_samples += ((global_sample_arr[-1] -
                                     global_sample_arr[0]) -
                                    block_sample_arr[-1])  # gaps within write
        self._total_samples_written += len(arr)
        self._next_avail_sample = (global_sample_arr[-1] +
                                   (len(arr) - block_sample_arr[-1]))

    def get_total_samples_written(self):
        """Return the total number of samples written in per channel.

        This does not include gaps.

        """
        return(self._total_samples_written)

    def get_next_available_sample(self):
        """Return the index of the next sample available for writing.

        This is equal to (total_samples_written + total_gap_samples).

        """
        return(self.next_available_sample)

    def get_total_gap_samples(self):
        """Return the total number of samples contained in data gaps."""
        return(self._total_gap_samples)

    def get_last_file_written(self):
        """Return the full path to the last file written."""
        return(_py_rf_write_hdf5.get_last_file_written(self._channelObj))

    def get_last_dir_written(self):
        """Return the full path to the last directory written."""
        return(_py_rf_write_hdf5.get_last_dir_written(self._channelObj))

    def get_last_utc_timestamp(self):
        """Return UTC timestamp of the time of the last data written."""
        return(_py_rf_write_hdf5.get_last_utc_timestamp(self._channelObj))

    def close(self):
        """Free memory of the underlying C object and close the last HDF5 file.

        No more data can be written using this writer instance after close has
        been called.

        """
        _py_rf_write_hdf5.free(self._channelObj)

    def _verify_input(self, arr):
        """Check for valid and consistent arrays for writing.

        Throws a ValueError if the array is invalid.

        Parameters
        ----------

        arr : array
            See `rf_write` method for a complete description of allowed values.

        """
        if self.is_complex:
            # there are three allowed ways to pass in complex data - see which
            # one used
            if arr.dtype.names is not None:
                # this must be be r/i format:
                for name in ('r', 'i'):
                    if name not in arr.dtype.names:
                        errstr = 'column names must be r and i, not %s'
                        raise ValueError(errstr % str(arr.dtype.names))
                    if not numpy.issubdtype(arr.dtype[name], self.itemdtype):
                        errstr = 'column %s must have dtype %s, not %s'
                        raise ValueError(errstr % (
                            name, str(self.itemdtype), str(arr.dtype[name]),
                        ))
                if len(arr.dtype.names) != 2:
                    errstr = 'column names must be only r and i, not %s'
                    raise ValueError(errstr % str(arr.dtype.names))
                if arr.shape[1] != self.num_subchannels:
                    errstr = (
                        'complex array in r/i form must have shape N x'
                        ' num_subchannels, not %s'
                    )
                    raise ValueError(errstr % str(arr.shape))
            elif numpy.issubdtype(arr.dtype, numpy.complexfloating):
                itemdtype = numpy.dtype('f{0}'.format(arr.dtype.itemsize/2))
                if not numpy.issubdtype(itemdtype, self.itemdtype):
                    errstr = (
                        'complex arr has item dtype %s, but dtype set in init'
                        ' was %s'
                    )
                    raise ValueError(
                        errstr % (str(itemdtype), str(self.itemdtype))
                    )
                if arr.shape[1] != self.num_subchannels:
                    errstr = (
                        'complex array in complex form must have shape N x'
                        ' num_subchannels, not %s'
                    )
                    raise ValueError(errstr % str(arr.shape))
            else:
                if not numpy.issubdtype(arr.dtype, self.itemdtype):
                    errstr = 'arr has dtype %s, but dtype set in init was %s'
                    raise ValueError(
                        errstr % (str(arr.dtype), str(self.itemdtype))
                    )
                if arr.shape[1] != 2 * self.num_subchannels:
                    errstr = (
                        'complex array in flat form must have shape N x'
                        ' 2*num_subchannels, not %s'
                    )
                    raise ValueError(errstr % str(arr.shape))

        else:  # single value checks
            if not numpy.issubdtype(arr.dtype, self.itemdtype):
                estr = 'arr has dtype %s, but dtype set in init was %s'
                raise ValueError(estr % (str(arr.dtype), str(self.itemdtype)))
            if len(arr.shape) == 1 and self.num_subchannels > 1:
                errstr = (
                    'single valued array must just have one subchannel, not'
                    ' shape %s num subchannels %i'
                )
                raise ValueError(
                    errstr % (str(arr.shape), self.num_subchannels)
                )
            if len(arr.shape) > 1:
                if arr.shape[1] != self.num_subchannels:
                    raise ValueError(
                        'input shape[1] %i must equal num subchannels %i' %
                        (arr.shape[1], self.num_subchannels))
            if len(arr.shape) > 2:
                raise ValueError('Illegal shape %s' % (str(arr.shape)))


class DigitalRFReader:
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
            <channel>/<YYYY-MM-DDTHH-MM-SS/rf@<seconds>.<%03i milliseconds>.h5

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
        if isinstance(top_level_directory_arg, types.StringType):
            top_level_arg = [top_level_directory_arg]
        else:
            top_level_arg = top_level_directory_arg

        # create static attribute self._top_level_dir_dict
        self._top_level_dir_dict = {}
        for top_level_directory in top_level_arg:
            if top_level_directory[0:7] == 'file://':
                self._top_level_dir_dict[top_level_directory] = 'file'
            elif top_level_directory[0:7] == 'http://':
                self._top_level_dir_dict[top_level_directory] = 'http'
            elif top_level_directory[0:7] == 'ftp://':
                self._top_level_dir_dict[top_level_directory] = 'ftp'
            else:
                # make sure absolute path used
                if top_level_directory[0] != '/':
                    this_top_level_dir = os.path.join(
                        os.getcwd(), top_level_directory)
                else:
                    this_top_level_dir = top_level_directory
                self._top_level_dir_dict[this_top_level_dir] = 'local'

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
                new_top_level_metaddata = _top_level_dir_properties(
                    top_level_dir, channel_name,
                    self._top_level_dir_dict[top_level_dir],
                )
                top_level_dir_properties_list.append(new_top_level_metaddata)
            new_channel_properties = _channel_properties(
                channel_name,
                top_level_dir_meta_list=top_level_dir_properties_list,
            )
            self._channel_dict[channel_name] = new_channel_properties

        if not self._channel_dict:
            errstr = (
                'No channels found: top_level_directory_arg = {0}. '
                'If path is correct, you may need to run '
                'recreate_properties_file to re-create missing '
                'drf_properties.h5 files.'
            )
            raise ValueError(errstr.format(top_level_directory_arg))

    def get_channels(self):
        """Return an alphabetically sorted list of channels."""
        channels = sorted(self._channel_dict.keys())
        return(channels)

    def read(self, start_sample, end_sample, channel_name, sub_channel=None):
        """Read continuous blocks of data between start and end samples.

        This is the basic read method, upon which more specialized read methods
        are based. For general use, `read_vector` is recommended. This method
        returns data as it is stored in the HDF5 file: in blocks of continous
        samples and with HDF5-native types (e.g. complex integer-typed data has
        a stuctured dtype with 'r' and 'i' fields).


        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : long
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will contain all subchannels of data and
            be 2-d or 1-d depending on the number of subchannels. If an
            integer, the return array will be 1-d and contain the data of the
            subchannel given by that integer index.


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
        is_continuous = file_properties['is_continuous']
        if end_sample < start_sample:
            errstr = 'start_sample %i greater than end sample %i'
            raise ValueError(errstr % (start_sample, end_sample))

        if sub_channel is not None:
            num_sub_channels = file_properties['num_subchannels']
            if num_sub_channels - 1 < sub_channel:
                errstr = (
                    'Data only has %i sub_channels, no sub_channel index %i'
                )
                raise ValueError(errstr % (num_sub_channels, sub_channel))

        # first get the names of all possible files with data
        subdir_cadence_secs = file_properties['subdir_cadence_secs']
        file_cadence_millisecs = file_properties['file_cadence_millisecs']
        samples_per_second = file_properties['samples_per_second']
        filepaths = self._get_file_list(
            start_sample, end_sample, samples_per_second,
            subdir_cadence_secs, file_cadence_millisecs,
        )

        # key = start_sample, value = numpy array of contiguous data as in file
        cont_data_dict = {}
        for top_level_obj in (
            self._channel_dict[channel_name].top_level_dir_meta_list
        ):
            top_level_obj._read(
                start_sample, end_sample, filepaths, cont_data_dict,
                False, sub_channel, is_continuous,
            )

        # merge contiguous blocks
        return(self._combine_blocks(cont_data_dict))

    def get_bounds(self, channel_name):
        """Get indices of first- and last-known sample for a given channel.

        Parameters
        ----------

        channel_name : string
            Name of channel, one of ``get_channels()``.


        Returns
        -------

        first_sample_index : long | None
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : long | None
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        """
        first_unix_sample = None
        last_unix_sample = None
        for top_level_obj in (
            self._channel_dict[channel_name].top_level_dir_meta_list
        ):
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

        return((first_unix_sample, last_unix_sample))

    def get_properties(self, channel_name, sample=None):
        """Get dictionary of the properties particular to a Digital RF channel.

        Parameters
        ----------

        channel_name : string
            Name of channel, one of ``get_channels()``.

        sample : None | long
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

            H5Tget_class : long
                Result of H5Tget_class(hdf5_data_object->hdf5_data_object)
            H5Tget_offset : long
                Result of H5Tget_offset(hdf5_data_object->hdf5_data_object)
            H5Tget_order : long
                Result of H5Tget_order(hdf5_data_object->hdf5_data_object)
            H5Tget_precision : long
                Result of H5Tget_precision(hdf5_data_object->hdf5_data_object)
            H5Tget_size : long
                Result of H5Tget_size(hdf5_data_object->hdf5_data_object)
            digital_rf_time_description : string
                Text description of Digital RF time conventions.
            digital_rf_version : string
                Version string of Digital RF writer.
            epoch : string
                Start time at sample 0 (always 1970-01-01 UT midnight)
            file_cadence_millisecs : long
            is_complex : int
            is_continuous : int
            num_subchannels : int
            sample_rate_numerator : long
            sample_rate_denominator : long
            samples_per_second : numpy.longdouble
            subdir_cadence_secs : long

        The additional properties particular to each file are:

            computer_time : long
                Unix time of initial file creation.
            init_utc_timestamp : long
                Changes at each restart of the recorder - needed if leap
                seconds correction applied.
            sequence_num : int
                Incremented for each file, starting at 0.
            uuid_str : string
                Set independently at each restart of the recorder.

        """
        global_properties = self._channel_dict[channel_name].properties
        if sample is None:
            return(global_properties)

        subdir_cadence_secs = global_properties['subdir_cadence_secs']
        file_cadence_millisecs = global_properties['file_cadence_millisecs']
        samples_per_second = global_properties['samples_per_second']

        file_list = self._get_file_list(
            sample, sample, samples_per_second,
            subdir_cadence_secs, file_cadence_millisecs,
        )

        if len(file_list) != 1:
            raise ValueError('file_list is %s' % (str(file_list)))

        sample_properties = global_properties.copy()
        for top_level_obj in (
            self._channel_dict[channel_name].top_level_dir_meta_list
        ):
            fullfile = os.path.join(
                top_level_obj.top_level_dir, top_level_obj.channel_name,
                file_list[0],
            )
            if os.access(fullfile, os.R_OK):
                with h5py.File(fullfile, 'r') as f:
                    md = {k: v.item() for k, v in f['rf_data'].attrs.items()}
                    sample_properties.update(md)
                    return(sample_properties)

        errstr = 'No data file found in channel %s associated with sample %i'
        raise IOError(errstr % (channel_name, sample))

    def get_digital_metadata(self, channel_name, top_level_dir=None):
        """Return `DigitalMetadataReader` object for <channel_name>/metadata.

        By convention, additional metadata in Digital Metadata format is
        stored in the 'metadata' directory in a particular channel directory.
        This method returns a reader object for accessing that metadata. If no
        such directory exists, an IOError is raised.


        Parameters
        ----------

        channel_name : string
            Name of channel, one of ``get_channels()``.

        top_level_dir : None | string
            If None, use *first* metadata path starting from the top-level
            directory list of the current DigitalRFReader object, in case there
            is more than one match. Otherwise, use the given path as the
            top-level directory.

        """
        if top_level_dir is None:
            top_level_dirs = self._top_level_dir_dict.keys()
        else:
            top_level_dirs = [top_level_dir]
        for this_top_level_dir in top_level_dirs:
            metadata_dir = os.path.join(
                this_top_level_dir, channel_name, 'metadata',
            )
            if os.access(metadata_dir, os.R_OK):
                return(digital_metadata.DigitalMetadataReader(metadata_dir))

        # None found
        errstr = 'Could not find valid digital_metadata in channel %s'
        raise IOError(errstr % channel_name)

    def get_continuous_blocks(self, start_sample, end_sample, channel_name):
        """Find continuous blocks of data between start and end samples.

        This is similar to `read`, except it returns the length of the blocks
        of continous data instead of the data itself.


        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : long
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
        subdir_cadence_secs = file_properties['subdir_cadence_secs']
        file_cadence_millisecs = file_properties['file_cadence_millisecs']
        samples_per_second = file_properties['samples_per_second']
        filepaths = self._get_file_list(
            start_sample, end_sample, samples_per_second,
            subdir_cadence_secs, file_cadence_millisecs,
        )

        # key = start_sample, value = len of contiguous data as in file
        cont_data_dict = {}
        for top_level_obj in (
            self._channel_dict[channel_name].top_level_dir_meta_list
        ):
            top_level_obj._read(
                start_sample, end_sample, filepaths, cont_data_dict,
                len_only=True,
            )

        # merge contiguous blocks
        return(self._combine_blocks(cont_data_dict, len_only=True))

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
            return((None, None))
        file_properties = self.get_properties(channel_name)
        subdir_cadence_seconds = file_properties['subdir_cadence_secs']
        file_cadence_millisecs = file_properties['file_cadence_millisecs']
        samples_per_second = file_properties['samples_per_second']
        file_list = self._get_file_list(
            last_sample - 1, last_sample, samples_per_second,
            subdir_cadence_seconds, file_cadence_millisecs,
        )
        file_list.reverse()
        for key in self._top_level_dir_dict.keys():
            for last_file in file_list:
                full_last_file = os.path.join(key, channel_name, last_file)
                if os.access(full_last_file, os.R_OK):
                    return((os.path.getmtime(full_last_file), full_last_file))

        # not found
        return((None, None))

    def read_vector(
        self, start_sample, vector_length, channel_name, sub_channel=None,
    ):
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

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will be 2-d and contain all subchannels
            of data. If an integer, the return array will be 1-d and contain
            the data of the subchannel given by that integer index.


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
        if vector_length < 1:
            estr = 'Number of samples requested must be greater than 0, not %i'
            raise IOError(estr % vector_length)

        start_sample = long(start_sample)
        end_sample = start_sample + (long(vector_length) - 1)
        data_dict = self.read(
            start_sample, end_sample, channel_name, sub_channel,
        )

        if len(data_dict.keys()) > 1:
            errstr = (
                'Data gaps found with start_sample %i and vector_length %i'
                ' with channel %s'
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))
        elif len(data_dict.keys()) == 0:
            errstr = (
                'No data found with start_sample %i and vector_length %i'
                ' with channel %s'
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))

        key = data_dict.keys()[0]
        z = data_dict[key]

        if len(z) != vector_length:
            errstr = 'Requested %i samples, but got %i'
            raise IOError(errstr % (vector_length, len(z)))

        if not hasattr(z.dtype, 'names'):
            return(numpy.array(z, dtype=numpy.complex64))
        elif z.dtype.names is None:
            return(numpy.array(z, dtype=numpy.complex64))
        z = numpy.array(z['r'] + z['i'] * 1.0j, dtype=numpy.complex64)
        return(z)

    def read_vector_raw(self, start_sample, vector_length, channel_name):
        """Read a vector of data beginning at the given sample index.

        This method returns the vector of the data beginning at `start_sample`
        with length `vector_length` for the given channel. The data is returned
        in its HDF5-native type (e.g. complex integer-typed data has a
        stuctured dtype with 'r' and 'i' fields) and includes all subchannels.

        This method calls `read` and converts the data appropriately. It will
        raise an IOError error if the returned vector would include any missing
        data.


        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.


        Returns
        -------

        array
            An array of shape (`vector_length`, N) where N is the number of
            subchannels.


        See Also
        --------

        read_vector : Read data into a vector of complex64 type.
        read_vector_c81d : Read data into a 1-d vector of complex64 type.
        read : Read continuous blocks of data between start and end samples.

        """
        if vector_length < 1:
            estr = 'Number of samples requested must be greater than 0, not %i'
            raise IOError(estr % vector_length)

        start_sample = long(start_sample)
        end_sample = start_sample + (long(vector_length) - 1)
        data_dict = self.read(start_sample, end_sample, channel_name)

        if len(data_dict.keys()) > 1:
            errstr = (
                'Data gaps found with start_sample %i and vector_length %i'
                ' with channel %s'
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))
        elif len(data_dict.keys()) == 0:
            errstr = (
                'No data found with start_sample %i and vector_length %i'
                ' with channel %s'
            )
            raise IOError(errstr % (start_sample, vector_length, channel_name))

        key = data_dict.keys()[0]
        z = data_dict[key]

        if len(z) != vector_length:
            errstr = 'Requested %i samples, but got %i'
            raise IOError(errstr % (vector_length, len(z)))

        return(z)

    def read_vector_c81d(
        self, start_sample, vector_length, channel_name, sub_channel=0,
    ):
        """Read a complex vector of data beginning at the given sample index.

        This method is identical to `read_vector`, except the default
        subchannel is 0 instead of None. As such, it always returns a 1-d
        vector of type complex64.


        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        vector_length : int
            Number of samples to read per subchannel.

        channel_name : string
            Name of channel to read from, one of ``get_channels()``.

        sub_channel : None | int, optional
            If None, the return array will be 2-d and contain all subchannels
            of data. If an integer, the return array will be 1-d and contain
            the data of the subchannel given by that integer index.


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
        return(self.read_vector(
            start_sample, vector_length, channel_name, sub_channel,
        ))

    @staticmethod
    def _get_file_list(
        sample0, sample1, samples_per_second,
        subdir_cadence_seconds, file_cadence_millisecs,
    ):
        """Get an ordered list of data file names that could contain data.

        This takes a first and last sample and generates the possible filenames
        spanning that time according to the subdirectory and file cadences.


        Parameters
        ----------

        sample0 : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        sample1 : long
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        samples_per_second : numpy.longdouble
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
            warnstr = 'Requested read size, %i samples, is very large'
            warnings.warn(warnstr % (sample1 - sample0), RuntimeWarning)
        sample0 = long(sample0)
        sample1 = long(sample1)
        # need to go through numpy uint64 to prevent conversion to float
        start_ts = long(numpy.uint64(sample0 / samples_per_second))
        end_ts = long(numpy.uint64(sample1 / samples_per_second)) + 1
        start_msts = long(numpy.uint64(sample0 / samples_per_second * 1000))
        end_msts = long(numpy.uint64(sample1 / samples_per_second * 1000))

        # get subdirectory start and end ts
        start_sub_ts = long(
            (start_ts // subdir_cadence_seconds) * subdir_cadence_seconds
        )
        end_sub_ts = long(
            (end_ts // subdir_cadence_seconds) * subdir_cadence_seconds
        )

        ret_list = []  # ordered list of full file paths to return

        for sub_ts in range(
            start_sub_ts,
            long(end_sub_ts + subdir_cadence_seconds),
            subdir_cadence_seconds,
        ):
            sub_datetime = datetime.datetime.utcfromtimestamp(sub_ts)
            subdir = sub_datetime.strftime('%Y-%m-%dT%H-%M-%S')
            # create numpy array of all file TS in subdir
            file_msts_in_subdir = numpy.arange(
                sub_ts * 1000,
                long(sub_ts + subdir_cadence_seconds) * 1000,
                file_cadence_millisecs,
            )
            # file has valid samples if last time in file is after start time
            # and first time in file is before end time
            valid_in_subdir = numpy.logical_and(
                file_msts_in_subdir + file_cadence_millisecs - 1 >= start_msts,
                file_msts_in_subdir <= end_msts,
            )
            valid_file_ts_list = numpy.compress(
                valid_in_subdir,
                file_msts_in_subdir,
            )
            for valid_file_ts in valid_file_ts_list:
                file_basename = 'rf@%i.%03i.h5' % (
                    valid_file_ts // 1000, valid_file_ts % 1000
                )
                full_file = os.path.join(subdir, file_basename)
                ret_list.append(full_file)

        return(ret_list)

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
        sample_keys = sorted(cont_data_dict.keys())
        ret_dict = collections.OrderedDict()
        if len(sample_keys) == 0:
            # no data
            return(ret_dict)

        present_arr = None
        next_cont_sample = None
        for i, key in enumerate(sample_keys):
            if present_arr is None:
                present_key = key
                present_arr = cont_data_dict[key]
            elif key == next_cont_sample:
                if len_only:
                    present_arr += cont_data_dict[key]
                else:
                    present_arr = numpy.concatenate(
                        (present_arr, cont_data_dict[key])
                    )
            else:
                # non-continuous data found
                ret_dict[present_key] = present_arr
                present_key = key
                present_arr = cont_data_dict[key]

            if len_only:
                next_cont_sample = key + cont_data_dict[key]
            else:
                next_cont_sample = key + len(cont_data_dict[key])

        # add last block
        ret_dict[present_key] = present_arr
        return(ret_dict)

    def _get_channels_in_dir(self, top_level_dir):
        """Return a list of channel names found in a top-level directory.

        A channel is any subdirectory with a drf_properties.h5 file.


        Parameters
        ----------

        top_level_dir : string
            Path of the top-level directory.


        Returns
        -------

        list
            A list of strings giving the channel names found.

        """
        retList = []
        access_mode = self._top_level_dir_dict[top_level_dir]

        if access_mode == 'local':
            # list and match all channel dirs with properties files
            potential_channels = [
                f for f in glob.glob(os.path.join(
                    top_level_dir, '*', list_drf.GLOB_DRFPROPFILE,
                )) if re.match(list_drf.RE_DRFPROP, f)
            ]
            for potential_channel in potential_channels:
                channel_name = os.path.dirname(potential_channel)
                if channel_name not in retList:
                    retList.append(channel_name)

        else:
            raise ValueError('access_mode %s not implemented' % (access_mode))

        return(retList)


class _channel_properties:
    """Properties for a Digital RF channel over one or more top-level dirs."""

    def __init__(self, channel_name, top_level_dir_meta_list=[]):
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
        self.channel_name = channel_name
        self.top_level_dir_meta_list = top_level_dir_meta_list
        self.properties = self._read_properties()
        file_cadence_millisecs = self.properties['file_cadence_millisecs']
        samples_per_second = self.properties['samples_per_second']
        self.max_samples_per_file = long(numpy.uint64(numpy.ceil(
            file_cadence_millisecs * samples_per_second / 1000
        )))

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
            if len(top_level_dir.properties.keys()) > 0:
                return(top_level_dir.properties)

        return(ret_dict)


class _top_level_dir_properties:
    """A Digital RF channel in a specific top-level directory."""

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
        # expect that _read_properties() will not raise error since we
        # already checked for existence of drf_properties.h5 before init
        self.properties = self._read_properties()
        try:
            version = self.properties['digital_rf_version']
        except KeyError:
            # version is before 2.3 when key was added to metadata.h5/
            # drf_properties.h5 (versions before 2.0 will not have metadata.h5/
            # drf_properties.h5, so the directories will not register as
            # channels and the reader will not try to read them, so we can
            # assume at least 2.0)
            version = '2.0'
        if (
            distutils.version.StrictVersion(version) <
            distutils.version.StrictVersion(_min_version)
        ):
            errstr = (
                'Digital RF files being read version %s,'
                ' less than required version %s'
            )
            raise IOError(errstr % (version, _min_version))
        self._cachedFilename = None  # full name of last file opened
        self._cachedFile = None  # h5py.File object of last file opened

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

        if self.access_mode == 'local':
            # list and match first properties file
            properties_file = next(
                (f for f in glob.glob(os.path.join(
                    self.top_level_dir, self.channel_name,
                    list_drf.GLOB_DRFPROPFILE,
                )) if re.match(list_drf.RE_DRFPROP, f)),
                None,
            )
            if properties_file is None:
                raise IOError('drf_properties.h5 not found')
            f = h5py.File(properties_file, 'r')
            for key in f.attrs.keys():
                ret_dict[key] = f.attrs[key].item()
            f.close()

        else:
            raise ValueError('mode %s not implemented' % (self.access_mode))

        # calculate samples_per_second as longdouble and add to properties
        # (so we only have to do this in one place)
        try:
            srn = ret_dict['sample_rate_numerator']
            srd = ret_dict['sample_rate_denominator']
        except KeyError:
            # if no sample_rate_numerator/sample_rate_denominator, then we must
            # have an older version with samples_per_second as uint64
            sps = ret_dict['samples_per_second']
            spsfrac = fractions.Fraction(sps).limit_denominator()
            ret_dict[u'samples_per_second'] = numpy.longdouble(sps)
            ret_dict[u'sample_rate_numerator'] = spsfrac.numerator
            ret_dict[u'sample_rate_denominator'] = spsfrac.denominator
        else:
            sps = (numpy.longdouble(numpy.uint64(srn)) /
                   numpy.longdouble(numpy.uint64(srd)))
            ret_dict[u'samples_per_second'] = sps

        # success
        return(ret_dict)

    def _read(
        self, start_sample, end_sample, filepaths, cont_data_dict,
        len_only=False, channel=None, is_continuous=0,
    ):
        """Add continous data entries to `cont_data_dict`.

        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : long
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

        is_continuous : 0 | 1, optional
            1 if continuous data, 0 if not. Used to speed up read.

        """
        if self.access_mode == 'local':
            for fp in filepaths:
                fullfile = os.path.join(
                    self.top_level_dir, self.channel_name, fp,
                )
                if not os.access(fullfile, os.R_OK):
                    continue
                if fullfile != self._cachedFilename:
                    if self._cachedFile is not None:
                        try:
                            self._cachedFile.close()
                        except ValueError:
                            # already closed
                            pass
                    self._cachedFile = h5py.File(fullfile, 'r')
                    self._cachedFilename = fullfile
                data_len = self._cachedFile['rf_data'].shape[0]

                rf_index = self._cachedFile['rf_data_index']
                # loop through each row in rf_index
                for row in range(rf_index.shape[0]):
                    this_sample = long(rf_index[row][0])
                    this_index = long(rf_index[row][1])
                    if row + 1 == rf_index.shape[0]:
                        last_index = long(data_len - 1)
                    else:
                        last_index = long(rf_index[row + 1][1] - 1)
                    last_sample = long(
                        this_sample + long(last_index - this_index)
                    )
                    if start_sample <= this_sample:
                        read_start_index = this_index
                        read_start_sample = long(this_sample)
                    elif start_sample <= last_sample:
                        read_start_index = long(
                            this_index + (start_sample - this_sample)
                        )
                        read_start_sample = long(
                            this_sample + (start_sample - this_sample)
                        )
                    else:
                        # no data in this block to read
                        continue
                    if end_sample >= last_sample:
                        read_end_index = long(last_index)
                    else:
                        read_end_index = long(
                            last_index - (last_sample - end_sample)
                        )

                    # skip if no data found
                    if read_start_index > read_end_index:
                        continue
                    if not len_only:
                        if channel is None:
                            data = self._cachedFile['rf_data'][
                                read_start_index:long(read_end_index + 1)
                            ]
                        else:
                            data = self._cachedFile['rf_data'][:, channel][
                                read_start_index:long(read_end_index + 1)
                            ]
                        cont_data_dict[read_start_sample] = data
                    else:
                        cont_data_dict[read_start_sample] = long(
                            (read_end_index + 1) - read_start_index
                        )

        else:
            raise ValueError('mode %s not implemented' % (self.access_mode))

    def _get_bounds(self):
        """Get indices of first- and last-known sample for the channel.

        Returns
        -------

        first_sample_index : long | None
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : long | None
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        """
        first_unix_sample = None
        last_unix_sample = None
        if self.access_mode == 'local':
            subdir_list = glob.glob(
                os.path.join(
                    self.top_level_dir,
                    self.channel_name,
                    list_drf.GLOB_SUBDIR,
                ),
            )
            if len(subdir_list) == 0:
                return((None, None))
            subdir_list.sort()
            rf_file_glob = list_drf.GLOB_DRFFILE.replace('*', 'rf', 1)
            for i, subdir in enumerate((subdir_list[0], subdir_list[-1])):
                rf_list = glob.glob(
                    os.path.join(subdir, rf_file_glob)
                )
                if len(rf_list) == 0:
                    continue
                rf_list.sort(key=list_drf.sortkey_drf)
                if i == 0:
                    this_first_sample = self._get_first_sample(rf_list[0])
                else:
                    for fullname in reversed(rf_list):
                        try:
                            this_last_sample = self._get_last_sample(fullname)
                            break
                        except Exception:
                            errstr = (
                                'Warning corrupt h5 file %s found - ignored'
                                ' - should be deleted'
                            )
                            print(errstr % fullname)
                            this_last_sample = None

            # check for all bad files in last subdirectory
            if this_last_sample is None:
                rf_list = glob.glob(
                    os.path.join(subdir_list[-2], rf_file_glob)
                )
                rf_list.sort(key=list_drf.sortkey_drf)
                this_last_sample = self._get_last_sample(rf_list[-1])

            if first_unix_sample is not None:
                if this_first_sample < first_unix_sample:
                    first_unix_sample = this_first_sample
                if this_last_sample > last_unix_sample:
                    last_unix_sample = this_last_sample
            else:
                first_unix_sample = this_first_sample
                last_unix_sample = this_last_sample

        else:
            raise ValueError('mode %s not implemented' % (self.access_mode))

        return((first_unix_sample, last_unix_sample))

    def _get_first_sample(self, fullname):
        """Return the first sample in a given rf file."""
        with h5py.File(fullname) as f:
            return(long(f['rf_data_index'][0][0]))

    def _get_last_sample(self, fullname):
        """Return the last sample in a given rf file."""
        with h5py.File(fullname) as f:
            total_samples = f['rf_data'].shape[0]
            rf_data_index = f['rf_data_index']
            last_start_sample = rf_data_index[-1][0]
            last_index = rf_data_index[-1][1]
            return(
                long(last_start_sample + (total_samples - (last_index + 1)))
            )

    def __del__(self):
        # Make sure cached file is closed - does not happen automatically
        try:
            if self._cachedFile is not None:
                self._cachedFile.close()
        except ValueError:
            # already closed
            pass
