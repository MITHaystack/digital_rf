"""Read and write data/metadata in Digital Metadata HDF5 format.

It uses h5py to read and write to HDF5 files.

Reading/writing functionality is available from two classes:
DigitalMetadataReader and DigitalMetadataWriter.

"""

import collections
import copy
import datetime
import fractions
import glob
import os
import re
import time
import traceback
import types
import urllib2
from distutils.version import StrictVersion

# third party imports
import h5py
import numpy

# local imports
from . import list_drf
from ._version import __version__

__all__ = (
    'DigitalMetadataReader', 'DigitalMetadataWriter',
)


class DigitalMetadataWriter:
    """Write data in Digital Metadata HDF5 format."""

    _min_version = StrictVersion('2.5')
    _max_version = StrictVersion(__version__)

    def __init__(
        self, metadata_dir, subdir_cadence_secs, file_cadence_secs,
        sample_rate_numerator, sample_rate_denominator, file_name,
    ):
        """Initialize writer to channel directory with given parameters.

        Parameters
        ----------

        metadata_dir : string
            The directory where this channel is to be written. It must already
            exist and be writable.

        subdir_cadence_secs : int
            The number of seconds of metadata to store in each subdirectory.
            The timestamp of any subdirectory will be an integer multiple of
            this value.

        file_cadence_secs : int
            The number of seconds of metadata to store in each file. Note that
            an integer number of files must exactly span a subdirectory,
            implying::

                (subdir_cadence_secs % file_cadence_secs) == 0

            This class enforces the rule that file name timestamps will be in
            the list::

                range(subdirectory_timestamp,
                      subdirectory_timestamp + subdir_cadence_secs,
                      file_cadence_secs)

        sample_rate_numerator : long | int
            Numerator of sample rate in Hz.

        sample_rate_denominator : long | int
            Denominator of sample rate in Hz.

        file_name : string
            Prefix for metadata file names. Files in each subdirectory will be
            named: "`file_name`@<timestamp>.h5".

        """
        # verify all input arguments
        if not os.access(metadata_dir, os.W_OK):
            errstr = 'metadata_dir %s does not exist or is not writable'
            raise IOError(errstr % metadata_dir)
        self._metadata_dir = metadata_dir

        intTypes = (types.IntType, types.LongType, numpy.integer)

        if not isinstance(subdir_cadence_secs, intTypes):
            errstr = 'subdir_cadence_secs must be int type, not %s'
            raise ValueError(errstr % str(type(subdir_cadence_secs)))
        if subdir_cadence_secs < 1:
            errstr = (
                'subdir_cadence_secs must be positive integer, not %s'
            )
            raise ValueError(errstr % str(subdir_cadence_secs))
        self._subdir_cadence_secs = int(subdir_cadence_secs)

        if not isinstance(file_cadence_secs, intTypes):
            errstr = 'file_cadence_secs must be int type, not %s'
            raise ValueError(errstr % str(type(file_cadence_secs)))
        if file_cadence_secs < 1:
            errstr = (
                'file_cadence_secs must be positive integer, not %s'
            )
            raise ValueError(errstr % str(file_cadence_secs))
        self._file_cadence_secs = int(file_cadence_secs)
        if ((self._subdir_cadence_secs % self._file_cadence_secs)
                != 0):
            raise ValueError(
                '(subdir_cadence_secs % file_cadence_secs) != 0'
            )

        if not isinstance(file_name, types.StringTypes):
            errstr = 'file_name must be a string, not type %s'
            raise ValueError(errstr % str(type(file_name)))
        self._file_name = file_name

        if not isinstance(sample_rate_numerator, intTypes):
            errstr = 'sample_rate_numerator must be int type, not %s'
            raise ValueError(errstr % str(type(sample_rate_numerator)))
        if sample_rate_numerator < 1:
            errstr = (
                'sample_rate_numerator must be positive integer, not %s'
            )
            raise ValueError(errstr % str(sample_rate_numerator))
        self._sample_rate_numerator = long(sample_rate_numerator)

        if not isinstance(sample_rate_denominator, intTypes):
            errstr = 'sample_rate_denominator must be int type, not %s'
            raise ValueError(errstr % str(type(sample_rate_denominator)))
        if sample_rate_denominator < 1:
            errstr = (
                'sample_rate_denominator must be positive integer, not %s'
            )
            raise ValueError(errstr % str(sample_rate_denominator))
        self._sample_rate_denominator = long(sample_rate_denominator)

        # have to go to uint64 before longdouble to ensure correct conversion
        # from long
        self._samples_per_second = (
            numpy.longdouble(numpy.uint64(self._sample_rate_numerator)) /
            numpy.longdouble(numpy.uint64(self._sample_rate_denominator))
        )

        if os.access(
            os.path.join(self._metadata_dir, 'dmd_properties.h5'),
            os.R_OK,
        ):
            self._parse_properties()
        else:
            self._digital_metadata_version = __version__
            self._fields = None  # No data written yet
            self._write_properties()

    def get_samples_per_second(self):
        """Return the sample rate in Hz as a numpy.longdouble."""
        return(self._samples_per_second)

    def write(self, samples, data_dict):
        """Write new metadata to the Digital Metadata channel.

        Parameters
        ----------

        samples : list | 1-D array | long | int | float
            A single sample index or an list of sample indices, given in
            the number of samples since the epoch (t_since_epoch*sample_rate),
            for the data to be written.

        data_dict : dict
            A dictionary representing the metadata to write with keys giving
            the field names and values giving the metadata itself. Each value
            must be either:

                - a 1-D numpy array or list/tuple of numpy objects with length
                    equal to ``len(samples)`` giving the metadata corresponding
                    to each sample index in `samples`
                - a single value or numpy array of shape different than
                    ``(len(samples),)``, giving the metadata for all of the
                    samples indices in `samples`
                - another dictionary with keys that are valid Group name
                    strings and leaf values that are one of the above

            This dictionary should always have the same keys each time the
            write method is called.

        """
        try:
            samples = numpy.array(
                samples, dtype=numpy.uint64, copy=False, ndmin=1,
            )
        except (TypeError, ValueError):
            raise ValueError(
                'Values in `samples` must be convertible to uint64'
            )
        if len(samples) == 0:
            raise ValueError('`samples` must not be empty')

        if self._fields is None:
            self._set_fields(data_dict.keys())

        # get mapping of sample ranges to subdir/filenames
        file_info_list = self._get_subdir_filename_info(samples)

        for sample_list, index_list, subdir, file_basename in file_info_list:
            self._write_metadata_range(
                data_dict, sample_list, index_list, len(samples), subdir,
                file_basename,
            )

    def _write_metadata_range(
        self, data_dict, sample_list, index_list, sample_len, subdir, filename,
    ):
        """Write metadata to a single file.

        This private method is called by `write` to carry out the writing of a
        list of samples.


        Parameters
        ----------

        data_dict : dict
            A dictionary representing the metadata to write with keys giving
            the field names and values giving the metadata itself. Each value
            must be either:

                - a 1-D numpy array or list/tuple of numpy objects of length
                    `sample_len`, with data at index ``index_list[k]``
                    corresponding to the kth sample index in `sample_list`
                - a single value or numpy array of shape different than
                    ``(sample_len,)``, in which case `index_list` will be
                    ignored
                - another dictionary with keys that are valid Group name
                    strings and leaf values that are one of the above

            This dictionary must always have the same keys each time the write
            method is called.

        sample_list : iterable
            Iterable of sample indices, given in the number of samples since
            the epoch (t_since_epoch*sample_rate), for the data to be written.

        index_list : list
            List of the same length as `sample_list` that gives the index into
            arrays/lists in `data_dict` for the corresponding sample in
            `sample_list`.

        sample_len : int
            Total number of samples to be written from this `data_dict`,
            possibly over multiple calls to this method. Used to determine if
            writing numpy arrays item by item or as a whole.

        subdir : string
            Full path to the subdirectory to write to. The directory will be
            created if it does not exist.

        filename : string
            Full name of file to write to. The file will be created if it does
            not exist.

        """
        if not os.access(subdir, os.W_OK):
            os.mkdir(subdir)
        this_file = os.path.join(subdir, filename)

        with h5py.File(this_file, 'a') as f:
            existing_samples = f.keys()
            for index, sample in enumerate(sample_list):
                if str(sample) in existing_samples:
                    errstr = (
                        'sample %i already in data - no overwriting allowed'
                    )
                    raise IOError(errstr % sample)
                grp = f.create_group(str(sample))
                # reset list of data_dict called already
                self._dict_list = []
                self._write_dict(grp, data_dict, index_list[index], sample_len)

    def _write_dict(self, grp, this_dict, sample_index, sample_len):
        """Write metadata for a single sample index from a dictionary.

        If the dictionary contains other dictionaries as keys, this method
        will be recursively called on those dictionaries to write their
        entries.


        Parameters
        ----------

        grp : h5py.Group object
            HDF5 group in which to write all non-dict values in `this_dict`.

        this_dict : dict
            A dictionary representing the metadata to write with keys giving
            the field names and values giving the metadata itself. Each value
            must be either:

                - a 1-D numpy array or list/tuple of numpy objects of length
                    `sample_len`, with data to be written during this call at
                    index `sample_index`
                - a single value or numpy array of shape different than
                    ``(sample_len,)``, in which case `sample_index` is ignored
                - another dictionary with keys that are valid Group name
                    strings and leaf values that are one of the above

            This dictionary must always have the same keys each time the write
            method is called.

        sample_index : int
            Index into arrays/lists in `data_dict` for the value that will be
            written.

        sample_len : int
            Total number of samples to be written from this `data_dict`,
            possibly over multiple calls to this method. Used to determine if
            writing numpy arrays item by item or as a whole.

        """
        # prevent loops by checking that this is a unique dict
        if this_dict in self._dict_list:
            errstr = 'Infinite loop in data - dict <%s> passed in twice'
            raise ValueError(errstr % str(this_dict)[0:500])
        self._dict_list.append(this_dict)

        for key in this_dict.keys():
            if isinstance(this_dict[key], types.DictType):
                new_grp = grp.create_group(str(key))
                self._write_dict(
                    new_grp, this_dict[key], sample_index, sample_len,
                )
                data = None
            elif isinstance(this_dict[key], (types.ListType, types.TupleType)):
                data = this_dict[key][sample_index]
            elif hasattr(this_dict[key], 'shape'):
                if this_dict[key].shape == (sample_len,):
                    data = this_dict[key][sample_index]  # value from array
                else:
                    data = this_dict[key]  # whole array
            else:
                data = this_dict[key]  # single numpy value
            if data is not None:
                grp.create_dataset(key, data=data)

    def _get_subdir_filename_info(self, samples):
        """Group sample indices into their appropriate files.

        This takes a list of samples and breaks it into groups belonging in
        single files. Those groups are returned along with the path and name
        of the corresponding file.


        Parameters
        ----------

        samples : 1-D numpy array of type uint64
            An array of sample indices, given in the number of samples since
            the epoch (time_since_epoch*sample_rate).


        Returns
        -------

        list of tuples
            A list of tuples, where each tuple corresponds to one file and has
            the following components:

            sample_list : list
                List of sample indices from `samples` included in the file.
            index_list : list
                List corresponding to `sample_list` that gives the indices into
                `samples` so that ``samples[index_list] == sample_list``.
            subdir : string
                Full path to the subdirectory containing the file.
            filename : string
                Full name of file.

        """
        samples_per_file = self._file_cadence_secs * self._samples_per_second
        # floor using uint64
        file_indices = numpy.uint64(samples / samples_per_file)
        ret_list = []
        for file_idx in numpy.unique(file_indices):
            idxs = (file_indices == file_idx)
            index_list = list(idxs.nonzero()[0])
            sample_list = list(samples[idxs])

            file_ts = file_idx * self._file_cadence_secs
            file_basename = '%s@%i.h5' % (self._file_name, file_ts)

            start_sub_ts = (
                (file_ts//self._subdir_cadence_secs)*self._subdir_cadence_secs
            )
            sub_dt = datetime.datetime.utcfromtimestamp(start_sub_ts)
            subdir = os.path.join(
                self._metadata_dir, sub_dt.strftime('%Y-%m-%dT%H-%M-%S'),
            )

            ret_list.append(
                (sample_list, index_list, subdir, file_basename)
            )

        return(ret_list)

    def _set_fields(self, field_names):
        """Set the field names used in this metadata channel.

        This method sets both the `_fields` attribute and writes the field
        names to the channels top-level 'dmd_properties.h5' file.


        Parameters
        ----------

        field_names : list
            List of field names used in this metadata channel.

        """
        # build recarray and self._fields
        recarray = numpy.recarray(
            (len(field_names),), dtype=[('column', '|S128')],
        )
        self._fields = field_names
        self._fields.sort()  # for reproducability, use alphabetic order
        for i, key in enumerate(self._fields):
            recarray[i] = (key,)

        # write recarray to metadata
        properties_file_path = os.path.join(
            self._metadata_dir, 'dmd_properties.h5',
        )
        with h5py.File(properties_file_path, 'a') as f:
            f.create_dataset('fields', data=recarray)

    def _parse_properties(self):
        """Check writer properties against existing ones for the channel.

        When a metadata channel already exists on disk, call this method when
        creating a new DigitalMetadataWriter to check its parameters against
        the existing ones. The `fields` attribute and metadata version of the
        current writer will be set according to parameters found on disk.

        Raises
        ------

        ValueError
            If the DigitalMetadataWriter parameters do not match those on disk.

        IOError
            If the Digital Metadata version of the existing metadata on disk
            is not compatible with this software version.

        """
        # reader will raise IOError if existing properties can't be read
        # because of version incompatibilities
        org_obj = DigitalMetadataReader(self._metadata_dir, accept_empty=True)
        # but we also have to check if the metadata is the current major
        # version so we know we can continue writing to it
        self._digital_metadata_version = org_obj._digital_metadata_version
        self._check_compatible_version()
        attr_list = (
            '_subdir_cadence_secs', '_file_cadence_secs',
            '_sample_rate_numerator',  '_sample_rate_denominator',
            '_file_name',
        )
        for attr in attr_list:
            if getattr(self, attr) != getattr(org_obj, attr):
                errstr = 'Mismatched %s: %s versus %s'
                raise ValueError(errstr % (attr,
                                           getattr(self, attr),
                                           getattr(org_obj, attr)))
        self._fields = getattr(org_obj, '_fields')

    def _check_compatible_version(self):
        version = StrictVersion(self._digital_metadata_version)

        if (version >= self._min_version) and (version <= self._max_version):
            pass
        else:
            errstr = (
                'Digital Metadata files being read are version %s, which is'
                'not in the range required (%s to %s).'
            )
            raise IOError(errstr % (str(version),
                                    str(self._min_version),
                                    str(self._max_version)))

    def _write_properties(self):
        """Write Digital Metadata properties to dmd_properties.h5 file."""
        properties_file_path = os.path.join(
            self._metadata_dir, 'dmd_properties.h5',
        )
        with h5py.File(properties_file_path, 'w') as f:
            f.attrs['subdir_cadence_secs'] = self._subdir_cadence_secs
            f.attrs['file_cadence_secs'] = self._file_cadence_secs
            f.attrs['sample_rate_numerator'] = self._sample_rate_numerator
            f.attrs['sample_rate_denominator'] = self._sample_rate_denominator
            f.attrs['file_name'] = self._file_name
            f.attrs['digital_metadata_version'] = \
                self._digital_metadata_version

    def __str__(self):
        """String summary of the DigitalMetadataWriter's parameters."""
        ret_str = ''
        attr_list = (
            '_subdir_cadence_secs', '_file_cadence_secs',
            '_samples_per_second', '_file_name',
        )
        for attr in attr_list:
            ret_str += '%s: %s\n' % (attr, str(getattr(self, attr)))
        if self._fields is None:
            ret_str += '_fields: None\n'
        else:
            ret_str += '_fields:\n'
            for key in self._fields:
                ret_str += '\t%s\n' % (key)
        return(ret_str)


class DigitalMetadataReader:
    """Read data in Digital Metadata HDF5 format."""

    _min_version = StrictVersion('2.0')

    def __init__(self, metadata_dir, accept_empty=True):
        """Initialize reader to metadata channel directory.

        Channel parameters are read from the attributes of the top-level file
        'dmd_properties.h5' in the `metadata_dir`.


        Parameters
        ----------

        metadata_dir : string
            Path to metadata channel directory, which contains a
            'dmd_properties.h5' file and timestamped subdirectories containing
            data.

        accept_empty : bool, optional
            If True, do not raise an IOError if the 'dmd_properties.h5' file is
            empty. If False, raise an IOError in that case and delete the
            empty 'dmd_properties.h5' file.


        Raises
        ------

        IOError
            If 'dmd_properties.h5' file is not found in `metadata_dir` or if
            `accept_empty` is False and the 'dmd_properties.h5' file is empty.

        """
        self._metadata_dir = metadata_dir
        if self._metadata_dir.find('http://') != -1:
            self._local = False
            # put properties file in /tmp/dmd_properties_%i.h5 % (pid)
            url = os.path.join(self._metadata_dir, 'dmd_properties.h5')
            try:
                f = urllib2.urlopen(url)
            except (urllib2.URLError, urllib2.HTTPError):
                url = os.path.join(self._metadata_dir, 'metadata.h5')
                f = urllib2.urlopen(url)
            tmp_file = os.path.join(
                '/tmp', 'dmd_properties_%i.h5' % (os.getpid()),
            )
            fo = open(tmp_file, 'w')
            fo.write(f.read())
            f.close()
            fo.close()

        else:
            self._local = True
            # list and match first properties file
            tmp_file = next(
                (f for f in glob.glob(os.path.join(
                    metadata_dir, list_drf.GLOB_DMDPROPFILE,
                )) if re.match(list_drf.RE_DMDPROP, f)),
                None,
            )
            if tmp_file is None:
                raise IOError('dmd_properties.h5 not found')

        with h5py.File(tmp_file, 'r') as f:
            try:
                subdir_cadence = f.attrs['subdir_cadence_secs'].item()
                file_cadence = f.attrs['file_cadence_secs'].item()
            except KeyError:
                # maybe an older version with subdirectory_cadence_seconds
                # and file_cadence_seconds
                subdir_cadence = f.attrs['subdirectory_cadence_seconds'].item()
                file_cadence = f.attrs['file_cadence_seconds'].item()
            self._subdir_cadence_secs = subdir_cadence
            self._file_cadence_secs = file_cadence
            try:
                try:
                    spsn = f.attrs['sample_rate_numerator'].item()
                    spsd = f.attrs['sample_rate_denominator'].item()
                except KeyError:
                    # maybe an older version with samples_per_second_*
                    spsn = f.attrs['samples_per_second_numerator'].item()
                    spsd = f.attrs['samples_per_second_denominator'].item()
            except KeyError:
                # must have an older version with samples_per_second attribute
                sps = f.attrs['samples_per_second'].item()
                spsfrac = fractions.Fraction(sps).limit_denominator()
                self._samples_per_second = numpy.longdouble(sps)
                self._sample_rate_numerator = long(spsfrac.numerator)
                self._sample_rate_denominator = long(spsfrac.denominator)
            else:
                self._sample_rate_numerator = spsn
                self._sample_rate_denominator = spsd
                # have to go to uint64 before longdouble to ensure correct
                # conversion from long
                self._samples_per_second = (
                    numpy.longdouble(numpy.uint64(
                        self._sample_rate_numerator
                    )) /
                    numpy.longdouble(numpy.uint64(
                        self._sample_rate_denominator
                    ))
                )
            self._file_name = f.attrs['file_name']
            try:
                version = f.attrs['digital_metadata_version']
            except KeyError:
                # version is before 2.3 when attribute was added
                version = '2.0'
            self._digital_metadata_version = version
            self._check_compatible_version()
            try:
                fields_dataset = f['fields']
            except KeyError:
                if not accept_empty:
                    os.remove(tmp_file)
                    errstr = (
                        'No metadata yet written to %s, removing empty'
                        ' "dmd_properties.h5"'
                    )
                    raise IOError(errstr % self._metadata_dir)
                else:
                    self._fields = None
                    return
            self._fields = []
            for i in range(len(fields_dataset)):
                self._fields.append(fields_dataset[i]['column'])

        if not self._local:
            os.remove(tmp_file)

    def get_bounds(self):
        """Get indices of first- and last-known sample as a tuple.

        Returns
        -------

        first_sample_index : long
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : long
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).


        Raises
        ------

        IOError
            If no data or first and last sample could not be determined.

        """
        # get subdirectory list
        subdir_path_glob = os.path.join(
            self._metadata_dir, list_drf.GLOB_SUBDIR,
        )
        subdir_list = glob.glob(subdir_path_glob)
        subdir_list.sort()
        if len(subdir_list) == 0:
            errstr = 'glob returned no directories for %s'
            raise IOError(errstr % subdir_path_glob)

        filename_glob = list_drf.GLOB_DMDFILE.replace('*', self._file_name, 1)
        first_sample = None
        # try first three subdirectories in case of subdirectory deletion
        for subdir in subdir_list[:3]:
            file_list = glob.glob(os.path.join(subdir, filename_glob))
            file_list.sort(key=list_drf.sortkey_drf)
            # try first three files in case of file deletion occuring
            for filepath in file_list[:3]:
                try:
                    with h5py.File(filepath, 'r') as f:
                        groups = f.keys()
                        groups.sort()
                        first_sample = long(groups[0])
                except (IOError, IndexError):
                    continue
                else:
                    break
            if first_sample is not None:
                break
        if first_sample is None:
            raise IOError('All attempts to read first sample failed')

        # now try to get last_file
        last_file = None
        for subdir in reversed(subdir_list[-20:]):
            file_list = glob.glob(os.path.join(subdir, filename_glob))
            file_list.sort(key=list_drf.sortkey_drf)
            # try last files
            for filepath in reversed(file_list[-20:]):
                if os.path.getsize(filepath) == 0:
                    continue
                last_file = filepath
                try:
                    with h5py.File(last_file, 'r') as f:
                        groups = f.keys()
                        groups.sort()
                        last_sample = long(groups[-1])
                except (IOError, IndexError):
                    # delete as corrupt if not too new
                    if not (time.time() - os.path.getmtime(last_file) <
                            self._file_cadence_secs):
                        # probable corrupt file
                        traceback.print_exc()
                        errstr = 'WARNING: removing %s since may be corrupt'
                        print(errstr % last_file)
                        os.remove(last_file)
                    last_file = None
                else:
                    return((first_sample, last_sample))

        if last_file is None:
            raise IOError('All attempts to read last file failed')

    def get_fields(self):
        """Return list of the field names in this metadata."""
        # _fields is an internal data structure, so make a copy for the user
        return(copy.deepcopy(self._fields))

    def get_sample_rate_numerator(self):
        """Return the numerator of the sample rate in Hz."""
        return(self._sample_rate_numerator)

    def get_sample_rate_denominator(self):
        """Return the denominator of the sample rate in Hz."""
        return(self._sample_rate_denominator)

    def get_samples_per_second(self):
        """Return the sample rate in Hz as a numpy.longdouble."""
        return(self._samples_per_second)

    def get_subdir_cadence_secs(self):
        """Return the number of seconds of data stored in each subdirectory."""
        return(self._subdir_cadence_secs)

    def get_file_cadence_secs(self):
        """Return the number of seconds of data stored in each file."""
        return(self._file_cadence_secs)

    def get_file_name_prefix(self):
        """Return the metadata file name prefix."""
        return(self._file_name)

    def read(self, start_sample, end_sample, columns=None):
        """Read metadata between start and end samples.

        Parameters
        ----------

        start_sample : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        end_sample : long
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.
            Using a string results in a different returned object than a one-
            element list containing the string, see below.


        Returns
        -------

        OrderedDict
            The dictionary's keys are the sample index for each sample of
            metadata found between `start_sample` and `end_sample` (inclusive).
            Each value is a metadata sample, given as either the column value
            (if `columns` is a string) or a dictionary with column names as
            keys and numpy objects as leaf values (if `columns` is None or
            a list).

        """
        if start_sample > end_sample:
            errstr = 'Start sample %i more than end sample %i'
            raise ValueError(errstr % (start_sample, end_sample))

        file_list = self._get_file_list(start_sample, end_sample)
        ret_dict = collections.OrderedDict()
        for this_file in file_list:
            if this_file in (file_list[0], file_list[-1]):
                is_edge = True
            else:
                is_edge = False
            self._add_metadata(
                ret_dict, this_file, columns, start_sample, end_sample,
                is_edge,
            )

        return(ret_dict)

    def read_latest(self):
        """Read the most recent metadata sample.

        This method calls `get_bounds` to find the last sample index and `read`
        to read the metadata near the last sample. It returns all columns of
        the metadata found at the latest sample.


        Returns
        -------

        dict
            Dictionary containing the latest metadata, where the key is the
            sample index and the value is the metadata sample itself given as
            a dictionary with column names as keys and numpy objects as leaf
            values.

        """
        start_sample, last_sample = self.get_bounds()
        dict = self.read(
            long(last_sample - 2*self._samples_per_second), last_sample,
        )
        keys = dict.keys()
        if len(keys) == 0:
            raise IOError('Unable to find metadata near the last sample')
        keys.sort()
        ret_dict = {}
        ret_dict[keys[-1]] = dict[keys[-1]]
        return(ret_dict)

    def _get_file_list(self, sample0, sample1):
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


        Returns
        -------

        list
            List of file paths that exist on disk, fall in the given time
            interval, and conform to the subdirectory and file cadence naming
            scheme.

        """
        # need to go through numpy uint64 to prevent conversion to float
        start_ts = long(numpy.uint64(sample0/self._samples_per_second))
        end_ts = long(numpy.uint64(sample1/self._samples_per_second))

        # convert ts to be divisible by self._file_cadence_secs
        start_ts = \
            (start_ts // self._file_cadence_secs) * self._file_cadence_secs
        end_ts = (end_ts // self._file_cadence_secs) * self._file_cadence_secs

        # get subdirectory start and end ts
        start_sub_ts = \
            (start_ts // self._subdir_cadence_secs) * self._subdir_cadence_secs
        end_sub_ts = \
            (end_ts // self._subdir_cadence_secs) * self._subdir_cadence_secs

        ret_list = []  # ordered list of full file paths to return

        for sub_ts in range(
            start_sub_ts,
            end_sub_ts + self._subdir_cadence_secs,
            self._subdir_cadence_secs,
        ):
            sub_datetime = datetime.datetime.utcfromtimestamp(sub_ts)
            subdir = sub_datetime.strftime('%Y-%m-%dT%H-%M-%S')
            # create numpy array of all file TS in subdir
            file_ts_in_subdir = numpy.arange(
                sub_ts,
                sub_ts + self._subdir_cadence_secs,
                self._file_cadence_secs,
            )
            # file has valid samples if last time in file is after start time
            # and first time in file is before end time
            valid_in_subdir = numpy.logical_and(
                file_ts_in_subdir + self._file_cadence_secs - 1 >= start_ts,
                file_ts_in_subdir <= end_ts,
            )
            valid_file_ts_list = numpy.compress(
                valid_in_subdir,
                file_ts_in_subdir,
            )
            for valid_file_ts in valid_file_ts_list:
                file_basename = '%s@%i.h5' % (self._file_name, valid_file_ts)
                full_file = os.path.join(
                    self._metadata_dir, subdir, file_basename,
                )
                # verify exists
                if not os.access(full_file, os.R_OK):
                    continue
                ret_list.append(full_file)

        return(ret_list)

    def _add_metadata(
        self, ret_dict, this_file, columns, sample0, sample1, is_edge,
    ):
        """Read metadata from a single file and add it to `ret_dict`.

        Parameters
        ----------

        ret_dict : OrderedDict
            Dictionary to which metadata will be added.

        this_file : string
            Full path to the file from which metadata will be read.

        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.

        sample0 : long
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        sample1 : long
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        is_edge : bool
            If True, then this is the first or last file in a sequence spanning
            indices from `sample0` to `sample1` and those read boundaries must
            be taken into account. If False, all samples from the file will be
            read ignoring `sample0` and `sample1`.

        """
        try:
            with h5py.File(this_file, 'r') as f:
                # get sorted numpy array of all samples in file
                keys = f.keys()
                idxs = numpy.fromiter(keys, numpy.int64, count=len(keys))
                idxs.sort()
                if is_edge:
                    # calculate valid samples based on sample range
                    valid = numpy.logical_and(idxs >= sample0, idxs <= sample1)
                    idxs = idxs[valid]
                for idx in idxs:
                    value = f[str(idx)]
                    if columns is None:
                        self._populate_data(ret_dict, value, idx)
                    elif isinstance(columns, types.StringTypes):
                        self._populate_data(ret_dict, value[columns], idx)
                    else:
                        ret_dict[idx] = {}
                        for column in columns:
                            self._populate_data(
                                ret_dict[idx], value[column], column,
                            )
        except IOError:
            # decide whether this file is corrupt, or too new, or just missing
            if os.access(this_file, os.R_OK) and os.access(this_file, os.W_OK):
                if (time.time() - os.path.getmtime(this_file) >
                        self._file_cadence_secs):
                    traceback.print_exc()
                    errstr = (
                        'WARNING: %s being deleted because it raised an error'
                        ' and is not new'
                    )
                    print(errstr % this_file)
                    os.remove(this_file)

    def _populate_data(self, ret_dict, obj, name):
        """Read data recursively from an HDF5 value and add it to `ret_dict`.

        If `obj` is a dataset, it is added to `ret_dict`. If `obj` is a group,
        a sub-dictionary is created in `ret_dict` for `obj` and populated
        recursively by calling this function on all of  the items in the `obj`
        group.

        Parameters
        ----------

        ret_dict : OrderedDict
            Dictionary to which metadata will be added.

        obj : h5py.Dataset | h5py.Group
            HDF5 value from which to read metadata.

        name : valid dictionary key
            Dictionary key in `ret_dict` under which to store the data from
            `obj`.

        """
        if isinstance(obj, h5py.Dataset):
            # [()] casts a Dataset as a numpy array
            ret_dict[name] = obj[()]
        else:
            # create a dictionary for this group
            ret_dict[name] = {}
            for key, value in obj.items():
                self._populate_data(ret_dict[name], value, key)

    def _check_compatible_version(self):
        version = StrictVersion(self._digital_metadata_version)

        if version >= self._min_version:
            pass
        else:
            errstr = (
                'Digital Metadata files being read are version %s, which is'
                'less than the required version (%s).'
            )
            raise IOError(errstr % (str(version), str(self._min_version)))

    def __str__(self):
        """String summary of the DigitalMetadataReader's parameters."""
        ret_str = ''
        attr_list = (
            '_subdir_cadence_secs', '_file_cadence_secs',
            '_samples_per_second', '_file_name',
        )
        for attr in attr_list:
            ret_str += '%s: %s\n' % (attr, str(getattr(self, attr)))
        if self._fields is None:
            ret_str += '_fields: None\n'
        else:
            ret_str += '_fields:\n'
            for key in self._fields:
                ret_str += '\t%s\n' % (key)
        return(ret_str)
