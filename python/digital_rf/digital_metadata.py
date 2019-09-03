# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Read and write data/metadata in Digital Metadata HDF5 format.

It uses h5py to read and write to HDF5 files.

Reading/writing functionality is available from two classes:
DigitalMetadataReader and DigitalMetadataWriter.

"""
from __future__ import absolute_import, division, print_function

import collections
import copy
import datetime
import fractions
import glob
import itertools
import os
import re
import time
import traceback
import warnings
from collections import defaultdict

# third party imports
import h5py
import numpy as np
import packaging.version
import six
from six.moves import urllib, zip

# local imports
from . import list_drf
from ._version import get_versions

try:
    import pandas
except ImportError:
    pass

__version__ = get_versions()["version"]
del get_versions

__all__ = ("DigitalMetadataReader", "DigitalMetadataWriter")


# disable file locking in HDF5 >= 1.10 (not present in earlier versions)
# through only way possible: setting an environment variable
# this allows reading and writing metadata using the same file, which should be
# safe since we don't allow multiple or partial writes to the same sample index
# and is something we've allowed in practice with HDF5 1.8 and earlier
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _recursive_items(d, prefix="", visited=None):
    """Generate (key, value) pairs for a dict, recursing into sub-dicts.

    Sub-dictionary (key, value) pairs will have '[parent_key]/' prepended
    to their key name.


    Parameters
    ----------
    d : dict
        The dictionary to iterate over.

    prefix : string
        The starting prefix to be added to keys to produce the returned name.

    visited : set | None
        Set of already visited dictionary ids, to flag infinite recursion.
        If None, the empty set is used.


    Yields
    ------
    (key, val) : tuple
        Key name (with prefix and parent dictionary key added) and value pairs
        for an entry in the dictionary or a sub-dictionary.

    """
    if visited is None:
        visited = set()
    visited.add(id(d))
    for k, v in six.iteritems(d):
        name = prefix + k
        if isinstance(v, dict):
            if id(v) not in visited:
                for subk, subv in _recursive_items(v, name + "/", visited):
                    yield subk, subv
            else:
                errstr = "Infinite loop in data - dict <%s> passed in twice."
                raise ValueError(errstr % str(v)[0:500])
        else:
            yield name, v


class DigitalMetadataWriter(object):
    """Write data in Digital Metadata HDF5 format."""

    _min_version = packaging.version.parse("2.5")
    _max_version = packaging.version.parse(
        packaging.version.parse(__version__).base_version
    )
    # increment to package version when format changes are made
    _writer_version = packaging.version.parse("2.5")

    def __init__(
        self,
        metadata_dir,
        subdir_cadence_secs,
        file_cadence_secs,
        sample_rate_numerator,
        sample_rate_denominator,
        file_name,
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

        sample_rate_numerator : int
            Numerator of sample rate in Hz.

        sample_rate_denominator : int
            Denominator of sample rate in Hz.

        file_name : string
            Prefix for metadata file names. Files in each subdirectory will be
            named: "`file_name`@<timestamp>.h5".

        """
        # verify all input arguments
        if not os.access(metadata_dir, os.W_OK):
            errstr = "metadata_dir %s does not exist or is not writable"
            raise IOError(errstr % metadata_dir)
        self._metadata_dir = metadata_dir

        if subdir_cadence_secs != int(subdir_cadence_secs) or subdir_cadence_secs < 1:
            errstr = "subdir_cadence_secs must be positive integer, not %s"
            raise ValueError(errstr % str(subdir_cadence_secs))
        self._subdir_cadence_secs = int(subdir_cadence_secs)

        if file_cadence_secs != int(file_cadence_secs) or file_cadence_secs < 1:
            errstr = "file_cadence_secs must be positive integer, not %s"
            raise ValueError(errstr % str(file_cadence_secs))
        self._file_cadence_secs = int(file_cadence_secs)
        if (self._subdir_cadence_secs % self._file_cadence_secs) != 0:
            raise ValueError("(subdir_cadence_secs % file_cadence_secs) != 0")

        if not isinstance(file_name, six.string_types):
            errstr = "file_name must be a string, not type %s"
            raise ValueError(errstr % str(type(file_name)))
        self._file_name = file_name

        if (
            sample_rate_numerator != int(sample_rate_numerator)
            or sample_rate_numerator < 1
        ):
            errstr = "sample_rate_numerator must be positive integer, not %s"
            raise ValueError(errstr % str(sample_rate_numerator))
        self._sample_rate_numerator = int(sample_rate_numerator)

        if (
            sample_rate_denominator != int(sample_rate_denominator)
            or sample_rate_denominator < 1
        ):
            errstr = "sample_rate_denominator must be positive integer, not %s"
            raise ValueError(errstr % str(sample_rate_denominator))
        self._sample_rate_denominator = int(sample_rate_denominator)

        # have to go to uint64 before longdouble to ensure correct conversion
        # from int
        self._samples_per_second = np.longdouble(
            np.uint64(self._sample_rate_numerator)
        ) / np.longdouble(np.uint64(self._sample_rate_denominator))

        if os.access(
            os.path.join(self._metadata_dir, "dmd_properties.h5"), os.R_OK
        ) or os.access(os.path.join(self._metadata_dir, "metadata.h5"), os.R_OK):
            self._parse_properties()
        else:
            self._digital_metadata_version = self._writer_version.base_version
            self._fields = None  # No data written yet
            self._write_properties()

    def get_samples_per_second(self):
        """Return the sample rate in Hz as a np.longdouble."""
        return self._samples_per_second

    def write(self, samples, data):
        """Write new metadata to the Digital Metadata channel.

        Parameters
        ----------
        samples : list | 1-D array | int | float
            A single sample index or an list of sample indices, given in
            the number of samples since the epoch (t_since_epoch*sample_rate),
            for the data to be written.

        data : list of dicts | dict
            If a list of dicts, each dictionary provides the metadata to be
            written for each corresponding sample (`data` must have the same
            length as `samples`). The dictionary keys give the field names,
            while the values must be HDF5-compatible (numpy) objects or sub-
            dictionaries meeting the same requirement.

            If a dict, the keys give the field names and each value must be
            one of the following:

                - a 1-D numpy array or list/tuple of numpy objects with length
                    equal to ``len(samples)`` giving the metadata corresponding
                    to each sample index in `samples`
                - a single value or numpy array of shape different than
                    ``(len(samples),)``, giving the metadata for *all* of the
                    samples indices in `samples`
                - another dictionary with keys that are valid Group name
                    strings and leaf values that are one of the above

            The fields should always be the same each time the write method is
            called to ensure that the fields are consistently present when
            reading.

        """
        try:
            samples = np.array(samples, dtype=np.uint64, copy=False, ndmin=1)
        except (TypeError, ValueError):
            raise ValueError("Values in `samples` must be convertible to uint64")

        N = len(samples)
        if N == 0:
            raise ValueError("`samples` must not be empty")

        if isinstance(data, dict):
            if self._fields is None:
                self._set_fields(list(data.keys()))

            keyval_iterators = []
            for key, val in _recursive_items(data):
                if not isinstance(val, six.string_types):
                    try:
                        if len(val) == N:
                            it = zip(itertools.repeat(key, N), val)
                            keyval_iterators.append(it)
                            continue
                    except TypeError:
                        pass
                # val is a string, doesn't have a length, or len(val) != N
                it = itertools.repeat((key, val), N)
                keyval_iterators.append(it)
            keyvals = zip(*keyval_iterators)
        elif len(data) == N:
            if self._fields is None:
                self._set_fields(list(data[0].keys()))

            keyvals = (_recursive_items(d) for d in data)
        else:
            errstr = (
                "`data` must be a dict or list of dicts with length equal to"
                " the length of `samples`."
            )
            raise ValueError(errstr)

        return self._write(samples, keyvals)

    def _write(self, samples, keyvals):
        """Write new metadata to the Digital Metadata channel.

        This function does no input checking, see `write` for that.


        Parameters
        ----------
        samples : 1-D numpy array of type uint64 sorted in ascending order
            An array of sample indices, given in the number of samples since
            the epoch (time_since_epoch*sample_rate).

        keyvals : iterable of iterables same length as `samples`
            Each element of this iterable corresponds to a sample in `samples`
            and should be another iterable that produces (key, value) pairs to
            write for that sample.

        """
        grp_iter = self._sample_group_generator(samples)
        for grp, keyval in zip(grp_iter, keyvals):
            for key, val in keyval:
                if val is not None:
                    grp.create_dataset(key, data=val)
                else:
                    # treat None as the empty string so there will always
                    # be a dataset written when it is passed to write
                    grp.create_dataset(key, data="")

    def _sample_group_generator(self, samples):
        """Yield HDF5 group for each sample in `samples`.

        Parameters
        ----------
        samples : 1-D numpy array of type uint64 sorted in ascending order
            An array of sample indices, given in the number of samples since
            the epoch (time_since_epoch*sample_rate).


        Yields
        ------
        grp : h5py.Group
            HDF5 group for the sample. The group is located in the appropriate
            Digital Metadata file and takes its name from the sample index.

        """
        samples_per_file = self._file_cadence_secs * self._samples_per_second
        for file_idx, sample_group in itertools.groupby(
            samples, lambda s: np.uint64(s / samples_per_file)
        ):
            file_ts = file_idx * self._file_cadence_secs
            file_basename = "%s@%i.h5" % (self._file_name, file_ts)

            start_sub_ts = (
                file_ts // self._subdir_cadence_secs
            ) * self._subdir_cadence_secs
            sub_dt = datetime.datetime.utcfromtimestamp(start_sub_ts)
            subdir = os.path.join(
                self._metadata_dir, sub_dt.strftime("%Y-%m-%dT%H-%M-%S")
            )

            if not os.path.exists(subdir):
                os.makedirs(subdir)
            this_file = os.path.join(subdir, file_basename)

            with h5py.File(this_file, "a") as f:
                for sample in sample_group:
                    try:
                        grp = f.create_group(str(sample))
                    except ValueError:
                        errstr = "Sample %i already in data: no overwriting allowed"
                        raise IOError(errstr % sample)
                    yield grp

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
        recarray = np.recarray((len(field_names),), dtype=[("column", "|S128")])
        self._fields = field_names
        self._fields.sort()  # for reproducability, use alphabetic order
        for i, key in enumerate(self._fields):
            recarray[i] = (key,)

        # write recarray to metadata
        properties_file_path = os.path.join(self._metadata_dir, "dmd_properties.h5")
        with h5py.File(properties_file_path, "a") as f:
            f.create_dataset("fields", data=recarray)

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
            "_subdir_cadence_secs",
            "_file_cadence_secs",
            "_sample_rate_numerator",
            "_sample_rate_denominator",
            "_file_name",
        )
        for attr in attr_list:
            if getattr(self, attr) != getattr(org_obj, attr):
                errstr = "Mismatched %s: %s versus %s"
                raise ValueError(
                    errstr % (attr, getattr(self, attr), getattr(org_obj, attr))
                )
        self._fields = org_obj._fields

    def _check_compatible_version(self):
        version = packaging.version.parse(self._digital_metadata_version)

        if (version >= self._min_version) and (version <= self._max_version):
            pass
        else:
            errstr = (
                "This existing Digital Metadata files are version %s, which is"
                " not in the range required (%s to %s)."
            )
            raise IOError(
                errstr
                % (
                    version.base_version,
                    self._min_version.base_version,
                    self._max_version.base_version,
                )
            )

    def _write_properties(self):
        """Write Digital Metadata properties to dmd_properties.h5 file."""
        properties_file_path = os.path.join(self._metadata_dir, "dmd_properties.h5")
        with h5py.File(properties_file_path, "w") as f:
            f.attrs["subdir_cadence_secs"] = self._subdir_cadence_secs
            f.attrs["file_cadence_secs"] = self._file_cadence_secs
            f.attrs["sample_rate_numerator"] = self._sample_rate_numerator
            f.attrs["sample_rate_denominator"] = self._sample_rate_denominator
            # use np.string_ to store as fixed-length ascii strings
            f.attrs["file_name"] = np.string_(self._file_name)
            f.attrs["digital_metadata_version"] = np.string_(
                self._digital_metadata_version
            )

    def __str__(self):
        """String summary of the DigitalMetadataWriter's parameters."""
        ret_str = ""
        attr_list = (
            "_subdir_cadence_secs",
            "_file_cadence_secs",
            "_samples_per_second",
            "_file_name",
        )
        for attr in attr_list:
            ret_str += "%s: %s\n" % (attr, str(getattr(self, attr)))
        if self._fields is None:
            ret_str += "_fields: None\n"
        else:
            ret_str += "_fields:\n"
            for key in self._fields:
                ret_str += "\t%s\n" % (key)
        return ret_str


class DigitalMetadataReader(object):
    """Read data in Digital Metadata HDF5 format."""

    _min_version = packaging.version.parse("2.0")
    _max_version = packaging.version.parse(
        packaging.version.parse(__version__).base_version
    )

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
        if self._metadata_dir.find("http://") != -1:
            self._local = False
            # put properties file in /tmp/dmd_properties_%i.h5 % (pid)
            url = os.path.join(self._metadata_dir, "dmd_properties.h5")
            try:
                f = urllib.request.urlopen(url)
            except (urllib.error.URLError, urllib.error.HTTPError):
                url = os.path.join(self._metadata_dir, "metadata.h5")
                f = urllib.request.urlopen(url)
            tmp_file = os.path.join("/tmp", "dmd_properties_%i.h5" % (os.getpid()))
            fo = open(tmp_file, "w")
            fo.write(f.read())
            f.close()
            fo.close()

        else:
            self._local = True
            # list and match first properties file
            tmp_file = next(
                (
                    f
                    for f in sorted(
                        glob.glob(os.path.join(metadata_dir, list_drf.GLOB_DMDPROPFILE))
                    )
                    if re.match(list_drf.RE_DMDPROP, f)
                ),
                None,
            )
            if tmp_file is None:
                raise IOError("dmd_properties.h5 not found")

        with h5py.File(tmp_file, "r") as f:
            try:
                subdir_cadence = f.attrs["subdir_cadence_secs"].item()
                file_cadence = f.attrs["file_cadence_secs"].item()
            except KeyError:
                # maybe an older version with subdirectory_cadence_seconds
                # and file_cadence_seconds
                subdir_cadence = f.attrs["subdirectory_cadence_seconds"].item()
                file_cadence = f.attrs["file_cadence_seconds"].item()
            self._subdir_cadence_secs = subdir_cadence
            self._file_cadence_secs = file_cadence
            try:
                try:
                    spsn = f.attrs["sample_rate_numerator"].item()
                    spsd = f.attrs["sample_rate_denominator"].item()
                except KeyError:
                    # maybe an older version with samples_per_second_*
                    spsn = f.attrs["samples_per_second_numerator"].item()
                    spsd = f.attrs["samples_per_second_denominator"].item()
            except KeyError:
                # must have an older version with samples_per_second attribute
                sps = f.attrs["samples_per_second"].item()
                spsfrac = fractions.Fraction(sps).limit_denominator()
                self._samples_per_second = np.longdouble(sps)
                self._sample_rate_numerator = int(spsfrac.numerator)
                self._sample_rate_denominator = int(spsfrac.denominator)
            else:
                self._sample_rate_numerator = spsn
                self._sample_rate_denominator = spsd
                # have to go to uint64 before longdouble to ensure correct
                # conversion from int
                self._samples_per_second = np.longdouble(
                    np.uint64(self._sample_rate_numerator)
                ) / np.longdouble(np.uint64(self._sample_rate_denominator))
            fname = f.attrs["file_name"]
            if isinstance(fname, bytes):
                # for convenience and forward-compatibility with h5py>=2.9
                fname = fname.decode("ascii")
            self._file_name = fname
            try:
                version = f.attrs["digital_metadata_version"]
            except KeyError:
                # version is before 2.3 when attribute was added
                version = "2.0"
            else:
                if isinstance(version, bytes):
                    # for convenience and forward-compatibility with h5py>=2.9
                    version = version.decode("ascii")
            self._digital_metadata_version = version
            self._check_compatible_version()
            try:
                fields_dataset = f["fields"]
            except KeyError:
                if not accept_empty:
                    os.remove(tmp_file)
                    errstr = (
                        "No metadata yet written to %s, removing empty"
                        ' "dmd_properties.h5"'
                    )
                    raise IOError(errstr % self._metadata_dir)
                else:
                    self._fields = None
                    return
            self._fields = []
            for i in range(len(fields_dataset)):
                field = fields_dataset[i]["column"]
                if isinstance(field, bytes):
                    # for convenience and forward-compatibility with h5py>=2.9
                    field = field.decode("ascii")
                self._fields.append(field)

        if not self._local:
            os.remove(tmp_file)

    def get_bounds(self):
        """Get indices of first- and last-known sample as a tuple.

        Returns
        -------
        first_sample_index : int
            Index of the first sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).

        last_sample_index : int
            Index of the last sample, given in the number of samples since the
            epoch (time_since_epoch*sample_rate).


        Raises
        ------
        IOError
            If no data or first and last sample could not be determined.

        """
        # loop through files in order to get first sample
        first_sample = None
        for path in list_drf.ilsdrf(
            self._metadata_dir,
            recursive=False,
            reverse=False,
            include_dmd=True,
            include_drf=False,
            include_dmd_properties=False,
        ):
            try:
                with h5py.File(path, "r") as f:
                    groups = list(f.keys())
                    groups.sort()
                    first_sample = int(groups[0])
            except IOError:
                # can't open file (e.g. doesn't exist anymore)
                continue
            except IndexError:
                errstr = (
                    "Corrupt or empty file %s found and ignored."
                    " Deleting it will speed up get_bounds()."
                )
                print(errstr % path)
                continue
            else:
                break
        if first_sample is None:
            raise IOError("All attempts to read first sample failed")

        # loop through files in reverse order to get last sample
        last_sample = None
        for path in list_drf.ilsdrf(
            self._metadata_dir,
            recursive=False,
            reverse=True,
            include_dmd=True,
            include_drf=False,
            include_dmd_properties=False,
        ):
            try:
                with h5py.File(path, "r") as f:
                    groups = list(f.keys())
                    groups.sort()
                    last_sample = int(groups[-1])
            except IOError:
                # can't open file (e.g. doesn't exist anymore)
                continue
            except IndexError:
                errstr = (
                    "Corrupt or empty file %s found and ignored."
                    " Deleting it will speed up get_bounds()."
                )
                print(errstr % path)
                continue
            else:
                break
        if last_sample is None:
            raise IOError("All attempts to read last sample failed")

        return (first_sample, last_sample)

    def get_fields(self):
        """Return list of the field names in this metadata."""
        # _fields is an internal data structure, so make a copy for the user
        return copy.deepcopy(self._fields)

    def get_sample_rate_numerator(self):
        """Return the numerator of the sample rate in Hz."""
        return self._sample_rate_numerator

    def get_sample_rate_denominator(self):
        """Return the denominator of the sample rate in Hz."""
        return self._sample_rate_denominator

    def get_samples_per_second(self):
        """Return the sample rate in Hz as a np.longdouble."""
        return self._samples_per_second

    def get_subdir_cadence_secs(self):
        """Return the number of seconds of data stored in each subdirectory."""
        return self._subdir_cadence_secs

    def get_file_cadence_secs(self):
        """Return the number of seconds of data stored in each file."""
        return self._file_cadence_secs

    def get_file_name_prefix(self):
        """Return the metadata file name prefix."""
        return self._file_name

    def read(self, start_sample=None, end_sample=None, columns=None, method=None):
        """Read metadata between start and end samples.

        Parameters
        ----------
        start_sample : None | int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate). If None,
            `get_bounds` is called and the last sample is used.

        end_sample : None | int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate). If None,
            use `end_sample` equal to `start_sample`.

        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.
            Using a string results in a different returned object than a one-
            element list containing the string, see below.

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
            Each value is a metadata sample, given as either the column value
            (if `columns` is a string) or a dictionary with column names as
            keys and numpy objects as leaf values (if `columns` is None or
            a list).


        See Also
        --------
        read_dataframe : Read metadata into a DataFrame.
        read_flatdict : Read metadata into a flat dictionary, keyed by field.

        """
        if start_sample is None:
            _, start_sample = self.get_bounds()
        if end_sample is None:
            end_sample = start_sample
        elif start_sample > end_sample:
            errstr = "Start sample %i more than end sample %i"
            raise ValueError(errstr % (start_sample, end_sample))

        ret_dict = collections.OrderedDict()

        if method in ("pad", "ffill"):
            # simple forward fill until something better is needed:
            #  get start bound of data and search for metadata within
            #  [start_bound, start_sample] until last sample is found
            ffill_dict = collections.OrderedDict()
            start_bound, end_bound = self.get_bounds()
            file_list = self._get_file_list(start_bound, start_sample)
            # go through files in reverse to break at last found sample
            for this_file in reversed(file_list):
                self._add_metadata(
                    ffill_dict,
                    this_file,
                    columns,
                    start_bound,
                    start_sample,
                    is_edge=False,
                )
                if ffill_dict:
                    # get last entry of ffill_dict which will be latest found
                    # sample in the file
                    key = next(reversed(ffill_dict))
                    ret_dict[key] = ffill_dict[key]
                    break
            # increment start sample so we don't re-add any data at that sample
            start_sample += 1

        file_list = self._get_file_list(start_sample, end_sample)
        for this_file in file_list:
            if this_file in (file_list[0], file_list[-1]):
                is_edge = True
            else:
                is_edge = False
            self._add_metadata(
                ret_dict, this_file, columns, start_sample, end_sample, is_edge
            )

        return ret_dict

    def read_dataframe(
        self, start_sample=None, end_sample=None, columns=None, method=None
    ):
        """Read metadata between start and end samples into a pandas DataFrame.

        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate). If None,
            `get_bounds` is called and the last sample is used.

        end_sample : None | int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate). If None,
            use `end_sample` equal to `start_sample`.

        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.

        method : None | 'pad'/'ffill'
            If None, return only samples within the given range. If 'pad' or
            'ffill', the first sample no later than `start_sample` (if any)
            will also be included so that values are forward filled into the
            desired range.


        Returns
        -------
        DataFrame
            Pandas DataFrame with rows corresponding to the sample index and
            columns corresponding to the metadata key.


        See Also
        --------
        read : Read metadata into an OrderedDict, keyed by sample index.
        read_flatdict : Read metadata into a flat dictionary, keyed by field.

        """
        if isinstance(columns, six.string_types):
            # preserve column name in returned dictionary so it appears in DF
            columns = [columns]
        res = self.read(
            start_sample=start_sample,
            end_sample=end_sample,
            columns=columns,
            method=method,
        )
        data = list(dict(_recursive_items(d)) for d in res.values())
        index = list(res.keys())
        return pandas.DataFrame(data, index=index)

    def read_flatdict(
        self,
        start_sample=None,
        end_sample=None,
        columns=None,
        method=None,
        squeeze=True,
    ):
        """Read metadata between start and end samples into a flat dictionary.

        Parameters
        ----------
        start_sample : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate). If None,
            `get_bounds` is called and the last sample is used.

        end_sample : None | int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate). If None,
            use `end_sample` equal to `start_sample`.

        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.

        method : None | 'pad'/'ffill'
            If None, return only samples within the given range. If 'pad' or
            'ffill', the first sample no later than `start_sample` (if any)
            will also be included so that values are forward filled into the
            desired range.

        squeeze : bool
            If True and end_sample is None (returning a single sample), return
            the column values for the sample directly instead of as arrays with
            a first dimension of one. Additionally, if only a single column
            name is given and the result contains no subfields, return the
            value of that column instead of a dictionary.


        Returns
        -------
        dict or object
            Dictionary with keys corresponding to the fields/columns of the
            requested metadata. The values are arrays with length equal to
            the number of samples. The dictionary also has an 'index' entry
            containing an array of the sample indices. If `squeeze` is True
            and a dictionary with a single sample and non-index column would be
            returned, the non-index value is returned instead.


        See Also
        --------
        read : Read metadata into an OrderedDict, keyed by sample index.
        read_dataframe : Read metadata into a DataFrame.

        """
        if isinstance(columns, six.string_types):
            # preserve column name in returned dictionary so it appears in DF
            columns = [columns]
        res = self.read(
            start_sample=start_sample,
            end_sample=end_sample,
            columns=columns,
            method=method,
        )
        dict_of_lists = defaultdict(lambda: [np.nan] * len(res))
        dict_of_lists[u"index"] = list(res.keys())
        for k, sample_dict in enumerate(res.values()):
            for key, val in _recursive_items(sample_dict):
                dict_of_lists[key][k] = val
        if squeeze and (end_sample is None):
            flatdict = {k: v[0] for k, v in dict_of_lists.items()}
            if len(flatdict) == 2:
                del flatdict["index"]
                return flatdict.popitem()[1]
            else:
                return flatdict
        else:
            return {k: np.array(v) for k, v in dict_of_lists.items()}

    def read_latest(self, columns=None):
        """Read the most recent metadata sample.

        This method calls `get_bounds` to find the last sample index and `read`
        to read the latest metadata at or before the last sample.

        Parameters
        ----------
        columns : None | string | list of strings
            A string or list of strings giving the field/column name of
            metadata to return. If None, all available columns will be read.
            Using a string results in a different returned object than a one-
            element list containing the string, see below.


        Returns
        -------
        dict
            Dictionary containing the latest metadata, where the key is the
            sample index and the value is the metadata sample given as either
            the column value (if `columns` is a string) or a dictionary with
            column names as keys and numpy objects as leaf values (if `columns`
            is None or a list).

        """
        start_sample, last_sample = self.get_bounds()
        return self.read(last_sample, columns=columns, method="ffill")

    def _get_file_list(self, sample0, sample1):
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


        Returns
        -------
        list
            List of file paths that exist on disk, fall in the given time
            interval, and conform to the subdirectory and file cadence naming
            scheme.

        """
        # need to go through numpy uint64 to prevent conversion to float
        start_ts = int(np.uint64(sample0 / self._samples_per_second))
        end_ts = int(np.uint64(sample1 / self._samples_per_second))

        # convert ts to be divisible by self._file_cadence_secs
        start_ts = (start_ts // self._file_cadence_secs) * self._file_cadence_secs
        end_ts = (end_ts // self._file_cadence_secs) * self._file_cadence_secs

        # get subdirectory start and end ts
        start_sub_ts = (
            start_ts // self._subdir_cadence_secs
        ) * self._subdir_cadence_secs
        end_sub_ts = (end_ts // self._subdir_cadence_secs) * self._subdir_cadence_secs

        ret_list = []  # ordered list of full file paths to return

        for sub_ts in range(
            start_sub_ts,
            end_sub_ts + self._subdir_cadence_secs,
            self._subdir_cadence_secs,
        ):
            sub_datetime = datetime.datetime.utcfromtimestamp(sub_ts)
            subdir = sub_datetime.strftime("%Y-%m-%dT%H-%M-%S")
            # create numpy array of all file TS in subdir
            file_ts_in_subdir = np.arange(
                sub_ts, sub_ts + self._subdir_cadence_secs, self._file_cadence_secs
            )
            # file has valid samples if last time in file is after start time
            # and first time in file is before end time
            valid_in_subdir = np.logical_and(
                file_ts_in_subdir + self._file_cadence_secs - 1 >= start_ts,
                file_ts_in_subdir <= end_ts,
            )
            valid_file_ts_list = np.compress(valid_in_subdir, file_ts_in_subdir)
            for valid_file_ts in valid_file_ts_list:
                file_basename = "%s@%i.h5" % (self._file_name, valid_file_ts)
                full_file = os.path.join(self._metadata_dir, subdir, file_basename)
                # verify exists
                if not os.access(full_file, os.R_OK):
                    continue
                ret_list.append(full_file)

        return ret_list

    def _add_metadata(self, ret_dict, this_file, columns, sample0, sample1, is_edge):
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

        sample0 : int
            Sample index for start of read, given in the number of samples
            since the epoch (time_since_epoch*sample_rate).

        sample1 : int
            Sample index for end of read (inclusive), given in the number of
            samples since the epoch (time_since_epoch*sample_rate).

        is_edge : bool
            If True, then this is the first or last file in a sequence spanning
            indices from `sample0` to `sample1` and those read boundaries must
            be taken into account. If False, all samples from the file will be
            read ignoring `sample0` and `sample1`.

        """
        try:
            with h5py.File(this_file, "r") as f:
                # get sorted numpy array of all samples in file
                keys = list(f.keys())
                idxs = np.fromiter(keys, np.int64, count=len(keys))
                idxs.sort()
                if is_edge:
                    # calculate valid samples based on sample range
                    valid = np.logical_and(idxs >= sample0, idxs <= sample1)
                    idxs = idxs[valid]
                for idx in idxs:
                    value = f[str(idx)]
                    if columns is None:
                        self._populate_data(ret_dict, value, idx)
                    elif isinstance(columns, six.string_types):
                        self._populate_data(ret_dict, value[columns], idx)
                    else:
                        ret_dict[idx] = {}
                        for column in columns:
                            self._populate_data(ret_dict[idx], value[column], column)
        except IOError:
            # decide whether this file is corrupt, or too new, or just missing
            if os.access(this_file, os.R_OK) and os.access(this_file, os.W_OK):
                if time.time() - os.path.getmtime(this_file) > self._file_cadence_secs:
                    traceback.print_exc()
                    errstr = (
                        "WARNING: %s being deleted because it raised an error"
                        " and is not new"
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
            # [()] casts a Dataset as a numpy array (or python object if the
            # Dataset is of object type, e.g. a variable length string)
            val = obj[()]
            if isinstance(val, np.generic):
                # if value is numpy scalar, get as python type
                # scalars without a corresponding type stay as numpy scalars
                # (numpy scalars often act like the corresponding python types,
                #  but not always, and this prevents surprises such as np.int_
                #  not subclassing from int in Python 3)
                val = val.item()
            elif isinstance(val, bytes):
                # h5py, as of version 2.9, returns ascii text as bytes
                # but since we never write arbitrary bytes, it's much more
                # convenient to convert this into a proper Python string
                try:
                    val = val.decode()
                except UnicodeDecodeError:
                    pass
            ret_dict[name] = val
        else:
            # create a dictionary for this group
            ret_dict[name] = {}
            for key, value in obj.items():
                self._populate_data(ret_dict[name], value, key)

    def _check_compatible_version(self):
        version = packaging.version.parse(self._digital_metadata_version)

        if version < self._min_version:
            errstr = (
                "The Digital Metadata files being read are version {0}, which"
                " is less than the required version ({1})."
            ).format(version.base_version, self._min_version.base_version)
            raise IOError(errstr)
        elif version > self._max_version:
            warnstr = (
                "The Digital Metadata files being read are version {0}, which"
                " is higher than the maximum supported version ({1}) for this"
                " digital_rf package. If you encounter errors, you will have"
                " upgrade to at least version {0} of digital_rf."
            ).format(version.base_version, self._max_version.base_version)
            warnings.warn(warnstr, RuntimeWarning)

    def __str__(self):
        """String summary of the DigitalMetadataReader's parameters."""
        ret_str = ""
        attr_list = (
            "_subdir_cadence_secs",
            "_file_cadence_secs",
            "_samples_per_second",
            "_file_name",
        )
        for attr in attr_list:
            ret_str += "%s: %s\n" % (attr, str(getattr(self, attr)))
        if self._fields is None:
            ret_str += "_fields: None\n"
        else:
            ret_str += "_fields:\n"
            for key in self._fields:
                ret_str += "\t%s\n" % (key)
        return ret_str
