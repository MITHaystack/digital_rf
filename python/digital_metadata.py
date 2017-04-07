"""digital_metadata.py is a module to read and write metadata in the standard digital_metadata format.

See XXXX for a description of digital_metadata

$Id$
"""

# standard python imports
import os
import collections
import datetime
import time
import types
import glob
import copy
import traceback
import urllib2
import distutils.version
import fractions

# third party imports
import h5py
import numpy

# local imports
from ._version import __version__

__all__ = (
    'DigitalMetadataReader', 'DigitalMetadataWriter',
)

# constants
_min_version = '2.0' # the version digital metadata must be to be successfully read

class DigitalMetadataWriter:
    """DigitalMetadataWriter is the class used to write digital_metadata
    """
    def __init__(self, metadata_dir, subdirectory_cadence_seconds, file_cadence_seconds,
                 samples_per_second_numerator, samples_per_second_denominator, file_name):
        """__init__ creates an object to write a single channel of digital_metadata

        Inputs:
            metadata_dir - the top level directory where the metadata will be written. Must already exist.
            subdirectory_cadence_seconds - the integer number of seconds of metadata to store in each subdirectory
                This API will enforce the rule that the timestamp of any subdirectory is n*subdirectory_cadence_seconds
            file_cadence_seconds - the integer number of seconds to store in each file
                Note that N files must span exactly subdirectory_cadence_seconds, which implies
                subdirectory_cadence_seconds % file_cadence_seconds == 0
                This API will enforce the rule that file name timestamps are in the list:
                    range(subdirectory_timestamp, subdirectory_timestamp+subdirectory_cadence_seconds, file_cadence_seconds)
            samples_per_second_numerator - samples per second numerator (long). Used since digital_metadata uses samples since 1970 in all indexing.
            samples_per_second_denominator - samples per second denominator (long). Used since digital_metadata uses samples since 1970 in all indexing.
            file_name - prefix for metadata file names.

        All inputs are saved as class attributes.  Also self._fields to None if no data yet, or data does exist, then reads
        "fields" dataset from at the top level of metadata.h5.  Then self._fields is set to a list of keys (dataset names)
        """
        # verify all input arguments
        self._metadata_dir = metadata_dir
        if not os.access(metadata_dir, os.W_OK):
            raise IOError, 'metadata_dir %s does not exist or is not writable' % (metadata_dir)

        intTypes = (types.IntType, types.LongType, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64)

        if (type(subdirectory_cadence_seconds)) not in intTypes:
            raise ValueError, 'subdirectory_cadence_seconds must be int type, not %s' % (str(type(subdirectory_cadence_seconds)))

        self._subdirectory_cadence_seconds = int(subdirectory_cadence_seconds)
        if self._subdirectory_cadence_seconds < 1:
            raise ValueError, 'subdirectory_cadence_seconds must be positive integer, not %s' % (str(subdirectory_cadence_seconds))

        if (type(file_cadence_seconds)) not in intTypes:
            raise ValueError, 'file_cadence_seconds must be int type, not %s' % (str(type(file_cadence_seconds)))

        self._file_cadence_seconds = int(file_cadence_seconds)
        if self._file_cadence_seconds < 1:
            raise ValueError, 'file_cadence_seconds must be positive integer, not %s' % (str(file_cadence_seconds))
        if self._subdirectory_cadence_seconds % self._file_cadence_seconds != 0:
            raise ValueError, 'subdirectory_cadence_seconds % file_cadence_seconds must be zero'

        self._file_name = file_name
        if not type(file_name) in (types.StringType, types.StringTypes):
            raise ValueError, 'file_name must be a string, not type %s' % (str(type(file_name)))

        if (type(samples_per_second_numerator)) not in intTypes:
            raise ValueError, 'samples_per_second_numerator must be int type, not %s' % (str(type(samples_per_second_numerator)))

        self._samples_per_second_numerator = long(samples_per_second_numerator)
        if self._samples_per_second_numerator < 1:
            raise ValueError, 'samples per second numerator must be positive, not %s' % (str(samples_per_second_numerator))

        if (type(samples_per_second_denominator)) not in intTypes:
            raise ValueError, 'samples_per_second_denominator must be int type, not %s' % (str(type(samples_per_second_denominator)))

        self._samples_per_second_denominator = long(samples_per_second_denominator)
        if self._samples_per_second_denominator < 1:
            raise ValueError, 'samples per second denominator must be positive, not %s' % (str(samples_per_second_denominator))

        # have to go to uint64 before longdouble to ensure correct conversion from long
        self._samples_per_second = (numpy.longdouble(numpy.uint64(self._samples_per_second_numerator)) /
                                    numpy.longdouble(numpy.uint64(self._samples_per_second_denominator)))

        if os.access(os.path.join(self._metadata_dir, 'metadata.h5'), os.R_OK):
            self._parse_metadata()
        else:
            self._write_metadata_summary()
            self._fields = None # No data written yet

    def get_samples_per_second(self):
        """returns the calculated samples per second as a numpy.longdouble for this metadata
        """
        return(self._samples_per_second)


    def write(self, samples, data_dict):
        """write is the main method used to write new metadata to a metadata channel.

        Inputs:
            samples - A single sample (long) or a list or numpy vector of samples, length = length data value lists. A sample
                is the unix time times the sample rate as a long.
            data_dict - a dictionary representing the metadata to write.  keys are the field names, values can be 1) a list
                of numpy objects or 2) a vector numpy array of length samples, or 3) a single numpy object if samples is length 1,
                or 4) another dictionary whose keys are string that are valid Group names and leaf values are
                one of the three types above.
                Length of list or vector must equal length of samples, and if a single numpy object,
                then length of samples must be one. Data must always have the same names for each call to write.
        """
        if self._fields is None:
            self._set_fields(data_dict)

        if type(samples) == types.FloatType:
            samples = long(samples)

        if type(samples)  in (types.LongType, types.IntType):
            samples = [samples]

        # get mapping of sample ranges to subdir/filenames
        file_info_list = self._get_subdir_filename_info(samples)

        for sample_list, index_list, subdir, file_basename in file_info_list:
            self._write_metadata_range(data_dict, sample_list, index_list, len(samples), subdir, file_basename)



    def _write_metadata_range(self, data_dict, sample_list, index_list, sample_len, subdir, filename):
        """_write_metadata_range is the private method that actually writes to an hdf5 file

        Inputs:
            data_dict - input data as described in write method
            sample_list - list of all samples to write to this particular file
            index_list - list of indexes into data_dict values for each sample in sample_list
            sample_len - total number of samples written. Used to determine if writing numpy arrays item by item or
                as a whole
            subdir - full path to subdirectory to write to.  May or may not exist.
            filename - basename of data file to write.  May or may not exist.
        """
        if not os.access(subdir, os.W_OK):
            os.mkdir(subdir)
        this_file = os.path.join(subdir, filename)

        with h5py.File(this_file, 'a') as f:
            existing_samples = f.keys()
            for index, sample in enumerate(sample_list):
                if str(sample) in existing_samples:
                    raise IOError, 'sample %i already in data - no overwriting allowed' % (sample)
                grp = f.create_group(str(sample))
                self._dict_list = [] # a list of data_dict called already - raises exception to prevent loops
                self._write_dict(grp, data_dict, index_list[index], sample_len)


    def _write_dict(self, grp, this_dict, sample_index, sample_len):
        """_write_dict is a recursive function that writes all non-dict values in this_dict,
        and calls itself for all dictionary values in this_dict.

        Inputs:
            grp - h5py.Group object in which to write all non-dict values in this_dict.
            this_dict - dictionary as described in write method
            sample_index - index of sample presently being written
            sample_len - total number of samples to write into this particular file
        """
        # prevent loops by checking that this is a unique dict
        if this_dict in self._dict_list:
            raise ValueError, 'Infinite loop in data - dict <%s> passed in twice' % (str(this_dict)[0:500])
        self._dict_list.append(this_dict)

        for key in this_dict.keys():
            if type(this_dict[key]) == types.DictType:
                new_grp = grp.create_group(str(key))
                self._write_dict(new_grp, this_dict[key], sample_index, sample_len)
                data = None
            elif type(this_dict[key]) in (types.ListType, types.TupleType):
                data = this_dict[key][sample_index]
            elif hasattr(this_dict[key], 'shape'):
                if this_dict[key].shape == (sample_len,):
                    data = this_dict[key][sample_index]
                else:
                    data = this_dict[key] # single numpy value
            else:
                data = this_dict[key] # single numpy value
            if not data is None:
                grp.create_dataset(key, data=data)



    def _get_subdir_filename_info(self, samples):
        """_get_subdir_filename_info returns a list of tuples, where each tuple has four items:
        1. list of samples to write to one file, 2. List of indexes of samples in 1, 3. the full subdirectory name,
        and 4. the base filename

        Inputs:
            samples - a list or numpy array of samples at which metadata is to be written
        """
        ret_list = []
        index = 0
        while True:
            # loop until index beyond length of samples or no more samples
            # start_ts: need to go through numpy uint64 to prevent conversion to float
            start_ts = long(numpy.uint64(samples[index]/self._samples_per_second))
            start_sub_ts = (start_ts//self._subdirectory_cadence_seconds)*self._subdirectory_cadence_seconds
            sub_dt = datetime.datetime.utcfromtimestamp(start_sub_ts)
            subdir = os.path.join(self._metadata_dir, sub_dt.strftime('%Y-%m-%dT%H-%M-%S'))
            file_num = (start_ts - start_sub_ts) // self._file_cadence_seconds
            file_ts = start_sub_ts + file_num * self._file_cadence_seconds
            file_basename = '%s@%i.h5' % (self._file_name, file_ts)
            # start sample list
            sample_list = [samples[index]]
            index_list = [index]
            # find sample of next file
            next_file_start_sample = long(numpy.uint64(numpy.ceil((file_ts+self._file_cadence_seconds)*self._samples_per_second)))
            sample_found = False
            while True:
                index += 1
                if index >= len(samples):
                    break
                if samples[index] >= next_file_start_sample:
                    sample_found = True
                    ret_list.append((sample_list, index_list, subdir, file_basename))
                    break
                else:
                    sample_list.append(samples[index])
                    index_list.append(index)

            if not sample_found:
                ret_list.append((sample_list, index_list, subdir, file_basename))
                break

        return(ret_list)




    def _set_fields(self, data_dict):
        """_set_fields sets self._fields based on the first set of input data data_dict

        Inputs:
            data_dict - a dictionary representing the metadata to write.  keys are the field names, values are either a list
                of numpy objects or a single numpy object.

        Affects - updates metadata.h5 and self._fields
        """
        # build recarray and self._fields
        recarray = numpy.recarray((len(data_dict.keys()),),
                                  dtype=[('column', '|S128'),])
        self._fields = data_dict.keys()
        self._fields.sort() # for reproducability, use alphabetic order
        for i, key in enumerate(self._fields):
            recarray[i] = (key,)

        # write recarray to metadata
        with h5py.File(os.path.join(self._metadata_dir, 'metadata.h5'), 'a') as f:
            f.create_dataset("fields", data=recarray)



    def _parse_metadata(self):
        """_parse_metadata compares all the values set in the writer init to the already existing ones in the channel, if the
        channel already exists.  Raises error if any mismatch.  Also sets self._fields. Also verifies version acceptable
        """
        org_obj = DigitalMetadataReader(self._metadata_dir, accept_empty=True)
        if not hasattr(org_obj, '_digital_metadata_version'):
            raise IOError, 'These digital metadata files too old to be read with this module'
        version = getattr(org_obj, '_digital_metadata_version')
        if distutils.version.StrictVersion(version) < distutils.version.StrictVersion(_min_version):
            raise IOError, 'Existing Digital Metadata files being read version %s, less than required version %s' % (version, _min_version)
        attr_list = ('_subdirectory_cadence_seconds', '_file_cadence_seconds', '_samples_per_second_numerator',
                     '_samples_per_second_denominator', '_file_name')
        for attr in attr_list:
            if getattr(self, attr) != getattr(org_obj, attr):
                raise ValueError, 'Mismatched %s: %s versus %s' % (attr, getattr(self, attr), getattr(org_obj, attr))
        self._fields = getattr(org_obj, '_fields') # may be None or list described in init


    def _write_metadata_summary(self):
        """
        """
        with h5py.File(os.path.join(self._metadata_dir, 'metadata.h5'), 'w') as f:
            f.attrs['subdirectory_cadence_seconds'] = self._subdirectory_cadence_seconds
            f.attrs['file_cadence_seconds'] = self._file_cadence_seconds
            f.attrs['samples_per_second_numerator'] = self._samples_per_second_numerator
            f.attrs['samples_per_second_denominator'] = self._samples_per_second_denominator
            f.attrs['file_name'] = self._file_name
            f.attrs['digital_metadata_version'] = __version__



    def __str__(self):
        ret_str = ''
        attr_list = ('_subdirectory_cadence_seconds', '_file_cadence_seconds', '_samples_per_second_numerator',
                     '_samples_per_second_denominator', '_file_name')
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
    """DigitalMetadataReader is the class used to access digital_metadata
    """

    def __init__(self, metadata_dir, accept_empty=True):
        """__init__ creates needed class attributes by reading <metadata_dir>/metadata.h5

        If accept_empty is False, raises IOError if metadata.h5 not found or cannot be parsed, and deletes data
        """
        self._metadata_dir = metadata_dir
        if self._metadata_dir.find('http://') != -1:
            self._local = False
            # put metadata file in /tmp/metadata_%i.h5 % (pid)
            url = os.path.join(self._metadata_dir, 'metadata.h5')
            f = urllib2.urlopen(url)
            tmp_file = os.path.join('/tmp', 'metadata_%i.h5' % (os.getpid()))
            fo = open(tmp_file, 'w')
            fo.write(f.read())
            f.close()
            fo.close()

        else:
            self._local = True
            tmp_file = os.path.join(metadata_dir, 'metadata.h5')

        with h5py.File(tmp_file, 'r') as f:
            self._subdirectory_cadence_seconds = f.attrs['subdirectory_cadence_seconds'].item()
            self._file_cadence_seconds = f.attrs['file_cadence_seconds'].item()
            try:
                spsn = f.attrs['samples_per_second_numerator'].item()
                spsd = f.attrs['samples_per_second_denominator'].item()
            except KeyError:
                # must have an older version with samples_per_second attribute
                sps = f.attrs['samples_per_second'].item()
                spsfrac = fractions.Fraction(sps).limit_denominator()
                self._samples_per_second = numpy.longdouble(sps)
                self._samples_per_second_numerator = long(spsfrac.numerator)
                self._samples_per_second_denominator = long(spsfrac.denominator)
            else:
                self._samples_per_second_numerator = spsn
                self._samples_per_second_denominator = spsd
                # have to go to uint64 before longdouble to ensure correct conversion from long
                self._samples_per_second = (
                    numpy.longdouble(numpy.uint64(
                        self._samples_per_second_numerator
                    )) /
                    numpy.longdouble(numpy.uint64(
                        self._samples_per_second_denominator
                    ))
                )
            self._file_name = f.attrs['file_name']
            try:
                version = f.attrs['digital_metadata_version']
            except:
                # version is before 2.3 when attribute was added
                version = '2.0'
            self._digital_metadata_version = version
            if distutils.version.StrictVersion(version) < distutils.version.StrictVersion(_min_version):
                raise IOError, 'Digital Metadata files being read version %s, less than required version %s' % (version, _min_version)
            try:
                fields_dataset = f['fields']
            except:
                f.close()
                if not accept_empty:
                    os.remove(os.path.join(metadata_dir, 'metadata.h5'))
                    raise IOError, 'No metadata yet written to %s' % (self._metadata_dir)
                else:
                    self._fields = None
                    return
            self._fields = []
            for i in range(len(fields_dataset)):
                self._fields.append(fields_dataset[i]['column'])

        if not self._local:
            os.remove(tmp_file)


    def get_bounds(self):
        """get_bounds returns a tuple of first sample, last sample for this metadata. A sample
                is the unix time times the sample rate as a long.

        Raises IOError if no data
        """
        # get subdirectory list
        subdir_list = glob.glob(os.path.join(self._metadata_dir, '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'))
        subdir_list.sort()
        if len(subdir_list) == 0:
            raise IOError, 'glob returned no directories for %s' % (os.path.join(self._metadata_dir, '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'))
        first_subdir = subdir_list[0]
        last_subdir = subdir_list[-1]
        first_sample = None
        # try first three subdirectories in case of subdirectory deletion occuring
        for i in range(min(3, len(subdir_list))):
            file_list = glob.glob(os.path.join(subdir_list[i], '%s@*.h5' % (self._file_name)))
            file_list.sort()
            # try first three files in case of file deletion occuring
            for j in range(min(3, len(file_list))):
                try:
                    with h5py.File(file_list[j], 'r') as f:
                        groups = f.keys()
                        groups.sort()
                        first_sample = long(groups[0])
                    break
                except:
                    time.sleep(1)
            if not first_sample is None:
                # No need to look at other subdirectories
                break
        if first_sample is None:
            raise IOError, 'All attempts to read first sample failed'

        # now try to get last_file
        last_file = None
        for i in range(min(20, len(subdir_list))):
            index = -1*i -1
            file_list = glob.glob(os.path.join(subdir_list[index], '%s@*.h5' % (self._file_name)))
            file_list.sort()
            # try last files
            for j in range(min(20, len(file_list))):
                index2 = -1*j -1
                if os.path.getsize(file_list[index2]) == 0:
                    continue
                last_file = file_list[index2]
                try:
                    with h5py.File(last_file, 'r') as f:
                        groups = f.keys()
                        groups.sort()
                        last_sample = long(groups[-1])
                    return((first_sample, last_sample))
                except KeyboardInterrupt:
                    raise
                except IOError:
                    # delete as corrupt if not too new
                    if not(time.time() - os.path.getmtime(last_file) < self._file_cadence_seconds):
                        # probable corrupt file
                        traceback.print_exc()
                        print('WARNING: removing %s since may be corrupt' % (last_file))
                        os.remove(last_file)
                    last_file = None

        if last_file is None:
            raise IOError, 'All attempts to read last file failed'



    def get_fields(self):
        """get_fields returns a list of all the field names available in this metadata
        """
        # self._fields is an internal data structure, so make a copy for the user
        return(copy.deepcopy(self._fields))


    def get_samples_per_second_numerator(self):
        """returns the samples per second numerator as a long for this metadata
        """
        return(self._samples_per_second_numerator)


    def get_samples_per_second_denominator(self):
        """returns the samples per second denominator as a long for this metadata
        """
        return(self._samples_per_second_denominator)


    def get_samples_per_second(self):
        """returns the samples per second as a numpy.longdouble for this metadata
        """
        return(self._samples_per_second)


    def get_subdirectory_cadence_seconds(self):
        """returns the subdirectory_cadence_seconds as a long for this metadata
        """
        return(self._subdirectory_cadence_seconds)


    def get_file_cadence_seconds(self):
        """returns the file_cadence_seconds as a long for this metadata
        """
        return(self._file_cadence_seconds)


    def get_file_name_prefix(self):
        """returns the file_name_prefix as a string for this metadata
        """
        return(self._file_name)




    def read(self, sample0, sample1, columns=None):
        """read returns a OrderedDict representing the requested metadata.

        Inputs:
            sample0 - first sample for which to return metadata
            sample1 - last sample for which to return metadata. A sample
                is the unix time times the sample rate as a long.
            columns - either a single string representing one column of metadata to return, or a
                list of column names to return.  If None (the default), return all columns available in files.

        Returns:
            a collections.OrderedDict with ordered keys = all samples found for which there is metadata.
                Value is a dictionary with key = column names, value = either a dictionary with leaf
                values numpy objects, or a numpy object.
        """
        if sample0 > sample1:
            raise ValueError, 'Start sample %i more than end sample %i' % (sample0, sample1)
        self._columns = columns # if None, will be filled in when first file opened
        if type(columns) in (types.StringType, types.StringTypes):
            self._columns = [columns]
        file_list = self._get_file_list(sample0, sample1)
        ret_dict = collections.OrderedDict()
        for this_file in file_list:
            if this_file in (file_list[0], file_list[-1]):
                is_edge = True
            else:
                is_edge = False
            self._add_metadata(ret_dict, this_file, columns, sample0, sample1, is_edge)

        return(ret_dict)


    def read_latest(self):
        """read_latest simply calls read for all columns with samples near the last sample time available
        as returned by get_bounds.  Returns dict with only the largest sample as key

        Returns: dict with key = last sample, value is a dict with keys=column names, values = numpy values
        """
        start_sample, last_sample = self.get_bounds()
        dict = self.read(long(last_sample - 2*self._samples_per_second), last_sample)
        keys = dict.keys()
        if len(keys) == 0:
            raise IOError, 'unable to find metadata near the last sample'
        keys.sort()
        ret_dict = {}
        ret_dict[keys[-1]] = dict[keys[-1]]
        return(ret_dict)


    # internal methods


    def _get_file_list(self, sample0, sample1):
        """_get_file_list returns an ordered list of full file names of metadata files that contain metadata.
        """
        # need to go through numpy uint64 to prevent conversion to float
        start_ts = long(numpy.uint64(sample0/self._samples_per_second))
        end_ts = long(numpy.uint64(sample1/self._samples_per_second))

        # convert ts to be divisible by self._file_cadence_seconds
        start_ts = (start_ts // self._file_cadence_seconds) * self._file_cadence_seconds
        end_ts = (end_ts // self._file_cadence_seconds) * self._file_cadence_seconds

        # get subdirectory start and end ts
        start_sub_ts = (start_ts // self._subdirectory_cadence_seconds) * self._subdirectory_cadence_seconds
        end_sub_ts = (end_ts // self._subdirectory_cadence_seconds) * self._subdirectory_cadence_seconds

        ret_list = [] # ordered list of full file paths to return

        for sub_ts in range(start_sub_ts, end_sub_ts + self._subdirectory_cadence_seconds, self._subdirectory_cadence_seconds):
            sub_datetime = datetime.datetime.utcfromtimestamp(sub_ts)
            subdir = os.path.join(self._metadata_dir, sub_datetime.strftime('%Y-%m-%dT%H-%M-%S'))
            # create numpy array of all file TS in subdir
            file_ts_in_subdir = numpy.arange(sub_ts, sub_ts + self._subdirectory_cadence_seconds, self._file_cadence_seconds)
            valid_file_ts_list = numpy.compress(numpy.logical_and(file_ts_in_subdir >= start_ts, file_ts_in_subdir <= end_ts),
                                                file_ts_in_subdir)
            for valid_file_ts in valid_file_ts_list:
                file_basename = '%s@%i.h5' % (self._file_name, valid_file_ts)
                full_file = os.path.join(subdir, file_basename)
                # verify exists
                if not os.access(full_file, os.R_OK):
                    continue
                ret_list.append(full_file)

        return(ret_list)


    def _add_metadata(self, ret_dict, this_file, columns, sample0, sample1, is_edge):
        """_add_metadata adds metadata from a single metadata file to ret_dict

        Inputs:
            ret_dict - the OrderedDictionary to add metadata to
            this_file - the full path to the file to get metadata from
            columns - either a single string representing one column of metadata to return, or a
                list of column names to return.  If None, return all columns available in this file.
            sample0 - first sample for which to return metadata
            sample1 - last sample for which to return metadata
            is_edge - True if this is first of last file; False otherwise
        """
        basename = os.path.basename(this_file)
        file_ts = long(basename[len(self._file_name)+1:-3])
        try:
            with h5py.File(this_file, 'r') as f:
                if self._columns is None:
                    self._columns = self._fields
                    self._columns.sort()
                idx = f.keys()
                idx = numpy.array([long(key) for key in idx], dtype=numpy.int64)
                idx.sort() # a list of all samples in file
                # file has idx column
                if is_edge:
                    # calculate indices
                    indices = numpy.where(numpy.logical_and(idx >= sample0, idx <= sample1))[0]
                else:
                    indices = range(len(idx))
                sample_len = len(indices)
                for i in indices:
                    key = idx[i]
                    f_key = str(key)
                    ret_dict[key] = {} # each sample is its own dictionary
                    for column in self._columns:
                        value = f[f_key][column]
                        if isinstance(value, h5py.Dataset):
                            # dataset found
                            if hasattr(value, 'shape'):
                                if value.shape == (sample_len,):
                                    ret_dict[key][column] = value[i].value
                                else:
                                    ret_dict[key][column] = value.value # single numpy value
                            else:
                                ret_dict[key][column] = value.value # single numpy value
                        else:
                            # subgroup found
                            self._populateData(ret_dict[key], value, column, i, sample_len)

        except KeyboardInterrupt:
            raise
        except IOError:
            # decide whether this file is corrupt, or too new, or just missing
            if os.access(this_file, os.R_OK) and os.access(this_file, os.W_OK):
                if time.time() - os.path.getmtime(this_file) > self._file_cadence_seconds:
                    traceback.print_exc()
                    print("WARNING: %s being deleted because it raised an error and is not new" % (this_file))
                    os.remove(this_file)



    def _populateData(self, ret_dict, grp, name, index, sample_len):
        """_populateData is a recursive function that will walk through a Digital Metadata Hdf5
        file, populating the dictionary ret_dict based on the information in the h5py.Group grp which
        has the group name given by name.

        Calls _populateData recursively for any group under grp.  Adds any dataset found directly under
        grp as a numpy object.  If the numpy object has length == sample_length, then only index <index>
        written.

        Inputs:
            ret_dict - the python dictionary being filled out
            grp - the h5py.Group being examined
            name - the name the h5py.Group
            index - the sample index being read
            sample_len - the total number of samples in this file
        """
        # create a dictionary for this group
        ret_dict[name] = {}

        for key, value in grp.items():
            if isinstance(value, h5py.Dataset):
                # dataset found
                if hasattr(value, 'shape'):
                    if value.shape == (sample_len,):
                        ret_dict[name][key] = value[i].value
                    else:
                        ret_dict[name][key] = value.value # single numpy value
                else:
                    ret_dict[name][key] = value.value # single numpy value
            else:
                # subgroup found
                self._populateData(ret_dict[name], value, key, index, sample_len)




    def __str__(self):
        ret_str = ''
        attr_list = ('_subdirectory_cadence_seconds', '_file_cadence_seconds', '_samples_per_second',
                     '_file_name')
        for attr in attr_list:
            ret_str += '%s: %s\n' % (attr, str(getattr(self, attr)))
        if self._fields is None:
            ret_str += '_fields: None\n'
        else:
            ret_str += '_fields:\n'
            for key in self._fields:
                ret_str += '\t%s\n' % (key)
        return(ret_str)
