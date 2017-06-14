"""example_digital_rf_hdf5.py is an example script using the digital_rf_hdf5 module

Assumes one of the example Digital RF scripts has already been run (C: example_rf_write_hdf5, or
Python: example_digital_rf.py)

$Id$
"""
# Millstone imports
import digital_rf


testReadObj = digital_rf.DigitalRFReader(['/tmp/hdf5'])
channels = testReadObj.get_channels()
if len(channels) == 0:
    raise IOError, """Please run one of the example write scripts
        C: example_rf_write_hdf5, or Python: example_digital_rf_hdf5.py
        before running this example"""
print('found channels: %s' % (str(channels)))

print('working on channel junk0')
start_index, end_index = testReadObj.get_bounds('junk0')
print('get_bounds returned %i - %i' % (start_index, end_index))
cont_data_arr = testReadObj.get_continuous_blocks(
    start_index, end_index, 'junk0')
print('The following is a OrderedDict of all continuous block of data in (start_sample, length) format: %s' %
      (str(cont_data_arr)))

# read data - the first 3 reads of four should succeed, the fourth read
# will be beyond the available data
start_sample = cont_data_arr.keys()[0]
for i in range(4):
    try:
        result = testReadObj.read_vector(start_sample, 200, 'junk0')
        print('read number %i got %i samples starting at sample %i' %
              (i, len(result), start_sample))
        start_sample += 200
    except IOError:
        print('Read number %i went beyond existing data and raised an IOError' % (i))

# finally, get all the built in rf properties
rf_dict = testReadObj.get_properties('junk0')
print('Here is the metadata built into drf_properties.h5 (valid for all data): %s' %
      (str(rf_dict)))
