"""example of digital_rf_metadata.write_digital_metadata

Now writes data into two levels of dictionaries/groups.  API allow any finite number
of levels.

$Id$
"""
# standard python imports
import os

# third party imports
import numpy

# Millstone imports
import digital_rf

metadata_dir = '/tmp/test_metadata'
subdirectory_cadence_seconds = 3600
file_cadence_seconds = 60
samples_per_second_numerator = 10
samples_per_second_denominator = 9
file_name = 'rideout'
stime = 1447082580

os.system('mkdir %s' % (metadata_dir))
os.system('rm -r %s/*' % (metadata_dir))

obj = digital_rf.DigitalMetadataWriter(metadata_dir, subdirectory_cadence_seconds, file_cadence_seconds,
                                       samples_per_second_numerator, samples_per_second_denominator,
                                       file_name)
print('first create okay')

data_dict = {}
start_idx = long(numpy.uint64(stime * obj.get_samples_per_second()))
idx_arr = numpy.arange(70, dtype=numpy.int64) + start_idx

int_data = numpy.arange(70, dtype=numpy.int32)
data_dict['int_data'] = int_data
float_data = numpy.arange(70, dtype=numpy.float32)
data_dict['float_data'] = float_data
complex_data = numpy.arange(70, dtype=numpy.complex64)
data_dict['complex_data'] = complex_data
single_int = 5
data_dict['single_int'] = numpy.int32(single_int)
single_float = 6.0
data_dict['single_float'] = numpy.float64(single_float)
single_complex = 7.0 + 8.0j
data_dict['single_complex'] = numpy.complex(single_complex)

# now create subdirectories
sub_dict = {}
sub_dict['single_int'] = numpy.int32(single_int)
sub_dict['single_float'] = numpy.float64(single_float)
sub_dict['single_complex'] = numpy.complex(single_complex)
level2_dict = {}  # embed yey another level
level2_dict['single_float'] = numpy.float64(single_float)
sub_dict['level2'] = level2_dict


data_dict['sub_system'] = sub_dict

# complex python object
n = numpy.ones((10, 4), dtype=numpy.float64)
n[5, :] = 17.0
data_dict['numpy_obj'] = [n for i in range(70)]


obj.write(idx_arr, data_dict)
print('first write_metadata okay')

# write same data again after incrementating inx
idx_arr += 70

del(obj)
obj = digital_rf.DigitalMetadataWriter(metadata_dir, subdirectory_cadence_seconds, file_cadence_seconds,
                                       samples_per_second_numerator, samples_per_second_denominator,
                                       file_name)
print('second create okay')


obj.write(idx_arr, data_dict)
print('second write_metadata okay')
