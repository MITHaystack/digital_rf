"""example of digital_metadata.read_metadata

$Id$
"""

# Millstone imports
import digital_rf

metadata_dir = '/tmp/test_metadata'
stime = 1447082580

try:
    obj = digital_rf.DigitalMetadataReader(metadata_dir)
except:
    print('Be sure you run test_write_digital_metadata.py before running this test code.')
    raise
print('init okay')

first_sample, last_sample = obj.get_bounds()
print('bounds are %i to %i' % (first_sample, last_sample))

fields = obj.get_fields()
print('Available fields are <%s>' % (str(fields)))

print('first read - just get one column simple_complex')
data_dict = obj.read(stime, stime + 2, 'single_complex')
for key in data_dict.keys():
    print((key, data_dict[key]))

print('second read - just 2 columns: simple_complex and numpy_obj')
data_dict = obj.read(stime, stime + 2, ('single_complex', 'numpy_obj'))
for key in data_dict.keys():
    print((key, data_dict[key]))

print('third read - get all columns')
data_dict = obj.read(stime, stime + 2)
for key in data_dict.keys():
    print((key, data_dict[key]))

print('just get latest metadata')
latest_meta = obj.read_latest()
print(latest_meta)

print('test of get_samples_per_second')
sps = obj.get_samples_per_second()
print(sps)
