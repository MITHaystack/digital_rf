"""test_digital_rf_hdf5.py is a script to test the digital_rf_hdf5 module

Duplicates a subset of the testing in test_rf_write_hdf5.c, then adds read testing

$Id$
"""
# standard python imports
import os
import time
import traceback
import glob
import shutil

# third party imports
import numpy
import h5py

# Millstone imports
import digital_rf

# base data
num_subchannels = 4
base_data = []
for i in range(100):
    if i < 50:
        base_data.append([2, 3] * num_subchannels)
    else:
        base_data.append([4, 6] * num_subchannels)

# create data in r/i to test using that to write
arr_data = numpy.ones((100, num_subchannels), dtype=[
                      ('r', numpy.int16), ('i', numpy.int16)])
arr_data[0:50, :]['r'] = 2
arr_data[50:, :]['r'] = 4
arr_data[0:50, :]['i'] = 3
arr_data[50:, :]['i'] = 6

# create single channel data in r/i to test using that to write
arr_data_single = numpy.ones(
    (100, 1), dtype=[('r', numpy.int16), ('i', numpy.int16)])
arr_data_single[0:50, :]['r'] = 2
arr_data_single[50:, :]['r'] = 4
arr_data_single[0:50, :]['i'] = 3
arr_data_single[50:, :]['i'] = 6


# simulate 10 samples of data followed by 10 sample gap
global_sample_arr = numpy.array(range(10), dtype=numpy.uint64) * 20
block_sample_arr = numpy.array(range(10), dtype=numpy.uint64) * 10

# constants
sample_rate_numerator = 200
sample_rate_denominator = 3
sample_rate = numpy.longdouble(sample_rate_numerator)/sample_rate_denominator
subdir_cadence_secs = 10
file_cadence_millisecs = 400
# start 2014-03-09 12:30:30 plus one sample
start_global_index = long(numpy.uint64(1394368230 * sample_rate)) + 1

# test get unix time
dt, picoseconds = digital_rf.get_unix_time(
    long(start_global_index), sample_rate_numerator, sample_rate_denominator)
print('For start_global_index=%i and sample_rate=%f, dt is %s and picoseconds is %i' % (start_global_index, sample_rate,
                                                                                        dt, picoseconds))

# set up top level directory
os.system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5")
os.system("rm -rf /tmp/hdf52 ; mkdir /tmp/hdf52")

print("Test 0 - simple single write to multiple files, no compress, no checksum - channel 0")
os.system("rm -rf /tmp/hdf5/junk0 ; mkdir /tmp/hdf5/junk0")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk0", 'i4', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_0", 0, False, True, num_subchannels=num_subchannels)
data = numpy.array(base_data, numpy.int32)
result = data_object.rf_write(data)
data_object.close()
print("done test 0.1")

print("Test 0.1 - simple single write to multiple files using r/i struct layout, no compress, no checksum - channel 0.1")
os.system("rm -rf /tmp/hdf5/junk0.1 ; mkdir /tmp/hdf5/junk0.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk0.1", 'i2', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_0.1", 0, False, True, num_subchannels=num_subchannels)
result = data_object.rf_write(arr_data)
data_object.close()
print("done test 0.1")

print("Test 0.11 - simple single write with one subchannel to multiple files using r/i struct layout, no compress, no checksum - channel 0.1")
os.system("rm -rf /tmp/hdf5/junk0.11 ; mkdir /tmp/hdf5/junk0.11")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk0.11", 'i2', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_0.11", 0, False, True, num_subchannels=1)
result = data_object.rf_write(arr_data_single)
data_object.close()
print("done test 0.11")

print("Test 0.2 - read data from test 0.1 and write as channel 0.2")
os.system("rm -rf /tmp/hdf5/junk0.2 ; mkdir /tmp/hdf5/junk0.2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk0.2", 'i2', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_0.2", 0, False, True, num_subchannels=num_subchannels)
files = glob.glob('/tmp/hdf5/junk0.1/*/*.h5')
files.sort()
f = h5py.File(files[0])
read_data = f['rf_data'].value
result = data_object.rf_write(read_data)
data_object.close()
print("done test 0.2")

print("Test 1 - use complex 1 byte ints, no compress, no checksum - channel 1")
os.system("rm -rf /tmp/hdf5/junk1 ; mkdir /tmp/hdf5/junk1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1", 0, False, True, num_subchannels=num_subchannels)
data = numpy.array(base_data, numpy.int8)
data_object.rf_write(data)
data_object.close()
print(result)
print("done test 1")

print("Test 1.01 - use single 1 byte ints with no data gap, no compress, no checksum - channel 1.01")
os.system("rm -rf /tmp/hdf5/junk1.01 ; mkdir /tmp/hdf5/junk1.01")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1.01", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.01", 0, False, False, num_subchannels=1)
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
data_object.rf_write(data)
data_object.close()
print("done test 1.01")

print("Test 1.1 - use single 1 byte ints with data gap, no compress, no checksum - channel 1.1")
os.system("rm -rf /tmp/hdf5/junk1.1 ; mkdir /tmp/hdf5/junk1.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1.1", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.1", 0, False, False, num_subchannels=1)
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
data_object.rf_write(data)
data_object.rf_write(data, 120)
data_object.close()
print("done test 1.1")

print("Test 1.2 - use single 1 byte ints with no gaps across multiple subdirectories and top_level dirs, no compress, no checksum - channel 1.1")
os.system("rm -rf /tmp/hdf5/junk1.2 ; mkdir /tmp/hdf5/junk1.2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1.2", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.2", 0, False, False, num_subchannels=1)
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
for i in range(100):
    data_object.rf_write(data)
data_object.close()
print('now use top level dir /tmp/hdf52')
os.system("rm -rf /tmp/hdf52/junk1.2 ; mkdir /tmp/hdf52/junk1.2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf52/junk1.2", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index + 100 * 100,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.2", 0, False, False, num_subchannels=1)
"""# buggy version
data_object = digital_rf.DigitalRFWriter("/tmp/hdf52/junk1.2", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index + 50*100,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.2", 0, False, False);   """
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
for i in range(100):
    data_object.rf_write(data)
data_object.close()
print("done test 1.2")

print("Test 2 - use 2 byte ints with data gap, level 1 compress, but no checksum - channel 2")
os.system("rm -rf /tmp/hdf5/junk2 ; mkdir /tmp/hdf5/junk2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk2", 'i2', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, num_subchannels=num_subchannels)
data = numpy.array(base_data, numpy.int16)
data_object.rf_write(data)
data_object.rf_write(data, 120)
data_object.close()
print("done test 2")

print("Test 3 - use complex float, no compress, no checksum - channel 3")
os.system("rm -rf /tmp/hdf5/junk3 ; mkdir /tmp/hdf5/junk3")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk3", 'f', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_3", 0, False, True, num_subchannels=num_subchannels)
data = numpy.array(base_data, numpy.float32)
data_object.rf_write(data)
data_object.close()
print(result)
print("done test 3")

print("Test 3.1 - use complex double, no compress, no checksum - channel 3")
os.system("rm -rf /tmp/hdf5/junk3.1 ; mkdir /tmp/hdf5/junk3.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk3.1", 'd', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_3.1", 0, False, True, num_subchannels=num_subchannels)
data = numpy.array(base_data, numpy.double)
data_object.rf_write(data)
data_object.close()
print(result)
print("done test 3.1")

print("Test 3.2 - use complex double in complex form, no compress, no checksum - channel 3")
os.system("rm -rf /tmp/hdf5/junk3.2 ; mkdir /tmp/hdf5/junk3.2")
dbl_data = numpy.array(base_data, numpy.double)
data = numpy.zeros((dbl_data.shape[0], dbl_data.shape[1]/2), numpy.complex128)
data.real = dbl_data[:, ::2]
data.imag = dbl_data[:, 1::2]
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk3.2", data.dtype, subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_3.2", 0, False, True, num_subchannels=num_subchannels)
data_object.rf_write(data)
data_object.close()
print(result)
print("done test 3.2")

print("Test 4.1 - use single 8 byte ints with 10 on/10 missing blocks, both compress (level 6) and checksum - channel 4.1")
os.system("rm -rf /tmp/hdf5/junk4.1 ; mkdir /tmp/hdf5/junk4.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk4.1", 'i8', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, num_subchannels=num_subchannels,
                                                 is_continuous=False)
data = numpy.array(base_data, numpy.int64)
for i in range(100):
    data_object.rf_write_blocks(data, global_sample_arr, block_sample_arr)
    global_sample_arr += 205
data_object.close()

print('Now write more of this same test 4.1 to top level directory /tmp/hdf52')
os.system("rm -rf /tmp/hdf52/junk4.1 ; mkdir /tmp/hdf52/junk4.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf52/junk4.1", 'i8', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                                 sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, num_subchannels=num_subchannels,
                                                 is_continuous=False)
data = numpy.array(base_data, numpy.int64)
for i in range(100):
    data_object.rf_write_blocks(data, global_sample_arr, block_sample_arr)
    global_sample_arr += 205
data_object.close()
print("done test 4.1")

# sleep for 4 seconds to make sure system knows all files closed
time.sleep(4)

# read
t = time.time()
# test relative paths
pwd = os.getcwd()
os.chdir('/tmp')
testReadObj = digital_rf.DigitalRFReader(['hdf5', 'hdf52'])
print('init took %f' % (time.time() - t))

channels = testReadObj.get_channels()
print(channels)

print('working on channel4.1')
start_index, end_index = testReadObj.get_bounds('junk4.1')
print((start_index, end_index))
print('calling get_continuous_blocks')
block_start = start_global_index + 205 * 100 - 67
block_end = block_start + 204
cont_data_dict = testReadObj.get_continuous_blocks(
    block_start, block_end, 'junk4.1')
print(cont_data_dict)
start_sample = cont_data_dict.keys()[0]
sample_len = cont_data_dict[start_sample]
print((start_sample, sample_len))

print('checking get_properties - global values')
d = testReadObj.get_properties('junk4.1')
keys = d.keys()
keys.sort()
for key in keys:
    print('%s: %s' % (str(key), str(d[key])))

print('checking get_properties at set sammple %i' % (start_sample))
d = testReadObj.get_properties('junk4.1', start_sample)
keys = d.keys()
keys.sort()
for key in keys:
    print('%s: %s' % (str(key), str(d[key])))

# normal read
result = testReadObj.read_vector_raw(start_sample, sample_len, 'junk4.1')
print(len(result))
print(result)
print(result.dtype)
try:
    # read too far
    result = testReadObj.read_vector_raw(
        start_sample, sample_len + 1, 'junk4.1')
    raise ValueError, 'whoops - no error when one expected!!!!!'
except IOError:
    traceback.print_exc()
    print('got expected error')
# read one less than block - should be okay
t = time.time()
result = testReadObj.read_vector_raw(start_sample, sample_len - 1, 'junk4.1')
print('read took %f' % (time.time() - t))
print(len(result))
print(result)

print('working on channel1.2 - large continuous data')
start_index, end_index = testReadObj.get_bounds('junk1.2')
print((start_index, end_index))
print(testReadObj.get_continuous_blocks(start_index, end_index, 'junk1.2'))
t = time.time()
result = testReadObj.read_vector_raw(
    start_index, end_index - start_index, 'junk1.2')
print('read took %f' % (time.time() - t))
print(len(result))

print('print all channel rf file metadata for junk1.2')
this_dict = testReadObj.get_properties('junk1.2')
keys = this_dict.keys()
keys.sort()
for key in keys:
    print((key, this_dict[key]))

metadata_files = glob.glob(os.path.join(pwd, 'metadata*.000.h5'))
for metadata_file in metadata_files:
    shutil.copy(metadata_file, '/tmp/hdf5/junk1.2')

print('Test of read_vector from single subchannel channel')
start_index, end_index = testReadObj.get_bounds('junk0.11')
result = testReadObj.read_vector(
    start_index, end_index - start_index, 'junk0.11')
print('result')
print(result)
print('result.shape is %s' % (str(result.shape)))
print(result.dtype)

print('Test of read_vector from 4 subchannel channel')
start_index, end_index = testReadObj.get_bounds('junk0')
result = testReadObj.read_vector(start_index, end_index - start_index, 'junk0')
print('result')
print(result)
print('result.shape is %s' % (str(result.shape)))
print(result.dtype)

print('Test of read_vector from 4 subchannel float channel')
start_index, end_index = testReadObj.get_bounds('junk3')
result = testReadObj.read_vector(start_index, end_index - start_index, 'junk3')
print('result')
print(result)
print('result.shape is %s' % (str(result.shape)))
print(result.dtype)

print('Test of read_vector from 4 subchannel double channel')
start_index, end_index = testReadObj.get_bounds('junk3.1')
result = testReadObj.read_vector(
    start_index, end_index - start_index, 'junk3.1')
print('result')
print(result)
print('result.shape is %s' % (str(result.shape)))
print(result.dtype)

print('Test of read_vector from 4 subchannel double channel v2')
start_index, end_index = testReadObj.get_bounds('junk3.2')
result = testReadObj.read_vector(
    start_index, end_index - start_index, 'junk3.2')
print('result')
print(result)
print('result.shape is %s' % (str(result.shape)))
print(result.dtype)

print('Verify read_vector sets all imaginary values to zero with single valued channel')
start_index, end_index = testReadObj.get_bounds('junk1.01')
result = testReadObj.read_vector(
    start_index, end_index - start_index, 'junk1.01')
if len(numpy.nonzero(result.imag.flatten())[0]) > 0:
    raise ValueError, 'Got imaginary part when not expected'

print('Overall test passed')
