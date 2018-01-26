# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""regression_test_digital_rf_hdf5.py is a script to regression test the digital_rf_hdf5 module

Exits with 0 if successful test, non-zero if failure

$Id$
"""
# standard python imports
import os
import sys
import time
import collections

# third party imports
import numpy

# Millstone imports
import digital_rf

# base data
base_data = []
for i in range(100):
    if i % 7 == 0:
        base_data.append([2, 3])
    else:
        base_data.append([4, 6])

# simulate 10 samples of data followed by 10 sample gap
global_sample_arr = numpy.array(range(10), dtype=numpy.uint64) * 20
block_sample_arr = numpy.array(range(10), dtype=numpy.uint64) * 10

# constants
sample_rate_numerator = int(100)
sample_rate_denominator = 1
sample_rate = numpy.longdouble(sample_rate_numerator) / sample_rate_denominator
subdir_cadence_secs = 10
file_cadence_millisecs = 400
# start 2014-03-09 12:30:30 plus one sample
start_global_index = int(numpy.uint64(1394368230 * sample_rate)) + 1

# test get unix time
dt, picoseconds = digital_rf.get_unix_time(
    int(start_global_index), sample_rate_numerator, sample_rate_denominator)
print('For start_global_index=%i and sample_rate=%f, dt is %s and picoseconds is %i' % (start_global_index, sample_rate,
                                                                                        dt, picoseconds))

# set up top level directory
os.system("rm -rf /tmp/hdf5 ; mkdir /tmp/hdf5")
os.system("rm -rf /tmp/hdf52 ; mkdir /tmp/hdf52")

print("Test 0 - simple single write to multiple files, no compress, no checksum - channel 0")
os.system("rm -rf /tmp/hdf5/junk0 ; mkdir /tmp/hdf5/junk0")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk0", 'i4', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_0", 0, False, True)
data = numpy.array(base_data, numpy.int32)
result = data_object.rf_write(data)
data_object.close()
print("done test 0")

print("Test 1 - use complex 1 byte ints, no compress, no checksum - channel 1")
os.system("rm -rf /tmp/hdf5/junk1 ; mkdir /tmp/hdf5/junk1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1", 0, False, True)
data = numpy.array(base_data, numpy.int8)
data_object.rf_write(data)
data_object.close()
print(result)
print("done test 1")

print("Test 1.1 - use single 1 byte ints with data gap, no compress, no checksum - channel 1.1")
os.system("rm -rf /tmp/hdf5/junk1.1 ; mkdir /tmp/hdf5/junk1.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1.1", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.1", 0, False, False)
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
data_object.rf_write(data)
data_object.rf_write(data, 120)
data_object.close()
print("done test 1.1")

print("Test 1.2 - use single 1 byte ints with no gaps across multiple subdirectories and top_level dirs, no compress, no checksum - channel 1.1")
os.system("rm -rf /tmp/hdf5/junk1.2 ; mkdir /tmp/hdf5/junk1.2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk1.2", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.2", 0, False, False)
data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
for i in range(100):
    data_object.rf_write(data)
data_object.close()
print('now use top level dir /tmp/hdf52')
os.system("rm -rf /tmp/hdf52/junk1.2 ; mkdir /tmp/hdf52/junk1.2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf52/junk1.2", 'i1', subdir_cadence_secs, file_cadence_millisecs, start_global_index + 100 * 100,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_1.2", 0, False, False)

data = numpy.array(base_data[:], numpy.int8)
data = data[:, 0]
for i in range(100):
    data_object.rf_write(data)
data_object.close()
print("done test 1.2")

print("Test 2 - use 2 byte ints with data gap, level 1 compress, but no checksum - channel 2")
os.system("rm -rf /tmp/hdf5/junk2 ; mkdir /tmp/hdf5/junk2")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk2", 'i2', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, is_continuous=False)
data = numpy.array(base_data, numpy.int16)
data_object.rf_write(data)
data_object.rf_write(data, 120)
data_object.close()
print("done test 2")

print("Test 4.1 - use single 8 byte ints with 10 on/10 missing blocks, both compress (level 6) and checksum - channel 4.1")
os.system("rm -rf /tmp/hdf5/junk4.1 ; mkdir /tmp/hdf5/junk4.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf5/junk4.1", 'i8', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, is_continuous=False)
data = numpy.array(base_data, numpy.int64)
for i in range(100):
    data_object.rf_write_blocks(data, global_sample_arr, block_sample_arr)
    global_sample_arr += 205
data_object.close()

print('Now write more of this same test 4.1 to top level directory /tmp/hdf52')
os.system("rm -rf /tmp/hdf52/junk4.1 ; mkdir /tmp/hdf52/junk4.1")
data_object = digital_rf.DigitalRFWriter("/tmp/hdf52/junk4.1", 'i8', subdir_cadence_secs, file_cadence_millisecs, start_global_index,
                                         sample_rate_numerator, sample_rate_denominator, "FAKE_UUID_2", 1, False, True, is_continuous=False)
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
testReadObj = digital_rf.DigitalRFReader(['/tmp/hdf5', '/tmp/hdf52'])
print('init took %f' % (time.time() - t))
channels = testReadObj.get_channels()
print('existing channels are <%s>' % (str(channels)))

chan_name = 'junk4.1'
print('\nworking on channel %s' % (chan_name))
start_index, end_index = testReadObj.get_bounds(chan_name)
print('Bounds are %i to %i' % (start_index, end_index))

# test one of get_continuous_blocks - end points in gaps
start_sample = 139436843434L
end_sample = 139436843538L
print('\ncalling get_continuous_blocks between %i and %i (edges in data gaps)' %
      (start_sample, end_sample))
cont_data_arr = testReadObj.get_continuous_blocks(
    start_sample, end_sample, chan_name)
corr_result = collections.OrderedDict([(139436843436L, 10L),
                                       (139436843456L, 10L),
                                       (139436843476L, 10L),
                                       (139436843501L, 10L),
                                       (139436843521L, 10L)])
if not numpy.array_equal(cont_data_arr, corr_result):
    raise ValueError, 'cont_data_arr = <%s>, but corr_result = <%s>' % (
        str(cont_data_arr), str(corr_result))
print('Got correct result %s' % (str(cont_data_arr)))

# test two of get_continuous_blocks - end points in data
start_sample = 139436843438L
end_sample = 139436843528L
print('\ncalling get_continuous_blocks between %i and %i (edges in continuous data)' %
      (start_sample, end_sample))
cont_data_arr = testReadObj.get_continuous_blocks(
    start_sample, end_sample, chan_name)
corr_result = collections.OrderedDict([(139436843438, 8),
                                       (139436843456, 10),
                                       (139436843476, 10),
                                       (139436843501, 10),
                                       (139436843521, 8)])
if not numpy.array_equal(cont_data_arr, corr_result):
    raise ValueError, 'cont_data_arr = <%s>, but corr_result = <%s>' % (
        str(cont_data_arr), str(corr_result))
print('Got correct result %s' % (str(cont_data_arr)))

# normal read
print('\ntest of read of continuous data')
first_key = cont_data_arr.keys()[0]
print((first_key, cont_data_arr[first_key]))
read_result = testReadObj.read_vector(
    first_key, cont_data_arr[first_key], chan_name)
corr_result = numpy.array([[4.0 + 6.0j],
                           [4.0 + 6.0j],
                           [4.0 + 6.0j],
                           [4.0 + 6.0j],
                           [4.0 + 6.0j],
                           [2.0 + 3.0j],
                           [4.0 + 6.0j],
                           [4.0 + 6.0j]])
if not numpy.array_equal(read_result, corr_result):
    raise ValueError, 'read_result = <%s>, but corr_result = <%s>' % (
        str(read_result), str(corr_result))
print('Got correct result %s' % (str(read_result)))

print('\nnow try reading across a data gap, which should raise an exception')
try:
    # read too far
    result = testReadObj.read_vector(first_key, cont_data_arr[
                                     first_key] + 1, chan_name)
    raise ValueError, 'whoops - no error when one expected!!!!!'
except IOError:
    print(sys.exc_info()[1])
    print('got expected error')


# read one less than block - should be okay
print('\ntest of reading subset of a continuous block')
read_result = testReadObj.read_vector(
    first_key, cont_data_arr[first_key] - 1, chan_name)
if not numpy.array_equal(read_result, corr_result[:-1]):
    raise ValueError, 'read_result = <%s>, but corr_result = <%s>' % (
        str(read_result), str(corr_result[:-1]))
print('Got correct result %s' % (str(read_result)))

chan_name = 'junk1.2'
print('\n now working on channel %s - large continuous data, with padding because continuous but not at file boundaries' % (chan_name))
start_index, end_index = testReadObj.get_bounds(chan_name)
corr_result = (139436823000L, 139436843039L)
if start_index != corr_result[0] or end_index != corr_result[1]:
    raise ValueError, 'got wrong bounds: %s versus %s' % (
        str((start_index, end_index)), str(corr_result))
print('got correct bounds %s' % (str((start_index, end_index))))

print('\ntest of get_continuous_blocks for continuous data')
result = testReadObj.get_continuous_blocks(start_index, end_index, chan_name)
key = result.keys()[0]
if not len(result.keys()) == 1 or key != corr_result[0] or result[key] != (corr_result[1] - corr_result[0]) + 1:
    raise ValueError, 'got wrong continuous blocks: %s versus %s' % (
        str(result), str(corr_result))
print('get correct continuous blocks %s' % (str(result)))

print('\ntest of read_vector for all continuous data')
t = time.time()
result = testReadObj.read_vector(
    start_index, end_index - start_index, chan_name)
if len(result) != corr_result[1] - corr_result[0]:
    raise ValueError, 'got length %i, expected %i' % (
        len(result), corr_result[1] - corr_result[0])
print('got expected len %i' % (len(result)))
print('read took %f' % (time.time() - t))


print('regression test of digital_rf_hdf5 SUCCESS')
