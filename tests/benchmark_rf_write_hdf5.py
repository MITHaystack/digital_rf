"""benchmark_rf_write_hdf5.py is a script to to run the same test as in benchmark_rf_write_hdf5.c,
except in C.  All tests single subchannel.

$Id$
"""
# standard python imports
import os
import time

# third party imports
import numpy

# Millstone imports
import digital_rf

# constants
WRITE_BLOCK_SIZE = 1000
N_WRITES = int(1e9 / WRITE_BLOCK_SIZE)
SAMPLE_RATE_NUMERATOR = long(1E9)
SAMPLE_RATE_DENOMINATOR = 1
sample_rate = (numpy.longdouble(numpy.uint64(SAMPLE_RATE_NUMERATOR)) /
               numpy.longdouble(numpy.uint64(SAMPLE_RATE_DENOMINATOR)))
subdir_cadence_secs = 3600
file_cadence_millisecs = 10

# start 2014-03-09 12:30:30 plus one sample
start_global_index = long(numpy.uint64(1394368230 * sample_rate)) + 1


# data to write
data_int16 = numpy.zeros((WRITE_BLOCK_SIZE, 2), dtype='i2')
# make random
for i in range(WRITE_BLOCK_SIZE):
    j = i * 2
    k = i * 2 + 1
    data_int16[i][0] = (j % 32768) * (j + 8192) * (j % 13)
    data_int16[i][1] = (k % 32768) * (k + 8192) * (k % 13)

print('creating top level dir /tmp/benchmark')
os.system("rm -rf /tmp/benchmark ; mkdir /tmp/benchmark")

print("Test 0 - simple single write to multiple files, no compress, no checksum - channel 0")
os.system("rm -rf /tmp/benchmark/junk0 ; mkdir /tmp/benchmark/junk0")
print("Start writing")
channelObj = digital_rf.DigitalRFWriter('/tmp/benchmark/junk0', 'i2', subdir_cadence_secs, file_cadence_millisecs,
                                        start_global_index, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR, 'Fake_uuid', 0, False)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0E6 * seconds)
print('Total time %i seconds, speed %1.2f MB/s' % (int(seconds), speedMB))


print("Test 1 - simple single write to multiple files, no compress, with checksum - channel 1")
os.system("rm -rf /tmp/benchmark/junk1 ; mkdir /tmp/benchmark/junk1")
print("Start writing")
channelObj = digital_rf.DigitalRFWriter('/tmp/benchmark/junk1', 'i2', subdir_cadence_secs, file_cadence_millisecs,
                                        start_global_index, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR, 'Fake_uuid', 0, True)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0E6 * seconds)
print('Total time %i seconds, speed %1.2f MB/s' % (int(seconds), speedMB))


print("Test 2 - simple single write to multiple files, compress to level 9, with checksum - channel 2")
os.system("rm -rf /tmp/benchmark/junk2 ; mkdir /tmp/benchmark/junk2")
print("Start writing")
channelObj = digital_rf.DigitalRFWriter('/tmp/benchmark/junk2', 'i2', subdir_cadence_secs, file_cadence_millisecs,
                                        start_global_index, SAMPLE_RATE_NUMERATOR, SAMPLE_RATE_DENOMINATOR, 'Fake_uuid', 9, True)
t = time.time()
for i in range(N_WRITES):
    channelObj.rf_write(data_int16)
channelObj.close()
seconds = time.time() - t
speedMB = (N_WRITES * 4 * WRITE_BLOCK_SIZE) / (1.0E6 * seconds)
print('Total time %i seconds, speed %1.2f MB/s' % (int(seconds), speedMB))
